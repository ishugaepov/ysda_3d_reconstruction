"""Disk cache for UniForge3D runs (≈ 35 min each): a finished run is reused when its inputs match.

The inputs are RGBA images (YSDA Prepare UniForge3D Inputs). Their RGB comes straight from the photos,
but their alpha is a background-removal mask, which is not guaranteed to be bit-identical between
ComfyUI sessions (BiRefNet on MPS). So RGB must match exactly, alpha up to a few stray pixels.
Everything else that changes the result is in the key: mode, seed and the Mac port's source code.

Register a run made before the cache existed:
  python -m ysda3d.runcache <run_dir> <input_dir> [--mode full] [--seed 42]
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from PIL import Image

MANIFEST = "cache_inputs.json"
ALPHA_DIR = "cache_alpha"
SUFFIXES = (".png", ".webp")        # what v2/uniforge_mac/run.py reads


def code_version(root: str) -> str:
    """Hash of the .py sources under root: editing the port invalidates its cached runs."""
    h = hashlib.sha256()
    for dirpath, dirnames, files in os.walk(root):
        dirnames.sort()
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                h.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:16]


def fingerprint(input_dir: str):
    """(per-view records: name, shape, RGB hash; alpha arrays) of the input images, in file order."""
    views, alphas = [], []
    for f in sorted(os.listdir(input_dir)):
        if not f.lower().endswith(SUFFIXES):
            continue
        a = np.asarray(Image.open(os.path.join(input_dir, f)).convert("RGBA"))
        views.append({"name": f, "shape": list(a.shape[:2]),
                      "rgb_sha256": hashlib.sha256(np.ascontiguousarray(a[..., :3]).tobytes()).hexdigest()})
        alphas.append(np.ascontiguousarray(a[..., 3]))
    return views, alphas


def _alpha_path(run_dir: str, name: str) -> str:
    return os.path.join(run_dir, ALPHA_DIR, os.path.splitext(name)[0] + ".png")


def save(run_dir: str, key: dict, views, alphas):
    os.makedirs(os.path.join(run_dir, ALPHA_DIR), exist_ok=True)
    for v, a in zip(views, alphas):
        Image.fromarray(a).save(_alpha_path(run_dir, v["name"]))
    with open(os.path.join(run_dir, MANIFEST), "w") as fh:
        json.dump({"key": key, "views": views}, fh, indent=1)


def find(runs_root: str, key: dict, views, alphas, level: int = 32, max_frac: float = 1e-4):
    """Newest finished run under runs_root with the same key, photos and masks; None if there is none.

    A mask matches if at most max_frac of its pixels differ by more than `level` (of 255).
    """
    if not os.path.isdir(runs_root):
        return None
    for name in sorted(os.listdir(runs_root), reverse=True):
        run = os.path.join(runs_root, name)
        try:
            with open(os.path.join(run, MANIFEST)) as fh:
                man = json.load(fh)
            with open(os.path.join(run, "status.json")) as fh:
                st = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if st.get("state") != "done" or man.get("key") != key or man.get("views") != views:
            continue
        files = [p for p in (st.get("output"), st.get("shape512")) if p]     # preview runs have no "output"
        if not files or not all(os.path.exists(p) for p in files):
            continue
        if all(alpha_mismatch(_alpha_path(run, v["name"]), a, level) <= max_frac for v, a in zip(views, alphas)):
            return run
    return None


def alpha_mismatch(path: str, alpha: np.ndarray, level: int = 32) -> float:
    """Fraction of pixels whose alpha differs from the stored one by more than `level` (1.0 if missing)."""
    if not os.path.exists(path):
        return 1.0
    ref = np.asarray(Image.open(path))
    if ref.shape != alpha.shape:
        return 1.0
    return float((np.abs(ref.astype(np.int16) - alpha) > level).mean())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Register a finished UniForge3D run in the cache")
    p.add_argument("run_dir")
    p.add_argument("input_dir", help="the RGBA folder that run was made from")
    p.add_argument("--mode", default="full")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    port = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2", "uniforge_mac")
    v, al = fingerprint(a.input_dir)
    save(a.run_dir, {"mode": a.mode, "seed": a.seed, "code": code_version(port)}, v, al)
    print(f"registered {a.run_dir}: {len(v)} views, code {code_version(port)}")
