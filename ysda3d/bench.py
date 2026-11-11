"""Benchmark over several examples (05_benchmark_plan.md, journal 0016): registry, data fetching, caching.

Examples (bench/examples.json):
  - real: our toy photos, or frames of a CO3D-small sequence (Hugging Face XeemPad/CO3D-small, CC BY-NC 4.0).
    Scored on held-out photos only.
  - synthetic: a Google Scanned Objects mesh (hugging-3d/google-scanned-objects, CC BY 4.0), rendered as a
    handheld capture by ysda3d.synth. Ground-truth cameras, masks and mesh add geometry metrics.
Data lives under $YSDA_BENCH_ROOT (default ~/datasets/ysda_bench), never in the repo:
  examples/<name>/photos, meta.json, gt/    the capture
  work/<name>/                              DA3 + masks, COLMAP, generator inputs (pipeline stages)
  uniforge3d_runs/                          generated meshes (ysda3d.runcache: reused when inputs match)
  cache/<name>/                             per-method intermediates (keyed by inputs + code)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.expanduser(os.environ.get("YSDA_BENCH_ROOT", "~/datasets/ysda_bench"))
HF = "https://huggingface.co"
CO3D_REPO = "XeemPad/CO3D-small"
GSO_REPO = "hugging-3d/google-scanned-objects"


def load_examples(names=None):
    cfg = json.load(open(os.path.join(REPO, "bench", "examples.json")))
    exs = [dict(cfg["defaults"], **e) for e in cfg["examples"]]
    if not names:
        exs = [e for e in exs if "disabled" not in e]       # disabled examples run only when named explicitly
    if names:
        missing = set(names) - {e["name"] for e in exs}
        if missing:
            raise ValueError(f"unknown examples: {sorted(missing)}")
        exs = [e for e in exs if e["name"] in names]
    return exs


def example_dir(ex):
    return os.path.join(ROOT, "examples", ex["name"])


def work_dir(ex):
    d = os.path.join(ROOT, "work", ex["name"])
    os.makedirs(d, exist_ok=True)
    return d


def photos_dir(ex):
    return os.path.join(REPO, ex["photos"]) if "photos" in ex else os.path.join(example_dir(ex), "photos")


def meta(ex):
    """Held-out names (+ ground truth for synthetic examples)."""
    if "photos" in ex:
        return {"kind": ex["kind"], "heldout": ex["heldout"]}
    return json.load(open(os.path.join(example_dir(ex), "meta.json")))


# ------------------------------------------------------------------------------------------- downloads
def _get(url: str, dst: str):
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    os.replace(tmp, dst)


def _hf_tree(repo: str, path: str):
    with urllib.request.urlopen(f"{HF}/api/datasets/{repo}/tree/main/{path}", timeout=60) as r:
        return json.load(r)


def fetch_co3d(ex, log=print):
    """n evenly spaced frames of one CO3D sequence (one orbit of a phone video), some of them held out."""
    d = example_dir(ex)
    if os.path.exists(os.path.join(d, "meta.json")):
        return
    seq = f"co3d_subset20k/{ex['co3d']}"
    frames = sorted(x["path"] for x in _hf_tree(CO3D_REPO, f"{seq}/images") if x["path"].endswith(".jpg"))
    picks = np.linspace(0, len(frames) - 1, ex["n_frames"]).round().astype(int)
    os.makedirs(os.path.join(d, "photos"), exist_ok=True)
    names = []
    for i in picks:
        name = os.path.basename(frames[i])
        _get(f"{HF}/datasets/{CO3D_REPO}/resolve/main/{frames[i]}", os.path.join(d, "photos", name))
        names.append(os.path.splitext(name)[0])
    heldout = [names[i] for i in ex["heldout_positions"]]
    json.dump({"kind": "real", "source": f"{CO3D_REPO}/{seq}", "license": "CC BY-NC 4.0 (CO3D)",
               "frames": names, "heldout": heldout}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    log(f"   {ex['name']}: {len(names)} frames of {seq} ({len(frames)} available), held out {heldout}")


def gso_object(ex):
    """(glb path, metadata) of a GSO object, downloading its shard (~0.5 GB, ~24 objects) once."""
    sd = os.path.join(ROOT, f"gso_shard{ex['gso_shard']}")
    tar = os.path.join(sd, f"gso-train-{ex['gso_shard']:05d}.tar")
    if not os.path.exists(tar):
        os.makedirs(sd, exist_ok=True)
        _get(f"{HF}/datasets/{GSO_REPO}/resolve/main/data/{os.path.basename(tar)}", tar)
    ext = os.path.join(sd, "extracted")
    glb, js = os.path.join(ext, ex["gso"] + ".glb"), os.path.join(ext, ex["gso"] + ".json")
    if not (os.path.exists(glb) and os.path.exists(js)):
        os.makedirs(ext, exist_ok=True)
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if os.path.basename(m.name) in (ex["gso"] + ".glb", ex["gso"] + ".json"):
                    m.name = os.path.basename(m.name)                  # flat, no path tricks
                    t.extract(m, ext)
    return glb, json.load(open(js))


def fetch_gso(ex, load_glb, log=print):
    """Render a GSO object as a handheld capture (ysda3d.synth): photos, GT cameras/masks/mesh, meta.json."""
    from PIL import Image
    from . import synth
    d = example_dir(ex)
    if os.path.exists(os.path.join(d, "meta.json")):
        return
    glb, info = gso_object(ex)
    v, f, uv, tex = load_glb(glb)
    rng = np.random.default_rng(ex["scene_seed"])
    v = synth.place_object(v, max(info["glb_processing"]["orig_extents"]), yaw_deg=float(rng.uniform(0, 360)))
    scene = synth.Scene(v, f, uv, tex, seed=ex["scene_seed"])
    target = np.array([0.0, float(v[:, 1].max()) / 2, 0.0])
    radius = float(np.linalg.norm(v.max(0) - v.min(0))) / 2
    W, H = 1152, 864
    K, cams = synth.capture_cameras(target, radius, seed=ex["scene_seed"], W=W, H=H)
    for sub in ("photos", "gt/masks"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    cam_meta = {}
    for name, R, t, role in cams:
        img, mask, _ = scene.render(K, R, t, W, H)
        img = np.clip(img + rng.normal(0, 1.5 / 255, img.shape), 0, 1)        # sensor noise, then JPEG
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "photos", name + ".jpg"), quality=92)
        Image.fromarray((mask * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "gt", "masks", name + ".png"))
        cam_meta[name] = {"role": role, "R": R.tolist(), "t": t.tolist()}
    np.savez_compressed(os.path.join(d, "gt", "mesh.npz"), v=v.astype(np.float32), f=f.astype(np.int64))
    json.dump({"kind": "synthetic", "source": f"{GSO_REPO}: {ex['gso']}", "license": "CC BY 4.0 (Google Scanned Objects)",
               "K": K.tolist(), "W": W, "H": H, "cameras": cam_meta,
               "heldout": [n for n, c in cam_meta.items() if c["role"] != "t"],
               "heldout_low": [n for n, c in cam_meta.items() if c["role"] == "h_low"]},
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    log(f"   {ex['name']}: rendered {len(cams)} views of {ex['gso']} "
        f"({np.round(v.max(0) - v.min(0), 3).tolist()} m)")


# ------------------------------------------------------------------------------------------- caching
def file_hash(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def code_hash(*modules) -> str:
    """Hash of ysda3d module sources: a stage's cache entry is invalid once its code changes."""
    return file_hash(*[os.path.join(REPO, "ysda3d", m + ".py") for m in modules])


def select_views(dirs: np.ndarray, areas: np.ndarray, k: int):
    """Generator inputs: the view with the largest object (anchor = front), then farthest-point sampling
    of the viewing directions (unit vectors object -> camera), so K views spread around the object."""
    picks = [int(np.argmax(areas))]
    while len(picks) < min(k, len(dirs)):
        dmin = np.min(np.arccos(np.clip(dirs @ dirs[picks].T, -1, 1)), axis=1)
        dmin[picks] = -1
        picks.append(int(np.argmax(dmin)))
    return picks
