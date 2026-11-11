"""Prepare benchmark examples: fetch or render the capture, then the expensive pipeline stages, each cached.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python -I scripts/bench_prepare.py [--examples a,b] [--no-generate]

Stages per example (skipped when their output exists):
  1. capture: CO3D frames / GSO render                      -> examples/<name>/
  2. masks (BiRefNet) + DA3, as scripts/run_da3_offline.py  -> work/<name>/da3.npz
  3. COLMAP cameras, as scripts/refine_offline.py           -> work/<name>/sfm.npz
  4. generator inputs: K = 8 training views (explicit list, or anchor + farthest viewing directions) as RGBA
  5. UniForge3D (~30 min; ysda3d.runcache reuses a finished run with the same inputs) -> work/<name>/generation.json
Stages 1-4 run for all examples first, then the generation queue: reconstruction-only methods can be
evaluated while the generator works through the queue. Benchmark data never goes into the repo.
"""
import argparse
import json
import os
import subprocess
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--examples", default="", help="comma-separated names (default: all)")
ap.add_argument("--no-generate", action="store_true", help="stop before UniForge3D")
args = ap.parse_args()
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
from ysda3d import bench, runcache  # noqa: E402
from ysda3d import images as yimages  # noqa: E402

PY = sys.executable
UNIFORGE_PYTHON = os.environ.get("YSDA_UNIFORGE_PYTHON", os.path.expanduser("~/src/uniforge3d-venv/bin/python"))
PORT = os.path.join(REPO, "v2", "uniforge_mac")
RUNS = [os.path.join(bench.ROOT, "uniforge3d_runs"), os.path.expanduser("~/ComfyUI-Shared/output/ysda/uniforge3d")]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_glb(path):
    from comfy_api.latest._util.geometry_types import File3D
    from comfy_extras.nodes_mesh_io import Get3DComponents
    m = Get3DComponents.execute(File3D(path)).result[0]
    return (m.vertices[0].numpy().astype(np.float64), m.faces[0].numpy().astype(np.int64),
            m.uvs[0].numpy().astype(np.float64), m.texture[0].numpy())


def run(cmd, logfile, env=None):
    with open(logfile, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[2])} failed (exit {r.returncode}), see {logfile}")


def object_centre(E):
    """Least-squares point closest to all optical axes (the object the photographer kept in view)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for e in E:
        R, t = e[:3, :3], e[:3, 3]
        c, d = -R.T @ t, R[2]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def prepare_recon(ex):
    w = bench.work_dir(ex)
    if ex["kind"] == "real" and "co3d" in ex:
        bench.fetch_co3d(ex, log=log)
    elif ex["kind"] == "synthetic":
        bench.fetch_gso(ex, load_glb, log=log)
    photos = bench.photos_dir(ex)
    da3, sfm = os.path.join(w, "da3.npz"), os.path.join(w, "sfm.npz")
    if not os.path.exists(da3):
        t = time.time()
        run([PY, "-I", os.path.join(REPO, "scripts", "run_da3_offline.py"), photos, da3], os.path.join(w, "da3.log"))
        log(f"   {ex['name']}: masks + DA3 {time.time() - t:.0f}s")
    if not os.path.exists(sfm):
        t = time.time()
        run([PY, "-I", os.path.join(REPO, "scripts", "refine_offline.py"), da3, photos, sfm], os.path.join(w, "sfm.log"),
            env=dict(os.environ, YSDA_SFM_WORK=os.path.join(w, "sfm_work")))
        log(f"   {ex['name']}: COLMAP {time.time() - t:.0f}s")
    split = os.path.join(w, "split.json")
    if not os.path.exists(split):
        s = np.load(sfm)
        names = [str(n) for n in s["names"]]
        stem = lambda n: os.path.splitext(n)[0]  # noqa: E731
        held = [n for n in names if stem(n) in bench.meta(ex)["heldout"]]
        train = [n for n in names if n not in held]
        if "generator_views" in ex:
            views = [next(n for n in train if stem(n) == v) for v in ex["generator_views"]]
        else:
            idx = [names.index(n) for n in train]
            E = s["extrinsics"][idx].astype(np.float64)
            x = object_centre(E)
            C = np.stack([-e[:3, :3].T @ e[:3, 3] for e in E])
            dirs = (C - x) / np.linalg.norm(C - x, axis=1, keepdims=True)
            areas = s["masks"][idx].reshape(len(idx), -1).mean(1)
            views = [train[i] for i in bench.select_views(dirs, areas, ex["generator_k"])]
        lost = sorted(set(bench.meta(ex)["heldout"]) - {stem(n) for n in held})
        json.dump({"train": train, "heldout": held, "heldout_not_registered": lost, "generator_views": views},
                  open(split, "w"), indent=1)
        log(f"   {ex['name']}: {len(train)} train / {len(held)} held-out registered (lost {lost}); "
            f"generator views {[stem(v) for v in views]}")
    uf_in = os.path.join(w, "uf_inputs")
    if "generated" not in ex and not os.path.isdir(uf_in):
        d = np.load(da3)
        names = [str(n) for n in d["names"]]
        views = json.load(open(split))["generator_views"]
        yimages.rgba_views_from_masks(photos, names, d["masks"], d["valid"], uf_in, [names.index(v) for v in views])
    if "generated" in ex:                       # precomputed generation: nothing to queue
        prepare_generation(ex)


def prepare_generation(ex):
    w = bench.work_dir(ex)
    out = os.path.join(w, "generation.json")
    if os.path.exists(out):
        return
    if "generated" in ex:
        json.dump({"glb": os.path.join(REPO, ex["generated"]), "source": "precomputed"}, open(out, "w"), indent=1)
        return
    uf_in = os.path.join(w, "uf_inputs")
    key = {"mode": "full", "seed": ex["seed"], "code": runcache.code_version(PORT)}
    views, alphas = runcache.fingerprint(uf_in)
    run_dir = next((r for r in (runcache.find(root, key, views, alphas) for root in RUNS) if r), None)
    if run_dir is None:
        run_dir = os.path.join(RUNS[0], f"{ex['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir)
        log(f"   {ex['name']}: UniForge3D started (~30 min) -> {run_dir}")
        t = time.time()
        with open(os.path.join(run_dir, "run.log"), "w") as fh:
            r = subprocess.run([UNIFORGE_PYTHON, "-u", os.path.join(PORT, "run.py"), uf_in, os.path.join(run_dir, "model.glb"),
                                "--seed", str(ex["seed"]), "--status-json", os.path.join(run_dir, "status.json")],
                               stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="", HF_HUB_OFFLINE="1"))
        if r.returncode != 0:
            raise RuntimeError(f"UniForge3D failed for {ex['name']}, see {run_dir}/run.log")
        runcache.save(run_dir, key, views, alphas)
        log(f"   {ex['name']}: UniForge3D done in {(time.time() - t) / 60:.0f} min")
    st = json.load(open(os.path.join(run_dir, "status.json")))
    json.dump({"glb": st["output"], "run_dir": run_dir, "source": "uniforge3d"}, open(out, "w"), indent=1)


def attempt(stage, fn, ex):
    """Run a stage; a failing example (e.g. COLMAP can't register its photos) is reported, not fatal."""
    err = os.path.join(bench.work_dir(ex), "error.txt")
    try:
        fn(ex)
        if os.path.exists(err):
            os.remove(err)
        return True
    except Exception as e:  # noqa: BLE001
        open(err, "w").write(f"{stage}: {e}\n")
        log(f"   {ex['name']}: FAILED at {stage}: {e}")
        return False


exs = bench.load_examples([n for n in args.examples.split(",") if n])
ready = []
for ex in exs:
    log(f"{ex['name']}: capture + reconstruction stages")
    if attempt("reconstruction stages", prepare_recon, ex):
        ready.append(ex)
if not args.no_generate:
    for ex in ready:
        log(f"{ex['name']}: generation")
        attempt("generation", prepare_generation, ex)
log("done")
