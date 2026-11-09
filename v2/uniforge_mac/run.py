"""Run UniForge3D on Apple Silicon (MPS) for one object.

  ~/src/uniforge3d-venv/bin/python v2/uniforge_mac/run.py <folder with RGBA PNGs> <out.glb> [--seed 42]

Images are used in sorted order; the first is the anchor view. Logs per-stage times and saves the
Shape-512 mesh next to the output (for debugging).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch  # noqa: E402

patch.apply()
os.environ.setdefault("YSDA_DINO_PATH", os.path.expanduser("~/Downloads/dinoV3"))

import torch  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="mps")
ap.add_argument("--stop-after", choices=["structure", "shape512", "all"], default="all",
                help="shape512 = geometry-only preview (~1/5 of the time)")
ap.add_argument("--status-json", help="write per-stage progress/timings here (read by the ComfyUI node)")
args = ap.parse_args()

MacPipeline = patch.patch_pipeline()
T0 = time.time()
pipe = MacPipeline.from_pretrained("xibushu/UniForge3D", device=args.device, local_files_only=True)
images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in {".png", ".webp"})
print(f"{len(images)} views, anchor = {images[0].name}", flush=True)


STATUS = {"stages": [], "state": "running"}


def write_status(**kw):
    STATUS.update(kw)
    if args.status_json:
        import json
        tmp = args.status_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(STATUS, f, indent=1)
        os.replace(tmp, args.status_json)


def stage(name, fn, *a):
    write_status(current=name)
    t = time.time()
    out = fn(*a)
    if args.device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t
    STATUS["stages"].append({"name": name, "seconds": round(dt, 1)})
    write_status(current=None)
    print(f"[{time.time() - T0:7.1f}s] {name}: {dt:.1f}s", flush=True)
    return out


with torch.no_grad():
    torch.manual_seed(args.seed)
    batch = stage("preprocess", pipe._preprocess_images, images)
    dino = stage("dino", pipe._encode_dino, batch)
    structure = stage("structure", pipe._generate_structure, batch, dino["sparse_structure"])
    print("   coords32:", tuple(structure.coords32.shape), "coords64:", tuple(structure.coords64.shape), flush=True)
    if args.stop_after == "structure":
        write_status(state="done", total_seconds=round(time.time() - T0, 1)); sys.exit(0)
    shape512 = stage("shape512", pipe._generate_shape512, batch, structure, dino["shape512"])
    shape512_path = os.path.splitext(args.out)[0] + "_shape512.glb"
    shape512.export(shape512_path)
    write_status(shape512=shape512_path)
    print(f"   shape512 mesh: {len(shape512.vertices)} verts, {len(shape512.faces)} faces", flush=True)
    if args.stop_after == "shape512":
        write_status(state="done", total_seconds=round(time.time() - T0, 1)); sys.exit(0)
    coords64 = stage("coords64", pipe._shape512_to_coords64, shape512)
    latent1024, views = stage("shape1024", pipe._generate_shape1024, batch, coords64, dino["shape1024"])
    textured = stage("texture", pipe._generate_texture, batch, latent1024, views, dino["shape1024"])
    textured.export(args.out)
    write_status(state="done", output=args.out, vertices=len(textured.vertices), faces=len(textured.faces),
                 total_seconds=round(time.time() - T0, 1))
    print(f"done: {args.out} ({len(textured.vertices)} verts) total {time.time() - T0:.0f}s", flush=True)
