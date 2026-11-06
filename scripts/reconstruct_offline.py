"""Reconstruct a coloured mesh from a saved DA3 run, outside ComfyUI, and render checks.

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/reconstruct_offline.py outputs/toy_da3.npz outputs/toy_v0
"""
import argparse
import json
import os
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out_prefix")
ap.add_argument("--res", type=int, default=192)
ap.add_argument("--depth-tol", type=float, default=0.02)
ap.add_argument("--max-violations", type=int, default=1)
ap.add_argument("--views", default="0,6,12,18,21", help="indices for render-vs-photo sheet")
args = ap.parse_args()
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from comfy_extras.nodes_gaussian_splat import _surface_nets  # noqa: E402
from comfy_extras.nodes_save_3d import save_glb  # noqa: E402
from ysda3d.recon import da3_processing_size, reconstruct, rescale_intrinsics, to_display  # noqa: E402
from ysda3d.render import render_points  # noqa: E402

torch.set_grad_enabled(False)
d = np.load(args.npz)
images = torch.from_numpy(d["images"])
masks = torch.from_numpy(d["masks"])
depth = torch.from_numpy(d["depth"])
conf = torch.from_numpy(d["confidence"])
S, H, W = depth.shape
K = rescale_intrinsics(torch.from_numpy(d["intrinsics"]), da3_processing_size(H, W, int(d["da3_res"])), (H, W))
E = torch.from_numpy(d["extrinsics"]).float()


def surface_fn(vol, level, voxel, origin):
    return _surface_nets(vol, level, voxel, origin, torch.device("cpu"))


t0 = time.time()
verts, faces, colors, report = reconstruct(depth, K, E, images, masks, surface_fn, res=args.res,
                                           depth_tol=args.depth_tol, max_violations=args.max_violations,
                                           conf=conf)
report["seconds"] = round(time.time() - t0, 1)
print(json.dumps(report))
save_glb(to_display(verts), faces, args.out_prefix + ".glb", vertex_colors=colors)
json.dump(report, open(args.out_prefix + "_report.json", "w"), indent=1)

# Render-vs-photo sheet: top row photos, bottom row the reconstruction from the same cameras.
idx = [int(i) for i in args.views.split(",") if int(i) < S]
tiles = []
for i in idx:
    r = render_points(verts, colors, K[i], E[i], (H, W), radius=1)
    tiles.append(np.concatenate([d["images"][i], r], 0))
sheet = (np.concatenate(tiles, 1) * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(sheet).resize((sheet.shape[1] // 3, sheet.shape[0] // 3)).save(args.out_prefix + "_compare.jpg", quality=85)
print("saved", args.out_prefix + ".glb", args.out_prefix + "_compare.jpg")
