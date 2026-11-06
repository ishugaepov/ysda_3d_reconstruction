"""Run all per-stage diagnostics on a saved DA3 run (no ComfyUI server needed).

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/diagnose_offline.py outputs/toy_da3.npz outputs/toy_diag
"""
import json
import os
import sys

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz, prefix = sys.argv[1], sys.argv[2]
max_viol = int(sys.argv[3]) if len(sys.argv) > 3 else 1
depth_tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from comfy_extras.nodes_gaussian_splat import _surface_nets  # noqa: E402
from ysda3d import diagnostics, recon  # noqa: E402

torch.set_grad_enabled(False)
d = np.load(npz)
images, masks = torch.from_numpy(d["images"]), torch.from_numpy(d["masks"])
depth, conf = torch.from_numpy(d["depth"]), torch.from_numpy(d["confidence"])
S, H, W = depth.shape
K = (torch.from_numpy(d["intrinsics"]).float() if "k_canvas" in d.files else
     recon.rescale_intrinsics(torch.from_numpy(d["intrinsics"]), recon.da3_processing_size(H, W, int(d["da3_res"])), (H, W)))
E = torch.from_numpy(d["extrinsics"]).float()
dist = torch.from_numpy(d["k1"]) if "k1" in d.files else None
keep = masks >= 0.5


def save_grid(batch, path, cols=4, scale=0.25):
    b = batch.numpy()
    n, h, w, _ = b.shape
    rows = (n + cols - 1) // cols
    g = np.ones((rows * h, cols * w, 3), np.float32)
    for k in range(n):
        g[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = b[k]
    im = Image.fromarray((g * 255).clip(0, 255).astype(np.uint8))
    im.resize((int(im.width * scale), int(im.height * scale))).save(path, quality=85)


tiles, stats, (pts, cols_) = diagnostics.consistency_check(depth, K, E, images, keep, dist=dist)
save_grid(tiles, prefix + "_consistency.jpg", cols=3, scale=0.2)
diagnostics.write_point_cloud_ply(prefix + "_fused.ply", pts, cols_, recon.camera_centers(E))
for s in stats:
    print(s)
meds = [s["median_rel_depth_err"] for s in stats]
print("median over views:", round(float(np.median(meds)), 4))

verts, faces, colors, report = recon.reconstruct(
    depth, K, E, images, masks, lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu")), conf=conf,
    max_violations=max_viol, depth_tol=depth_tol, dist=dist)
sil, ious = diagnostics.silhouette_check(verts, faces, K, E, masks, dist=dist)
save_grid(sil, prefix + "_silhouettes.jpg", cols=7, scale=0.2)
print("silhouette IoU per view:", ious, "mean", round(float(np.mean(ious)), 3))
json.dump({"consistency": stats, "silhouette_iou": ious, "report": report}, open(prefix + "_diag.json", "w"), indent=1)
