"""Compare reconstruction/colouring variants on held-out photos (offline, from a refined .npz).

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/eval_variants.py outputs/toy_sfm.npz outputs/v11

Held-out photos are excluded from reconstruction and used only for evaluation (masked PSNR, coverage,
silhouette IoU), so better numbers mean better on photos the method did not see.
"""
import json
import os
import sys

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz, prefix = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
sys.path[:0] = [COMFY, REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from comfy_extras.nodes_gaussian_splat import _surface_nets  # noqa: E402
from ysda3d import diagnostics, recon  # noqa: E402

torch.set_grad_enabled(False)
d = np.load(npz)
names = [str(n) for n in d["names"]]
images, masks = torch.from_numpy(d["images"]), torch.from_numpy(d["masks"])
depth, conf = torch.from_numpy(d["depth"]), torch.from_numpy(d["confidence"])
K, E = torch.from_numpy(d["intrinsics"]).float(), torch.from_numpy(d["extrinsics"]).float()
dist = torch.from_numpy(d["k1"]) if "k1" in d.files else None
S = len(names)
held = sorted({round(x) for x in np.linspace(1, S - 2, 3)})
train = [i for i in range(S) if i not in held]
print("held-out:", [names[i] for i in held])

VARIANTS = {
    "A_v1_baseline":        dict(depth_tol=0.02, max_violations=1, visibility="da3", color_power=2, mask_erode_px=0),
    "B_depthcarve_meshvis": dict(depth_tol=0.02, max_violations=1, visibility="mesh", color_power=2, mask_erode_px=0),
    "C_hull_meshvis":       dict(depth_tol=10.0, max_violations=1, visibility="mesh", color_power=2, mask_erode_px=0),
    "D_hull_meshvis_p6":    dict(depth_tol=10.0, max_violations=1, visibility="mesh", color_power=6, mask_erode_px=0),
    "E_hull_meshvis_p6_e3": dict(depth_tol=10.0, max_violations=1, visibility="mesh", color_power=6, mask_erode_px=3),
    "F_strict_p6_e3":       dict(depth_tol=10.0, max_violations=0, visibility="mesh", color_power=6, mask_erode_px=3),
    "G_carve05_meshvis":    dict(depth_tol=0.05, max_violations=1, visibility="mesh", color_power=2, mask_erode_px=0),
    "H_carve05_bestview":   dict(depth_tol=0.05, max_violations=1, visibility="mesh", color_power=50, mask_erode_px=0),
    "I_strict_bestview":    dict(depth_tol=10.0, max_violations=0, visibility="mesh", color_power=50, mask_erode_px=0),
    "J_carve02_bestview":   dict(depth_tol=0.02, max_violations=1, visibility="mesh", color_power=50, mask_erode_px=0),
    "K_carve05_strict_bv":  dict(depth_tol=0.05, max_violations=0, visibility="mesh", color_power=50, mask_erode_px=0),
}
only = os.environ.get("VARIANTS")
results = {}
sub = lambda t: t[train]  # noqa: E731
for name, kw in VARIANTS.items():
    if only and name not in only.split(","):
        continue
    verts, faces, colors, rep = recon.reconstruct(
        sub(depth), sub(K), sub(E), sub(images), sub(masks),
        lambda v, l, vx, o: _surface_nets(v, l, vx, o, torch.device("cpu")),
        conf=sub(conf), dist=sub(dist) if dist is not None else None, log=lambda *a: None, **kw)
    pm, tiles = diagnostics.photometric_check(verts, faces, colors, K, E, images, masks, held, dist=dist)
    _, ious = diagnostics.silhouette_check(verts, faces, K[held], E[held], masks[held],
                                           dist=dist[held] if dist is not None else None)
    r = {"heldout_psnr_full_mask_mean": round(float(np.mean([x["psnr_full_mask"] for x in pm])), 2),
         "heldout_psnr_mean": round(float(np.mean([x["psnr"] for x in pm])), 2),
         "heldout_coverage_mean": round(float(np.mean([x["coverage"] for x in pm])), 3),
         "heldout_silhouette_iou_mean": round(float(np.mean(ious)), 3),
         "vertices_seen_fraction": rep["vertices_seen_fraction"], "per_view": pm}
    results[name] = r
    print(f"{name:24s} PSNRfull {r['heldout_psnr_full_mask_mean']:6.2f}  PSNRcov {r['heldout_psnr_mean']:6.2f}  cov {r['heldout_coverage_mean']:.3f}  "
          f"IoU {r['heldout_silhouette_iou_mean']:.3f}  seen {r['vertices_seen_fraction']:.3f}", flush=True)
    sheet = (torch.cat(list(tiles), 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(sheet).resize((sheet.shape[1] // 4, sheet.shape[0] // 4)).save(f"{prefix}_{name}_heldout.jpg", quality=85)
    from comfy_extras.nodes_save_3d import save_glb
    save_glb(recon.to_display(verts), faces, f"{prefix}_{name}.glb", vertex_colors=colors)
json.dump({"held_out": [names[i] for i in held], "variants": VARIANTS, "results": results},
          open(prefix + "_results.json", "w"), indent=1)
