"""Refine cameras of a saved DA3 run with COLMAP SfM and save a new .npz (intrinsics already on canvas).

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/refine_offline.py outputs/toy_da3.npz data/toy/images outputs/toy_sfm.npz
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ysda3d import recon, sfm  # noqa: E402

src, folder, out = sys.argv[1], sys.argv[2], sys.argv[3]
max_side = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1024
torch.set_grad_enabled(False)
d = np.load(src)
names = [str(n) for n in d["names"]]
depth = torch.from_numpy(d["depth"])
S, H, W = depth.shape
K = recon.rescale_intrinsics(torch.from_numpy(d["intrinsics"]), recon.da3_processing_size(H, W, int(d["da3_res"])), (H, W))
geom = {"depth": depth, "extrinsics": torch.from_numpy(d["extrinsics"])[None], "ysda_K_canvas": K,
        "intrinsics": torch.from_numpy(d["intrinsics"])[None]}
masks = torch.from_numpy(d["masks"])
t0 = time.time()
g, report = sfm.refine_geometry(geom, names, [os.path.join(folder, n) for n in names],
                                work_dir=os.path.join(REPO, "outputs", "sfm_work"), max_side=max_side,
                                masks=masks, drop_unregistered="--keep-unregistered" not in sys.argv)
report["seconds"] = round(time.time() - t0, 1)
print(json.dumps(report, indent=1))
keep = g["ysda_view_index"]
arrs = {k: (d[k][keep] if d[k].ndim and d[k].shape[0] == S else d[k]) for k in d.files}
arrs.update(depth=g["depth"].numpy(), intrinsics=g["intrinsics"][0].numpy(), extrinsics=g["extrinsics"][0].numpy(),
            k_canvas=True, k1=g["ysda_k1"].numpy())
np.savez_compressed(out, **arrs)
print("saved", out)
