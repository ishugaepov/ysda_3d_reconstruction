"""Run ComfyUI's native background removal + Depth Anything 3 on a photo folder, outside the server.

Lets us iterate on ysda3d without restarting Comfy Desktop. Saves everything the reconstruction
needs to an .npz. Run with ComfyUI's own venv python:

  ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/run_da3_offline.py data/toy/images out.npz
"""
import argparse
import os
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")
DESKTOP_MODEL_PATHS = os.path.expanduser(
    "~/Library/Application Support/Comfy Desktop/instance-model-paths")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("out")
ap.add_argument("--max-side", type=int, default=1024)
ap.add_argument("--da3-res", type=int, default=504)
ap.add_argument("--pose-method", default="cam_dec")
args = ap.parse_args()
sys.argv = sys.argv[:1]  # keep ComfyUI's own argparse quiet

sys.path[:0] = [COMFY, REPO]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import folder_paths  # noqa: E402,F401
from utils.extra_config import load_extra_path_config  # noqa: E402

for f in os.listdir(DESKTOP_MODEL_PATHS):
    if f.endswith(".yaml"):
        load_extra_path_config(os.path.join(DESKTOP_MODEL_PATHS, f))

from comfy_extras.nodes_bg_removal import LoadBackgroundRemovalModel, RemoveBackground  # noqa: E402
from comfy_extras.nodes_depth_anything_3 import LoadDA3Model, DA3Inference  # noqa: E402
from ysda3d.images import load_folder  # noqa: E402
from ysda3d.comfy_compat import memory_efficient_da3_attention  # noqa: E402


def first(out):
    return out.result[0] if hasattr(out, "result") else out[0]


torch.set_grad_enabled(False)  # the server runs nodes in inference mode; without this memory explodes
t0 = time.time()
images, valid, names = load_folder(args.folder, args.max_side)
img_t = torch.from_numpy(images)
print(f"loaded {len(names)} images -> batch {tuple(img_t.shape)}  ({time.time()-t0:.1f}s)")

bg = first(LoadBackgroundRemovalModel.execute("birefnet.safetensors"))
masks = torch.cat([first(RemoveBackground.execute(bg, img_t[i:i + 1])) for i in range(len(names))])
masks = masks * torch.from_numpy(valid)
print(f"masks {tuple(masks.shape)}  ({time.time()-t0:.1f}s)")

model = first(LoadDA3Model.execute("depth_anything_3_base.safetensors", "fp32"))
mode = {"mode": "multiview", "ref_view_strategy": "saddle_balanced", "pose_method": args.pose_method}
with memory_efficient_da3_attention():
    geom = first(DA3Inference.execute(model, img_t, args.da3_res, "upper_bound_resize", mode))
print(f"DA3 done ({time.time()-t0:.1f}s); keys {sorted(geom)}")

np.savez_compressed(
    args.out, names=np.array(names), images=images, valid=valid, masks=masks.cpu().numpy(),
    depth=geom["depth"].cpu().numpy(), confidence=geom["confidence"].cpu().numpy(),
    intrinsics=geom["intrinsics"][0].cpu().numpy(), extrinsics=geom["extrinsics"][0].cpu().numpy(),
    da3_res=args.da3_res)
print("saved", args.out)
