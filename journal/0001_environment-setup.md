# 0001 — Environment: ComfyUI + comfy-mcp + Depth Anything 3 on Mac

- **Date:** 2026-11-02
- **Status:** done
- **Stage(s):** infra, B
- **Git commit:** (pre-repo)

## Question / hypothesis
Can Claude drive the local Comfy Desktop reliably, and does native Depth Anything 3 multi-view run on an M3 Max?

## Setup
- Comfy Desktop 1.1.6 runs ComfyUI 0.39.2, Python 3.13, PyTorch 2.12.1 on MPS, at 127.0.0.1:8188.
- comfy-cli 1.24.0 (`~/.local/bin/comfy`) and comfy-mcp 0.10.0 (39 tools).
- Models:
  - `geometry_estimation/depth_anything_3_base.safetensors` (Comfy-Org/Depth-Anything-3, Apache-2.0);
  - `background_removal/birefnet.safetensors`.
- Test: 2-view Sydney Opera House example from the DA3 repo (`input/da3_test_soh`).
- Network: the Apple Claude Code proxy allowlist needed github.com, raw/objects/release-assets.githubusercontent.com, huggingface.co, xethub.hf.co, **hf.co** (Xet data comes from `us.aws.cdn.hf.co`), and api.comfy.org.

## Results
- Claude → UI round trip works: `claude_test_crop` (LoadImage → ImageCrop → SaveImage) runs both via comfy-mcp and from the Desktop UI.
- DA3 multi-view with default weight dtype **crashed the ComfyUI process**: an MPS matmul assertion in `comfy/ldm/depth_anything_3/camera.py`.
- With `weight_dtype=fp32` it runs in **3.4 s** for 2 views, and the depth looks plausible. Workflow: `~/ComfyUI-Shared/claude-workflows/da3_multiview_smoke_fp32.api.json`.
- `DA3GeometryToPointCloud` exists in ComfyUI's source but is **not registered**. Multi-view fusion has to be our own node.
- DA3 multi-view (with camera poses) is only supported for **Small/Base** checkpoints.

## Conclusion
The environment is usable.

**Always load DA3 in fp32 on MPS.**

## Next
- First custom node: fuse DA3 depth and poses into one point cloud.
- Verify DA3's extrinsic convention first.
