# 0011 — v2 in ComfyUI: UniForge3D workflow (subprocess node)

- **Date:** 2026-11-09
- **Status:** done offline (node test); the Desktop run is pending a restart (new nodes)
- **Stage(s):** infra
- **Builds on:** #0009, #0010

## What
A ComfyUI workflow that runs the v2 pipeline end to end, as the base for the planned iterations (v2.2 registration + held-out evaluation, v2.3 re-texturing, NBV).

```
YSDA Load Image Folder → Remove Background (BiRefNet) → [masks preview]
  → YSDA Prepare UniForge3D Inputs (v2)   full-resolution RGBA views; explicit list (first = anchor) or K evenly spaced
  → YSDA UniForge3D (v2, Mac)             subprocess: ~/src/uniforge3d-venv + v2/uniforge_mac/run.py
       → Save 3D Model (final GLB) · Save 3D Model (shape-512 geometry) · report (stage timings)
       → MESH output (native ComfyUI mesh) for the next iterations
```

## Design
- **Why a subprocess:** UniForge3D needs its own dependency set (transformers 4.57, peft, …), and its port patches `torch` globally (autocast, `segment_reduce`). Neither should touch the ComfyUI server. Same pattern as the COLMAP worker (journal 0005).
- **Memory:** ComfyUI's loaded models are freed first (`unload_all_models`); the subprocess needs ~40 GB.
- **Progress:** `run.py --status-json` writes per-stage status; the node maps stages to time weights for ComfyUI's progress bar. Cancelling the prompt kills the subprocess.
- **`mode = preview`** stops after shape 512 (geometry only, ~5 min) for fast iteration on view selection; `full` ≈ 28–33 min.
- Outputs go to `ComfyUI-Shared/output/ysda/uniforge3d/<timestamp>/` (model.glb, model_shape512.glb, status.json, run.log).

## Test (offline, node code called as the server would)
- Prepare (auto K = 4) → views 4505, 4510, 4515, 4521. **The even spread picks the top-down shot (4521)**, so the workflow defaults to the known-good explicit list 4505, 4509, 4513, 4518. View selection from COLMAP poses is a v2.1 follow-up.
- UniForge3D preview: **283 s** (dino 5.5 s, structure 64 s, shape 512 102 s + model loading). Native MESH with 293k vertices returned.

## Next
- Run `ysda_photos_to_3d_v2_uniforge3d` in Desktop (full mode).
- v2.2: registration node (UniForge3D mesh → COLMAP frame) + held-out photometric evaluation, built on this workflow's MESH output.
