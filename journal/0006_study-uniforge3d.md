# 0006 — Study: UniForge3D (reconstruction + generation) and its feasibility on our Mac

- **Date:** 2026-11-07
- **Status:** done (study + downloads); experiments start in v2.0
- **Stage(s):** planning, infra
- **Builds on:** #0005; new plan [04_plan_v2_recon_plus_generation.md](../04_plan_v2_recon_plus_generation.md)

## Question
Can UniForge3D (arXiv 2610.15357, code `xibushu/UniForge3D`) become the core of our pipeline for real photos on an M3 Max, without training anything?

## What we learned (paper)
- Joint model in a **shared anchor-guided space**: upright, azimuth from the first image.
  - VGGT (LoRA) and the TRELLIS.2 sparse-structure DiT (LoRA) exchange features in both directions, producing point maps plus a 64³ structure.
  - A structure-conditioned VGGT then gives pixel↔voxel correspondences, and **voxel-wise view routing** (support = largest connected pixel component; argmax over views) conditions TRELLIS.2 shape (512 → 1024) and texture.
- Trained on TRELLIS-500K renders (4 views); evaluated on Toys4K/GSO. GSO F-score 0.89 vs 0.65–0.74 for alternatives.
- Limitation stated by the authors: synthetic training; occlusion, reflections and lighting are hard. Real-world inputs are our job.

## What we learned (code)
- `infer.py`: one folder per object, **any number of images**, first = anchor. RGBA inputs with real alpha **skip** background removal; each view is cropped to its own square alpha box and composited on black.
- Weights load with `torch.load(weights_only=True)` (safe). The models are plain PyTorch.
- **CUDA-only parts:**
  - device hard-coded to cuda;
  - flash-attn default (but `ATTN_BACKEND=sdpa` exists);
  - sparse conv only via FlexGEMM;
  - o-voxel mesh extraction;
  - CuMesh post-processing;
  - nvdiffrast baking.
- ComfyUI's native TRELLIS.2 has **pure-torch** replacements (`comfy/ldm/trellis2/flexgemm.py`, `vae.py`, mesh3d post-processing, `BakeTextureFromVoxel`), so a Mac port looks feasible but is real work.
- External weights:
  - TRELLIS.2-4B (3 ckpts) and TRELLIS-image-large ss decoder: MIT, downloaded.
  - **DINOv3-L is gated (manual approval)**.
  - RMBG-2.0 gated, NC (not needed).
  - UniForge3D weights CC BY-NC 4.0, 27 GB, downloaded.
- ComfyUI's Pixal3D multi-view template expects turnaround-style views, so it's a weaker baseline for arbitrary angles.
- Training-free related work (UMI3D, ReconPlusGen) suggests **Track B**: inject our own reconstruction as TRELLIS.2's sparse structure, natively in ComfyUI.

## Downloads (2026-11-07)
- Code: `~/src/UniForge3D` (commit 8b60947), outside the repo (third-party).
- Hugging Face cache (`~/.cache/huggingface/hub`):
  - `xibushu/UniForge3D` (27 GB);
  - `microsoft/TRELLIS.2-4B` @af44b45 (ss_flow_img_dit_1_3B_64_bf16, shape_dec_next_dc_f16c32_fp16, tex_dec_next_dc_f16c32_fp16);
  - `microsoft/TRELLIS-image-large` @25e0d31 (ss_dec_conv3d_16l8_fp16).
- Earlier, for ComfyUI: native TRELLIS.2 (int8), Hunyuan3D-2.1, Qwen-Image-2.1, TripoSR.
- **Still missing:** DINOv3-L in Hugging Face format (gated).

## Conclusion
Adopt reconstruction + generation as the main track. Plan in `04_plan_v2_recon_plus_generation.md`:
- v2.0 feasibility (native TRELLIS.2 on MPS; port UniForge3D);
- v2.1 real-photo adapter and view selection;
- v2.2 registration to COLMAP and held-out evaluation;
- v2.3 photo-faithful re-texturing;
- v2.4 NBV from support.

Track B (native TRELLIS.2 + our structure) runs in parallel.
