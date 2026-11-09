# UniForge3D on Apple Silicon (MPS)

Runs the official UniForge3D inference (`xibushu/UniForge3D`, code in `~/src/UniForge3D`, unmodified)
on an M-series Mac by injecting pure-torch stand-ins for its CUDA-only dependencies. See journal 0009.

| CUDA dependency | Mac replacement |
|---|---|
| flash-attn (dense, packed, varlen) | `shims/flash_attn`: torch SDPA, query-chunked |
| FlexGEMM submanifold conv | `shims/flex_gemm/ops/spconv.py` (sorted-key hash map; fp32 matmul) |
| FlexGEMM `grid_sample_3d` | `shims/flex_gemm/ops/grid_sample.py` (cell-centre trilinear) |
| o-voxel `flexible_dual_grid_to_mesh` | `shims/o_voxel/convert.py` |
| CuMesh (simplify, cleanup) | `shims/cumesh`: fast-simplification + trimesh; hole filling / remeshing skipped, so the output has cracks, flipped faces and an open bottom. The ComfyUI workflow repairs it downstream (YSDA Repair Generated Mesh, journal 0013) |
| nvdiffrast UV baking | replaced: per-vertex base colour sampled from the PBR voxels |
| `torch.device("cuda")`, autocast("cuda"), `torch.cuda.*`, `segment_reduce` | `patch.py` |
| VGGT conditioner bf16 (autocast only on CUDA) | runs in fp32 (MPS asserts on mixed-dtype matmul) |

## Setup
```bash
uv venv --python 3.12 ~/src/uniforge3d-venv
uv pip install --python ~/src/uniforge3d-venv/bin/python torch torchvision "transformers==4.57.3" "peft==0.17.1" \
  "huggingface-hub>=0.36,<1" "timm>=1.0,<2" "kornia>=0.8,<0.9" "numpy>=2.1,<2.3" "Pillow>=10.4,<12" \
  "safetensors>=0.7,<1" "trimesh==4.6.3" "opencv-python-headless>=4.10,<5" "easydict>=1.13,<2" tqdm scipy fast-simplification
hf download xibushu/UniForge3D            # + TRELLIS.2-4B / TRELLIS-image-large ckpts (journal 0006)
```
DINOv3-L (gated) is read from `$YSDA_DINO_PATH` (default `~/Downloads/dinoV3`).

## Run
```bash
~/src/uniforge3d-venv/bin/python v2/uniforge_mac/run.py <folder of RGBA PNGs> out.glb
```
Inputs must be RGBA with a real alpha mask (the gated RMBG remover is not used). The first image (sorted) is the anchor.
