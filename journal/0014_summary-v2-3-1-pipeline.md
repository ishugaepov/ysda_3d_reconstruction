# 0014 — Summary: how the current pipeline (v2.3.1) works

- **Date:** 2026-11-10
- **Status:** done (summary of #0006, #0009–#0013; no new experiment)
- **Workflow:** `workflows/photos_to_3d_v2_3_fused.ui.json` (Desktop: `ysda_photos_to_3d_v2_3_fused`)

## In one paragraph
Photos of an object from arbitrary angles go in; an upright, textured 3D model (GLB) comes out.
- **Generation for completeness:** a generative model (UniForge3D) builds the complete shape, including parts no photo saw.
- **Reconstruction for faithfulness:** calibrated cameras from the photos place that shape exactly where the real object is. The photos then supply the real colours wherever they see the surface.

Everything runs locally on an M3 Max with open models and no training. On the toy (21 photos), the end-to-end run takes about 45 min, of which UniForge3D is about 32 min. Generation is cached, so later runs take about 12 min.

## Main steps
| Step (workflow group) | Node(s) | What it does |
|---|---|---|
| 0. Load | YSDA Load Image Folder | All photos of a folder: EXIF rotation, resized to 1024 px, padded into one batch (portrait and landscape mixed) |
| A. Masks | Remove Background (BiRefNet) | Object mask per photo, so backgrounds are ignored downstream |
| B1. Depth | YSDA Depth Anything 3 Multiview | Per-photo depth and initial cameras (DA3, fp32, memory-efficient attention) |
| B2. Cameras | YSDA Refine Cameras (COLMAP) | Structure-from-motion on the full-resolution photos. One camera model per lens; radial distortion k1; registrations with < 200 points rejected (17 of 21 photos kept); DA3 depth rescaled to the SfM scale. Accurate cameras are what make photo colours land in the right place |
| G0. Views | YSDA Prepare UniForge3D Inputs | Picks K = 8 photos (the first is the anchor, which becomes the model's front) and writes them as full-resolution RGBA (photo + mask) |
| G. Generation | YSDA UniForge3D (Mac) | UniForge3D in its own venv on MPS. A joint VGGT + TRELLIS.2 model generates a complete 64³ structure, then shape at 512 → 1024 with per-voxel routing to the best-supporting view, then colours. Finished runs are cached on disk |
| F0. Repair | YSDA Repair Generated Mesh | Rebuilds the surface as the outer boundary of a thin band around the generated mesh. Closes cracks, merges overlapping patches, caps the never-photographed bottom, winds faces outward (holes: 442k open edges → 0) |
| F. Fusion | YSDA Fit Generated Mesh To Photos | **Registration:** a 7-DoF similarity (scale, rotation, translation) fitted to the masks of *all* registered photos, starting from a quick reconstruction (floor plane = up, anchor = front). **Colours:** each vertex takes the colour of the photos that see it (mesh z-buffer visibility, angle weighting). Unseen parts keep the generated colour, mapped onto the photo palette |
| Output | Save GLB | Upright model with per-vertex colours and smooth normals, plus preview tiles (photos vs model) and reports |

Results on 3 held-out photos (used by no method):

| | full-mask PSNR | covered PSNR | silhouette IoU |
|---|---|---|---|
| reconstruction only (v1.2) | 17.61 | 19.99 | 0.925 |
| generation only (UniForge3D mesh, registered, with its own colours) | 11.00 | 11.38 | 0.962 |
| **current pipeline (v2.3.1)** | **17.82** | **20.58** | **0.962** |

## How it differs from the original UniForge3D (paper and code)
| | UniForge3D | Our pipeline |
|---|---|---|
| Input | Clean object images. Trained on synthetic renders (4 views); evaluated on GSO/Toys4K renders | Real handheld photos with backgrounds, any number, mixed lenses. 8 go to the generator; *all* are used for cameras and colour |
| Hardware | CUDA only (flash-attn, FlexGEMM, o-voxel, CuMesh, nvdiffrast) | Apple Silicon (MPS), with pure-torch/CPU stand-ins for those libraries. Upstream code and weights are unmodified |
| Output frame | Its own normalised "anchor-guided upright" space | Registered to the real cameras, so the model lines up with every photo; upright from the floor plane |
| Appearance | Generated PBR base colour (lighting removed), baked into a UV texture | Real photo colours where photos see the surface (72%); generated colours harmonised to the photo palette elsewhere; per-vertex colours instead of a UV texture |
| Mesh clean-up | CuMesh hole filling + narrow-band remesh (GPU) | A CPU narrow-band remesh of our own, which also closes the open bottom |
| Evaluation | Synthetic benchmarks (F-score against ground-truth meshes) | Held-out real photos (silhouette IoU, PSNR) |
| Packaging | Command-line script | ComfyUI workflow with a preview per stage, the generator in a subprocess, and an on-disk cache |

## Main advantages
1. **Complete and faithful at once.** Generation fills what no photo saw (back, underside, occluded areas); the photos give true colours and fur. It beats both pure reconstruction and pure generation on held-out photos.
2. **Arbitrary real photo sets, on a laptop.** No turntable, no fixed number of views, no cloud, no training; open models only.
3. **Camera-consistent output.** Because the model is registered to the real cameras, it can be checked against photos it never used. Next-best-view planning can happen in the same frame. The output comes out upright and front-facing.
4. **Robust to real captures.** It handles mixed lenses and distortion, rejects wrong camera registrations, and corrects colour casts and the generator's mesh defects.
5. **Inspectable and quick to iterate.** Every stage has a preview or report in ComfyUI. The 32-minute generation is cached, so downstream changes cost about 12 minutes. Every experiment is journaled.

## Planned improvements
1. **Automatic view selection (v2.1):** choose the anchor and the K views for the generator from the COLMAP poses, instead of a hand-picked list.
2. **Next-best-view (v2.4), the planned extension:** combine photo coverage with the generator's per-voxel support to suggest new camera angles. Show them as frustums plus preview renders of the complete model. 28% of the surface is currently unobserved.
3. **Colour fidelity:** per-photo 2D alignment or small vertex refinement, and exposure compensation. Possibly a UV texture instead of per-vertex colours, and a neutral colour for the never-photographed bottom.
4. **Mesh quality and size:** repair inside the port before its decimation, from the full-resolution dual-grid mesh, as UniForge3D does. Decimate the export to ~0.3–1M faces.
5. **Speed:** generation dominates, so try fewer sampling steps, and use the preview mode (~5 min) for fast iterations.
6. **Generality:** more objects, including hard ones (the laptop's thin screen, reflective surfaces). Also try Track B (native TRELLIS.2 conditioned on our own reconstruction) as a lighter alternative.

Details per step: journals [0005](0005_v1-colmap-cameras.md) (COLMAP), [0009](0009_v2-0-uniforge3d-on-mac.md) (Mac port), [0010](0010_v2-1-uniforge3d-toy.md) (toy, K = 4/8), [0011](0011_v2-comfyui-workflow.md) (workflow), [0012](0012_v2-2-v2-3-register-recolour.md) (registration + colours), [0013](0013_v2-3-1-repair-generated-mesh.md) (mesh repair).
