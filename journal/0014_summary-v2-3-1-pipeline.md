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

## How B (reconstruction) and G (generation) work together
**What each branch produces**
- **G (generation)**, from 8 photos:
  - a complete, sharp mesh with *generated* colours;
  - in the generator's own frame: a normalised box (±0.5), with "up" and "front" as the generator guesses them (front = the anchor photo);
  - it says nothing usable about where the photos' cameras were.
- **B (reconstruction)**, from all 21 photos:
  - **B2 (COLMAP):** pixel-accurate cameras for 17 photos (intrinsics, pose, lens distortion; reprojection error 0.7 px), all in one shared world frame.
  - **B1 (DA3):** a depth map per photo, rescaled to that frame. It provides the floor plane and a rough reconstruction of the object. DA3's own camera poses also cross-check the COLMAP registrations (> 10° disagreement → rejected).

**How they meet** in node F, after the generated mesh is repaired (F0):

```
21 photos ─ A. masks ─┬─ B1 DA3 depth ─ B2 COLMAP ─► cameras of 17 photos, depth at their scale, floor plane ─┐
                      │                                                                                     │
                      └─ G0: 8 photos ─ G UniForge3D ─► complete mesh, generator's own frame ─ F0 repair ───┤
                                                                                                            ▼
                     F: 1 initial alignment ─ 2 silhouette fit in all 17 cameras ─ 3 colours from the photos
                                                                                                            ▼
                                                         upright model in the cameras' frame, real colours
```

1. **Initial alignment.** The generator's conventions (upright, front = anchor) are matched to B's floor-plane normal (the true up) and the anchor photo's COLMAP camera (the true front). Size and centre come from the rough reconstruction. 12 trial yaw angles guard against a wrong front.
2. **Silhouette fit.** The generated mesh is projected through each of the 17 COLMAP cameras. Adam adjusts 7 numbers (scale, rotation, translation) until the projection fills each photo's object mask without spilling outside it.
   - Silhouette IoU goes from 0.68 to 0.96, and also reaches 0.957–0.964 on 3 photos the fit never saw.
   - The fit still corrects about 14° of rotation after the initial alignment.
3. **Colour transfer.** With the mesh now in the cameras' frame, every vertex is projected into every photo. The mesh's own depth buffer decides whether that photo really sees the vertex.
   - Seen vertices (72%) take the angle-weighted photo colour.
   - The rest keep the generated colour, mapped onto the photo palette.
4. **Output frame.** B's floor plane and the anchor camera make the output upright and front-facing.
5. **Evaluation** uses B as well. The held-out photos' COLMAP cameras let us render the model exactly as those photos were taken, and compare.

**Why G alone is not enough**
1. **Its colours aren't the object's.** The texture model predicts flat "albedo" colours (no fur, no shading, slightly wrong hues): held-out PSNR is 11.0 against 17.8 with photo colours. Painting photo colours onto the mesh requires knowing where each surface point appears in each photo, i.e. pixel-accurate cameras in the mesh's frame. That is exactly what B provides.
2. **UniForge3D's internal camera estimates can't replace COLMAP.** Internally, UniForge3D predicts point maps (VGGT) for its inputs, but:
   - they are feed-forward estimates. We measured DA3's, which are the same kind: 1.4–1.8% cross-view inconsistency, which blurred colours in v0. Bundle-adjusted COLMAP cameras agree to 0.7 px;
   - they exist only for the 8 input photos;
   - the official pipeline doesn't return them.
3. **Only a few photos fit into G.** It was trained with 4 views; 8 already take 33 min, and memory grows with each view. B uses *all* photos: more colour coverage, and the photos G never saw become a free test set.
4. **G's frame is a guess.** "Up", "front" and size live in a normalised box. B gives the true up (floor), the real camera layout and a consistent scale; the fit above still had to correct about 14°.
5. **Faithfulness has to be checked.** A generator can invent plausible but wrong details. Only with cameras can we render the model from a photo's exact viewpoint and measure the mismatch, so versions are chosen on numbers, not impressions.
6. **Next-best-view needs the same thing:** which surface points the real photos cover, and from where.

**And why B alone is not enough:** reconstruction only knows what was photographed. v1.2 had no underside, a smeared back and a softer face (DA3 depth disagrees across views), with IoU 0.925. G supplies completeness and sharp geometry.

In short: **G answers "what does the whole object look like?"; B answers "where exactly is it in each photo?"** Only the second lets us use the photos' real colours and check the result against reality.

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
