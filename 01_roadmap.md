# 01 — Roadmap: from photos to a textured 3D model

## 1. The problem, stated precisely

**Input:** N photos $\{I_i\}_{i=1}^N$ of a static object.
- Taken from unknown, arbitrary viewpoints.
- Camera intrinsics unknown, or only approximately known from EXIF.
- Lighting is roughly constant.

**Output:**
- a surface $S$ (a triangle mesh);
- a UV parameterisation $u: S \to [0,1]^2$;
- a texture $T: [0,1]^2 \to \mathbb{R}^3$, such that rendering $(S, T)$ with the recovered cameras reproduces the object's pixels in every photo.

The background is excluded.

**Success is judged by appearance fidelity.** Rendered from the photos' own viewpoints, the model should reproduce the object's colours and markings in the right places. Geometric precision matters only as far as it supports this.

Two consequences:
- The texture must be **taken from the photos**, not generated.
- Camera poses must be accurate enough that the projected photos line up. Misaligned poses show up directly as smeared or doubled markings.

The recovery splits into four sub-problems, and every stage below solves one of them:

1. **Segmentation**: which pixels belong to the object.
2. **Camera estimation (structure from motion, SfM)**: the poses $(R_i, t_i)$ and intrinsics $K_i$, recoverable only up to a global similarity transform (rotation, translation, scale).
3. **Dense geometry**: per-pixel depth, fused into one surface.
4. **Appearance**: choose, for each surface point, which photos to take its colour from, then blend them into a seamless texture atlas.

### Design decision: reconstruct, don't generate

There are two families of methods.

- **Reconstruction** (photogrammetry, learned multi-view geometry, Gaussian splatting) is faithful to the photos. It can't fill in regions it never saw.
- **Generation** (TRELLIS.2, Hunyuan3D) produces complete, plausible shapes from 1–4 images, but the result is not *your* object.

We build on **reconstruction**, because:
- faithfulness is the goal;
- arbitrary photo sets only make sense for reconstruction;
- the NBV extension needs camera poses and observation statistics, which only reconstruction provides.

Generation is an **optional gap-filler** for regions that are hard to observe, such as the underside.

## 2. Target architecture

```
photos ─► [A] preprocess ─► [B] cameras + depth ─► [C] dense surface ─► [D] UV + texture ─► GLB
              mask object        (poses K,R,t)         mesh               bake from photos
                                     │                   │
                                     └──────► [E] coverage / NBV ◄─────┘   (extension)
                                                     │
                              [F] optional generative completion (TRELLIS.2)
```

| Stage | v0 (prototype) | Stronger options (later iterations) | Ready-made component |
|---|---|---|---|
| A. Masking | BiRefNet background removal | SAM 3 with text or point prompts; mask consistency across views | Native ComfyUI nodes |
| B. Cameras + depth | **Depth Anything 3** multi-view (poses + depth in one forward pass) | VGGT / MapAnything; refine with **COLMAP/pycolmap** bundle adjustment | DA3 native node; pycolmap wheel |
| C. Surface | Unproject depth → fused point cloud → **TSDF fusion** → marching cubes (Open3D) | Screened Poisson; OpenMVS dense MVS at full resolution; 2DGS/PGSR-style splat-to-mesh | Open3D, OpenMVS |
| D. Texture | UV unwrap (xatlas) + per-face best-view projection | Seam leveling (Let There Be Color!), colour-map optimisation, OpenMVS `TextureMesh` | xatlas, Open3D, OpenMVS |
| E. NBV | — | Per-surface observation quality → candidate views on a sphere → greedy selection → render previews | Custom node (trimesh/Open3D raycasting) |
| F. Completion | — | TRELLIS.2 / Hunyuan3D shape, registered to the reconstruction and textured from the real photos | Native ComfyUI nodes |

### Why Depth Anything 3 for stage B in v0

- It's already native in ComfyUI and runs on the Mac GPU (MPS) in fp32.
- It takes any number of photos with unknown camera positions.
- In one forward pass it returns consistent depth maps, intrinsics and extrinsics.

It replaces three classical stages at once: feature matching, SfM, and depth estimation.

**Known weakness:** it runs at low resolution (around 500 px), so geometry is coarse. Iterations fix this by refining poses with bundle adjustment and running full-resolution stereo.

## 3. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Environment (done) | ComfyUI + comfy-mcp + DA3 working; Claude ↔ UI round trip verified |
| M1 | **Fused point cloud** | All views' DA3 depth unprojected into one coloured, background-free point cloud. Cameras exported and viewable in ComfyUI's 3D preview |
| M2 | **Mesh** | Watertight-ish mesh of the object from TSDF/Poisson; saved as GLB |
| M3 | **Textured mesh (v1 prototype)** | UV atlas baked from the real photos; GLB opens correctly in ComfyUI Preview 3D and Blender |
| M4 | **Evaluation harness** | Metrics on synthetic (known ground truth) and real captures; regressions visible per iteration |
| M5 | **Quality iterations** | Pose refinement (BA), higher-resolution depth/MVS, better texturing (seams, exposure) |
| M6 | **NBV extension** | Coverage heatmap on the mesh plus top-k suggested viewpoints rendered as previews |
| M7 | Optional completion | Generative fill for unobserved regions, clearly marked as hallucinated |

## 4. Evaluation (decided early so iterations are measurable)

**Synthetic data with ground truth:**
- Render views of scanned objects with known meshes and cameras, e.g. **Google Scanned Objects** (CC-BY).
- Measure:
  - camera rotation and translation error, after a similarity alignment (Umeyama);
  - Chamfer distance and F-score between the reconstructed and true mesh;
  - PSNR/SSIM on held-out views.

**Real captures (primary: [`data/toy`](data/README.md)):**
- **Colour fidelity on held-out photos.** This is the main metric. Leave out k photos, reconstruct, render the textured mesh from the held-out cameras, and compare inside the object mask using PSNR, SSIM, LPIPS, and mean colour difference ΔE (CIEDE2000).
- Side-by-side render-vs-photo sheets for visual review, recorded in the journal.
- Reprojection error of the camera poses;
- visual comparison against KIRI / Polycam / Apple Object Capture results for the same photos.

**Coverage:** the fraction of surface area observed by at least 3 views at an incidence angle under 60°. This also feeds NBV.

## 5. Risks and mitigations

| Risk | Why it happens | Mitigation |
|---|---|---|
| Plain or shiny surfaces vanish (e.g. a laptop screen) | Matching and depth need texture; reflections break the assumption that a point looks the same from every view | Capture protocol (matte cover, diffuse light); learned depth priors (DA3 helps here); visibility-based meshing; NBV flags weakly supported areas |
| Thin structures dropped | TSDF voxel size and outlier filtering remove sparse thin sheets | Smaller voxels in the object's bounding box; per-view confidence weighting; Poisson with point normals |
| DA3 crashes or poor poses at scale | MPS bugs (fp16 crash already observed); memory grows with view count | fp32; cap views per pass and fuse in chunks; BA refinement |
| Texture seams / exposure jumps | Different photos have different exposure and white balance | Global colour adjustment + seam leveling (Waechter et al. 2014) |
| Scale/orientation arbitrary | SfM only recovers geometry up to a similarity | Normalise to unit box; "up" = mean camera up-vector; optional scale from a known object dimension |
| Mixed cameras within one capture (the toy set mixes the iPhone main and ultra-wide lenses) | Phones switch lenses automatically, which changes intrinsics and distortion per photo | Per-view intrinsics everywhere (DA3 predicts them per view); EXIF focal length as a prior; per-camera distortion in BA (v2) |
| Exposure varies between photos (the toy set's ISO ranges 320–1000) | Auto exposure | Per-image gain/white-balance compensation before texture blending |
| Fuzzy materials (fur) | Soft silhouettes, view-dependent sheen | Soft mask edges; weight texture samples by viewing angle; Gaussian splats as a fallback appearance model |
| Mac-only constraint | Much 3D research code is CUDA-only | Prefer native ComfyUI nodes and pure PyTorch / CPU libraries (Open3D, pycolmap, xatlas) |

## 6. How we work

1. Prototype each stage as **plain Python functions**, tested on saved intermediate data.
2. Wrap them as a ComfyUI **custom node pack** (`ysda3d`) that mirrors those functions one-to-one.
3. Claude writes and validates the workflows, runs them via comfy-mcp, and inspects outputs. You open the same workflows in Comfy Desktop.
4. Every iteration changes **one** stage and is judged on the evaluation harness (M4).
5. Every iteration or experiment gets a **journal entry** (`journal/NNNN_*.md`): hypothesis before running, results and conclusion after, failures included.
