# 02 — Prototype plan: a simple working version, then iterate

The aim is the **shortest path to a textured GLB of a real object**: correctness first, quality later. Every step produces an artifact you can look at (an image, a point cloud, or a mesh) and a pass/fail criterion.

## 0. Project layout

```
~/ysda_3d_reconstruction/            (git repo)
  README.md, 01_roadmap.md, 02_prototype_plan.md, 03_literature.md
  journal/                 # one entry per experiment, see journal/README.md
  ysda3d/                  # Python package: pure functions, no ComfyUI imports
    geometry.py            # unprojection, transforms, camera conventions
    fusion.py              # point cloud fusion, TSDF, meshing
    texture.py             # UV unwrap, view selection, baking
    coverage.py            # (extension) observation statistics, NBV
  comfy_nodes/ysda3d_nodes/  # thin ComfyUI wrappers; symlinked into ComfyUI custom_nodes/
  workflows/               # API-format JSON (Claude/comfy-mcp) and UI-format JSON (Desktop)
  scripts/                 # data prep, synthetic rendering, evaluation
  data/<capture_name>/images/  # input photo sets, gitignored; copied to ComfyUI input/ by scripts/sync_to_comfy.sh
  tests/                   # unit tests on synthetic data
```

**Dependencies** go into ComfyUI's venv with uv (internal mirror). Python 3.13 / Apple Silicon wheels were verified for:
- `open3d` 0.20
- `trimesh`
- `xatlas`
- `pycolmap`

**Custom nodes load at ComfyUI startup.** After changing node code, restart Comfy Desktop, then Claude reruns the workflow.

## 1. Inputs

**Primary real dataset: `data/toy`** ([journal 0002](journal/0002_toy-dataset.md)). It's handheld, mixes lenses, and is stored with EXIF rotation. The prototype must work on it *as is*. Every stage treats these as hard requirements:
- apply EXIF orientation;
- use per-view intrinsics;
- don't assume constant exposure.

**Capture protocol for new captures** (for clean comparison datasets):
- **Object:** matte, richly textured, rigid, about 15–40 cm (a printed box, a sneaker, a plush toy). Keep the laptop as a later "hard case".
- **Photos:** 20–40, in 2–3 rings at different heights, every ~15–20° around the object, plus a few top-down shots. Overlap neighbouring photos by 60–80%.
- **Lighting and camera:** soft and even light; fixed exposure if the phone allows; no zoom changes.
- **Background:** a textured surface under the object helps camera estimation, and masks remove it afterwards.
- **Lens and exposure:** stay on the main (1×) lens and lock exposure (long-press AE/AF lock), so iPhone doesn't switch to the ultra-wide sensor.
- **Storage:** `data/<capture_name>/images/`, then `scripts/sync_to_comfy.sh <capture_name>`. Add a journal entry describing the capture.

## 2. Step-by-step build

### Step 1 — Masks (native nodes only)
Graph: `LoadImageDataSetFromFolder` → resize so the longest side is 1024 → BiRefNet background removal (the model is already in `models/background_removal`; confirm the exact native node name) → masks.

**Done when:** a contact sheet of masked images looks clean.

### Step 2 — Cameras + depth (native)
Graph: `RebatchImages` → `LoadDA3Model` (**fp32**) → `DA3Inference` (`multiview`, `cam_dec`), producing geometry with `depth`, `confidence`, `intrinsics`, `extrinsics`.

**Done when:** `DA3Render` depth images look plausible and the poses look plausible. The poses are checked in Step 3.

**First thing to verify:** DA3's camera convention. Is the extrinsic world→camera or camera→world? Is it OpenCV-style (x right, y down, z forward)? And how do the intrinsics scale with resizing? Test this with a synthetic scene where the answer is known (Step 7).

### Step 3 — Fused point cloud (custom node #1: `YSDA Fuse DA3 Point Cloud`)

For each view $i$ and pixel $p=(u,v)$ with depth $d$:

$$X_{world} = R_i^\top \left(d\,K_i^{-1}\tilde p - t_i\right) \quad \text{(for world→camera } [R_i|t_i])$$

**Inputs:** DA3 geometry, the images and the masks.

**Filtering:** keep pixels that are inside the mask, above a confidence threshold, and not on a depth discontinuity.

**Outputs:**
- a coloured point cloud (PLY, viewable with the native Preview Point Cloud / Save nodes);
- `cameras.json` with K, R, t per view, plus small camera frustums added as geometry for viewing.

**Done when:** all views overlap into one consistent object, without "ghost" duplicates, and the frustums sit around the object.

**This step is the first real test of the whole approach.** If views don't overlap well, either the poses are wrong or the convention is.

### Step 4 — Mesh (custom node #2: `YSDA TSDF Mesh`)

Option A (default): Open3D `ScalableTSDFVolume`.
- Integrate each view's masked depth with its pose.
- Set the voxel size from the object's bounding box (e.g. 1/256 of its diagonal), and the truncation to 4–5 voxels.
- Extract the mesh with marching cubes.

Option B: Screened Poisson on the fused cloud, with normals oriented towards the cameras.

Postprocess: keep the largest connected component, trim faces outside the visual hull of the masks, and decimate to about 100–200k faces.

**Done when:** a closed-looking mesh in Preview 3D, with no background slab.

### Step 5 — UV + texture (custom node #3: `YSDA Bake Texture`)

1. UV unwrap with `xatlas` (or the native UV-unwrap node).
2. **Visibility:** for each face and view, check whether the face is visible, using z-buffer rendering or raycasting.
3. **View selection, v0 = greedy per face:** score = $\cos\theta \times$ (projected area), and take the argmax.
4. **Bake:** for each texel, sample the chosen view's pixel. Fill gaps by dilating the atlas.
5. Attach the texture via native `ApplyTextureToMesh` → `MeshToFile3D`/`SaveGLB` (or write the GLB with trimesh).

**Done when:**
- the GLB opens in ComfyUI Preview 3D and in Blender;
- **on `data/toy`, rendered from the input cameras, the face, cheeks, red cone and fins, and white feet appear with the right colours in the right places**;
- seams are visible but tolerable.

### Step 6 — One workflow
Assemble Steps 1–5 into `workflows/photos_to_glb_v0` (UI and API formats). Run it from Desktop on a new capture with no manual steps.

**This is the v0 prototype.**

### Step 7 — Evaluation harness (in parallel with Steps 3–5)

- `scripts/render_synthetic.py`:
  - load a Google Scanned Objects mesh;
  - render 30 views with known K, R, t (pyrender/Open3D offscreen, or Blender);
  - add random backgrounds.
- `scripts/evaluate.py`:
  - pose error after Umeyama alignment;
  - Chamfer distance / F-score versus the ground-truth mesh;
  - held-out-view PSNR (render the textured mesh into views left out of reconstruction).
- On real data (`data/toy`): hold out 3 photos, then report masked PSNR/SSIM/LPIPS and ΔE between render and photo, plus a side-by-side sheet.
- Each run writes a small `metrics.json`, so iterations become comparable. The numbers and sheet go into that iteration's journal entry.

## 3. Iteration backlog (ordered by expected gain per effort)

| v | Change | Hypothesis / metric to watch |
|---|---|---|
| v1 | DA3 resolution 504 → 756/1008; chunked inference for >30 views | Finer depth → Chamfer ↓; watch memory |
| v2 | **Bundle-adjust DA3 poses with pycolmap** (features on full-resolution images, DA3 poses as initialisation) | Pose error ↓, sharper fused cloud, texture alignment ↑ |
| v3 | Confidence-weighted TSDF, adaptive voxel size; Poisson vs TSDF comparison | Thin parts kept, fewer holes |
| v3.5 | **Exposure / white-balance compensation** per image before baking (the toy set's ISO varies 320–1000) | ΔE ↓, seams ↓ |
| v4 | Texture: Markov-random-field view selection (data term + smoothness) and seam leveling; exposure compensation | Seams ↓, held-out PSNR ↑ |
| v5 | Full-resolution MVS (OpenMVS densify/refine) using the refined poses | Fine detail; compare against the DA3-only geometry |
| v6 | **NBV**: observation-quality map + greedy next-view suggestions + preview renders | Number of extra photos needed to reach a target coverage |
| v7 | Gaussian splatting (e.g. Brush or OpenSplat on Mac) as an alternative appearance model / splat-to-mesh | Quality on shiny or plain objects |
| v8 | Generative completion for unobserved regions | Completeness; mark generated regions explicitly |

## 4. NBV extension sketch (v6)

1. For each surface element $s$ and camera $c$, compute:
   - visibility $v_{sc}$;
   - incidence quality $q_{sc}=\cos\theta_{sc}$;
   - resolution $r_{sc}$ (pixels per unit area);
   - triangulation baseline: the maximum angle between viewing rays from the cameras that see $s$.
2. Combine these into an observation score $Q(s)$.
3. Generate candidate cameras on a sphere at the median capture distance, all looking at the centroid.
4. Greedily pick $k$ candidates by $\sum_s A_s\,[Q(s\mid C\cup\{c\})-Q(s\mid C)]$.
   - Coverage-type objectives are monotone submodular, which gives greedy selection a $(1-1/e)$ guarantee.
   - The baseline term breaks that guarantee, but greedy still works well in practice.
5. **Visualise:**
   - a heatmap of $Q$ on the mesh;
   - camera frustums for the suggestions;
   - for each suggestion, a render of the current model from that viewpoint, plus a relative instruction ("between photos 7 and 12, ~40° higher").

Alternative signals: DA3 confidence maps, or Fisher-information-based selection (FisherRF).

## 5. Division of labour with Claude

- **Claude:** writes the Python modules and nodes, the tests and the synthetic data; builds and validates workflows; runs them via comfy-mcp; reports metrics and renders.
- **You:**
  - restart Desktop after node changes;
  - take captures;
  - judge visual quality;
  - pick the next iteration from the backlog, using the theory in 03_literature.md.
