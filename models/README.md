# Result models

One GLB per pipeline iteration and dataset, so iterations can be compared side by side. View them in ComfyUI (Load 3D), Blender, or any glTF viewer, or render them together with:

```bash
~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python scripts/render_compare_models.py out.jpg \
    v0=models/toy/v0_da3_carving.glb v1=models/toy/v1_colmap.glb
```

Naming: `models/<dataset>/<version>_<short-method>.glb`. Add a row below for every model, and reference it from the iteration's journal entry.

| Model | Dataset | Iteration | Method | Workflow | Notes |
|---|---|---|---|---|---|
| [`toy/v0_da3_carving.glb`](toy/v0_da3_carving.glb) | toy (21 photos) | v0, [journal 0003](../journal/0003_pipeline-v0.md) / [0004](../journal/0004_diagnostics.md) | DA3 cameras + depth, voxel carving, vertex colours | `photos_to_mesh_v0_diagnostics` | Blobby, soft colours; silhouette IoU 0.81. Produced in Comfy Desktop 2026-11-06 |
| [`toy/v1_colmap.glb`](toy/v1_colmap.glb) | toy (17 of 21 photos kept) | v1, [journal 0005](../journal/0005_v1-colmap-cameras.md) | COLMAP SfM cameras (bundle adjustment, k1), DA3 depth rescaled per view, carving, vertex colours | `photos_to_mesh_v1_colmap` | Sharp face, distinct feet; IoU 0.895; arbitrary orientation, dark underside. Produced in Comfy Desktop 2026-11-06 |
| [`toy/v1_1_meshvis.glb`](toy/v1_1_meshvis.glb) | toy (17 of 21) | v1.1, [journal 0007](../journal/0007_v1-1-mesh-visibility.md) | v1 cameras; carving depth_tol 0.05; colour visibility from the mesh z-buffer | `photos_to_mesh_v1_1` | Fuller shape, fewer artifacts, slightly softer face; held-out full-mask PSNR 17.4 dB (v1: 13.7). Produced offline with the node code 2026-11-07 |
| [`toy/v1_2_upright.glb`](toy/v1_2_upright.glb) | toy (17 of 21) | v1.2, [journal 0008](../journal/0008_v1-2-tsdf-upright.md) | v1.1 surface + colours; upright output (floor plane → y=0, +Y up; first photo = front) | `photos_to_mesh_v1_2` | Same geometry/colours as v1.1, now standing upright like the Polycam result. Produced offline with the node code 2026-11-09 |

Reference results from commercial apps (same photos) are **not** stored here; they're in `~/Downloads/toy_kiri` and `~/Downloads/toy_poly.glb` on the author's machine.
