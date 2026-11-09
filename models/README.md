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
| [`toy/v2_uniforge3d_k4.glb`](toy/v2_uniforge3d_k4.glb) | toy (4 views: 4505, 4509, 4513, 4518) | v2.1, [journal 0010](../journal/0010_v2-1-uniforge3d-toy.md) | UniForge3D (Mac port, journal 0009), RGBA from BiRefNet masks at photo resolution | `v2/uniforge_mac/run.py` | Complete, upright, crisp shape and face; albedo-flat colours; small gaps in feet. 2.2M faces (39 MB). 2026-11-09 |
| [`toy/v2_uniforge3d_k8.glb`](toy/v2_uniforge3d_k8.glb) | toy (8 views: 4505, 4507, 4509, 4511, 4513, 4515, 4518, 4520) | v2.1, [journal 0010](../journal/0010_v2-1-uniforge3d-toy.md) | as K=4 | `v2/uniforge_mac/run.py` | More photo-faithful (grey face plate, fur texture, fuller feet), slightly softer face. 3.1M faces. 2026-11-09 |
| [`toy/v2_3_fused_k8.glb`](toy/v2_3_fused_k8.glb) | toy (UniForge3D K=8 + 17 COLMAP photos) | v2.3, [journal 0012](../journal/0012_v2-2-v2-3-register-recolour.md) | UniForge3D mesh, visibility-culled, registered to COLMAP (7-DoF silhouette fit), coloured from photos (generated colours harmonised where unseen), upright | `photos_to_3d_v2_3_fused` | Best so far visually: complete, clean face, real colours/fur. Held-out PSNR 17.2, IoU 0.962. Decimated to 0.93M faces. 2026-11-09 |
| [`toy/v2_3_1_repaired_k8.glb`](toy/v2_3_1_repaired_k8.glb) | toy (UniForge3D K=8 + 17 COLMAP photos) | v2.3.1, [journal 0013](../journal/0013_v2-3-1-repair-generated-mesh.md) | As v2.3, but the generated mesh is repaired first (narrow-band remesh: holes closed, open bottom capped, outward winding, smooth normals) | `photos_to_3d_v2_3_fused` | No holes (0 open edges). Held-out PSNR 17.8 (v1.2: 17.6), IoU 0.962. Decimated to 600k faces (workflow output: 1.85M). 2026-11-09 |

Reference results from commercial apps (same photos) are **not** stored here; they're in `~/Downloads/toy_kiri` and `~/Downloads/toy_poly.glb` on the author's machine.
