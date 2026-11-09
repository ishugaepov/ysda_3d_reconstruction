# Journal

Every iteration or experiment gets an entry, **including failures and dead ends**. The goal is to never re-try something blindly, and to be able to answer "why is it done this way?" months later.

## Rules

- One file per experiment: `NNNN_short-slug.md`. Numbers are sequential and never reused.
- Copy [`TEMPLATE.md`](TEMPLATE.md). Fill in the **hypothesis before running** and the **results/conclusion after**.
- Record exact inputs: dataset name, workflow file, git commit, model files, and any parameters that differ from defaults.
- Put small artifacts (contact sheets, renders, plots, metrics) in `assets/NNNN_*`. Big outputs (meshes, point clouds) stay in `~/ComfyUI-Shared/output/ysda/NNNN/` and are referenced by path.
- Final models of an iteration go into `models/<dataset>/<version>_<method>.glb`, with a row in `models/README.md`.
- Statuses: `planned` → `running` → `done` / `failed` / `abandoned`.
- Add a line to the index below for every entry.

## Index

| # | Date | Title | Status | One-line outcome |
|---|---|---|---|---|
| [0001](0001_environment-setup.md) | 2026-11-02 | Environment: ComfyUI + comfy-mcp + DA3 on Mac | done | Works; DA3 must run in fp32 on MPS (fp16 crashes the server) |
| [0002](0002_toy-dataset.md) | 2026-11-04 | Reference dataset: plush rocket toy (21 photos) | done | Good capture; mixed lenses/EXIF rotation and fuzzy fur are the main challenges |
| [0003](0003_pipeline-v0.md) | 2026-11-04 | Pipeline v0: masks + DA3 + voxel carving + vertex colours | done | Works on the toy: right shape and colour placement, blobby; fixed DA3 OOM (attention) and intrinsics scale |
| [0004](0004_diagnostics.md) | 2026-11-06 | Per-stage diagnostics vs KIRI/Polycam | done | Bottleneck = DA3 cross-view inconsistency (~1.8%, no DA3 setting fixes it); carving shaves, averaging blurs → next: bundle adjustment |
| [0005](0005_v1-colmap-cameras.md) | 2026-11-06 | v1: COLMAP SfM cameras (+ distortion, near-object depth scale, registration checks) | done | IoU 0.813 → 0.895 (strict hull 0.956), face much sharper; 17/21 photos kept; next: mesh-based visibility for colours |
| [0006](0006_study-uniforge3d.md) | 2026-11-07 | Study: UniForge3D + Mac feasibility; plan v2 | done | New main direction; port needed (CUDA deps), DINOv3 gated; Track B = native TRELLIS.2 + our structure |
| [0007](0007_v1-1-mesh-visibility.md) | 2026-11-07 | v1.1: mesh z-buffer visibility + looser carving; held-out evaluation | done | Held-out full-mask PSNR 13.7 → 17.4 dB, IoU 0.88 → 0.92; fuller shape, slightly softer face |
| [0008](0008_v1-2-tsdf-upright.md) | 2026-11-09 | v1.2: TSDF fusion in hull (rejected) + upright orientation (adopted) | done | TSDF: sharper face XOR full silhouette, never beats v1.1; found a ~4% DA3 depth bias. Upright from floor plane works |
| [0009](0009_v2-0-uniforge3d-on-mac.md) | 2026-11-09 | v2.0: UniForge3D running on Apple Silicon (pure-torch shims) | done | Works without source changes; case01 in 16 min on M3 Max; vertex colours instead of UV bake |
| [0010](0010_v2-1-uniforge3d-toy.md) | 2026-11-09 | v2.1: UniForge3D on real toy photos (K=4, K=8) | done | Best geometry so far, complete & upright; K=8 more photo-faithful (grey face plate, fur), K=4 crisper face; 28/33 min |
| [0011](0011_v2-comfyui-workflow.md) | 2026-11-09 | v2 in ComfyUI: UniForge3D workflow (subprocess node, preview/full) | done | Nodes tested offline (preview 283 s); workflow ysda_photos_to_3d_v2_uniforge3d |
| [0012](0012_v2-2-v2-3-register-recolour.md) | 2026-11-09 | v2.2 + v2.3: register UniForge3D to COLMAP, colour from photos | done | Held-out IoU 0.962 (v1.2: 0.925); PSNR 17.2 (v1.2: 17.6); clean face, complete; raw mesh needs culling + abs-cos |
