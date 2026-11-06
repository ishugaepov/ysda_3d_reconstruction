# Journal

Every iteration or experiment gets an entry, **including failures and dead ends**. The goal is to never re-try something blindly, and to be able to answer "why is it done this way?" months later.

## Rules

- One file per experiment: `NNNN_short-slug.md`. Numbers are sequential and never reused.
- Copy [`TEMPLATE.md`](TEMPLATE.md). Fill in the **hypothesis before running** and the **results/conclusion after**.
- Record exact inputs: dataset name, workflow file, git commit, model files, and any parameters that differ from defaults.
- Put small artifacts (contact sheets, renders, plots, metrics) in `assets/NNNN_*`. Big outputs (meshes, point clouds) stay in `~/ComfyUI-Shared/output/ysda/NNNN/` and are referenced by path.
- Statuses: `planned` → `running` → `done` / `failed` / `abandoned`.
- Add a line to the index below for every entry.

## Index

| # | Date | Title | Status | One-line outcome |
|---|---|---|---|---|
| [0001](0001_environment-setup.md) | 2026-11-02 | Environment: ComfyUI + comfy-mcp + DA3 on Mac | done | Works; DA3 must run in fp32 on MPS (fp16 crashes the server) |
| [0002](0002_toy-dataset.md) | 2026-11-04 | Reference dataset: plush rocket toy (21 photos) | done | Good capture; mixed lenses/EXIF rotation and fuzzy fur are the main challenges |
| [0003](0003_pipeline-v0.md) | 2026-11-04 | Pipeline v0: masks + DA3 + voxel carving + vertex colours | done | Works on the toy: right shape and colour placement, blobby; fixed DA3 OOM (attention) and intrinsics scale |
| [0004](0004_diagnostics.md) | 2026-11-06 | Per-stage diagnostics vs KIRI/Polycam | done | Bottleneck = DA3 cross-view inconsistency (~1.8%, no DA3 setting fixes it); carving shaves, averaging blurs → next: bundle adjustment |
