# YSDA 3D reconstruction — photos → textured 3D model in ComfyUI

## Goal

Build a **ComfyUI pipeline** that takes **a set of photos of one object, taken from arbitrary angles**, and produces a **textured 3D model of that object alone** (background ignored).

**"Textured" means the model looks like the real object in its colouring.** Viewed from any angle, the colours, markings and their placement match the photos: the toy's face, pink cheeks, red fins and white feet are in the right places with the right colours. The deliverable is a GLB mesh with a UV colour texture taken from the real photos. Physically based materials (separate roughness/metal maps) are not a goal.

**Reference input:** [`data/toy`](data/README.md), 21 handheld iPhone photos of a plush toy on a parquet floor ([journal 0002](journal/0002_toy-dataset.md)). The pipeline must handle captures like this.

Constraints:
- runs locally on a MacBook Pro (M3 Max, 64 GB, Apple GPU via MPS, no CUDA);
- uses **open models and open-source components**, reusing as much existing work as possible;
- ComfyUI orchestrates everything, and Claude drives it through comfy-mcp.

**Extension (not part of v1):** the pipeline suggests the camera angles to photograph next, to improve the model, and visualises them (next-best-view, NBV).

## Documents

| File | Contents |
|---|---|
| [01_roadmap.md](01_roadmap.md) | High-level path: architecture, design decisions, milestones, risks |
| [02_prototype_plan.md](02_prototype_plan.md) | Detailed execution plan for the first working prototype and how to iterate on it |
| [03_literature.md](03_literature.md) | Reading list: the math and machinery behind every stage, plus a suggested reading order |
| [journal/](journal/README.md) | **One entry per iteration/experiment**, including failures, so we never repeat what we already tried |
| [data/](data/README.md) | Capture datasets (input photos) |
| [models/](models/README.md) | Result GLB per iteration, for side-by-side comparison |

## Repo layout

```
README.md, 01_roadmap.md, 02_prototype_plan.md, 03_literature.md
journal/        experiment log (NNNN_slug.md, TEMPLATE.md, assets/)
data/<name>/    captures (input photos)
models/<name>/  result GLB per iteration
ysda3d/         pure-Python pipeline code (no ComfyUI imports)
comfy_nodes/    thin ComfyUI wrappers around ysda3d
workflows/      API-format and UI-format workflow JSON
scripts/        data sync, synthetic rendering, evaluation
tests/          unit tests
```

## Current state (2026-11-04)

- Comfy Desktop runs ComfyUI 0.39.2 at `127.0.0.1:8188`.
- comfy-cli and comfy-mcp work, so Claude can build, validate, run and inspect workflows.
- Models available: **Depth Anything 3 Base** (multi-view depth + camera poses; must load as **fp32** on MPS) and **BiRefNet** (background removal).
- A smoke test of DA3 multi-view runs in about 3 s for 2 views (`~/ComfyUI-Shared/claude-workflows/da3_multiview_smoke_fp32.api.json`).
