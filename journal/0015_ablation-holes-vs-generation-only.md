# 0015 — Ablation: do the holes explain the low "generation only" score?

- **Date:** 2026-11-10
- **Status:** done
- **Stage(s):** evaluation
- **Builds on:** #0012 (held-out protocol), #0013 (mesh repair), #0014 (summary table)

## Question
The raw UniForge3D output is full of tiny holes (#0013). Is the generation-only baseline in #0014 (11.0 dB) low because of these holes, rather than because of its colours?

## Protocol
- **Mesh:** UniForge3D, K = 8 (`outputs/uf3d/toy_k8.glb`, seed 42).
- **Test and fit:** the same held-out photos as before (IMG_4506, 4510, 4516) and the same registration to the COLMAP cameras (7-DoF silhouette fit on the 14 training photos).
- **2 meshes × 3 colourings**:
  - meshes: **raw** (with holes, no culling) vs **repaired** (YSDA Repair Generated Mesh, 512 grid);
  - colours: **generated** (UniForge3D's own); **generated + global colour correction**; **photo colours** (v2.3).
  - The global colour correction is one affine RGB map (12 numbers), fitted on vertices well observed in the training photos. It's the harmonisation step of v2.3 applied on its own, and it is the fairest generation-only baseline.
- Commands:
  - `CULL=0 ONLY=uf3d_k8 scripts/eval_v2.py outputs/toy_sfm.npz outputs/ablation_holes/raw`
  - `REPAIR=1 ONLY=uf3d_k8 scripts/eval_v2.py outputs/toy_sfm.npz outputs/ablation_holes/repaired`

## Results
| Mesh | Colours | PSNR full | PSNR covered | IoU |
|---|---|---|---|---|
| raw (holes) | generated | 11.08 | 11.40 | 0.962 |
| repaired | generated | 11.00 | 11.36 | 0.962 |
| raw (holes) | generated + global colour correction | 15.96 | 17.23 | 0.962 |
| repaired | generated + global colour correction | 15.96 | 17.44 | 0.962 |
| raw (holes) | photo colours | 16.41 | 17.86 | 0.962 |
| repaired | photo colours (= current pipeline) | **17.88** | **20.57** | 0.961 |
| *v1.2, reconstruction only* | *photo colours* | *17.60* | *20.00* | *0.925* |

Run-to-run spread: repeating a configuration moves PSNR by about 0.06 dB (17.82 vs 17.88 for the current pipeline; 17.61 vs 17.60 for v1.2).

## Conclusion
1. **The holes don't explain the generation-only score.** With generated colours, the raw and repaired meshes score the same (11.08 vs 11.00), and IoU is identical. The holes are too small to change silhouettes or the colour error.
2. **A global colour cast explains most of the deficit,** about 5 dB of the 6.9 dB. The generated base colour has no lighting and a different brightness and white balance. A 12-number colour map removes it: 15.96 dB.
3. **The rest is detail only the photos have:** about 1.9 dB full and 3.1 dB covered (fur, shading, local colour variation). That's the genuine gain of photo colours. Both the colour map and the photo colours need B's cameras, to know which pixel each vertex corresponds to.
4. **The holes and the messy topology do hurt photo colouring:** raw 16.41 → repaired 17.88 (covered 17.86 → 20.57). That is why the repair (#0013) improved the combined pipeline.
5. **Against the fairest generation-only baseline (15.96 / 17.44), the combined pipeline is still clearly ahead (17.88 / 20.57).** So is reconstruction only (17.60 / 20.00). Caveat as before: one object, three test photos.

## Artifacts
- `scripts/eval_v2.py` also reports **generated colours, harmonised** (`*_genharm`).
- `generated.recolour_from_photos(..., return_harmonised=True)`.
- Logs and renders are in `outputs/ablation_holes/` (not committed).

## Next
- Move the repair into G, so the generated model shown in ComfyUI is already hole-free. The repair uses no cameras or photos, only the generated mesh.
  - Done (2026-11-10): G1 in the workflow's G group. G's preview shows the repaired model, with smooth normals.
