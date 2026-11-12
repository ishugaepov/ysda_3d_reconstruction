# 0016 — Benchmark v1: every approach on 6 objects (3 real, 3 synthetic)

- **Date:** 2026-11-11
- **Status:** done
- **Stage(s):** evaluation, infrastructure
- **Builds on:** #0012–#0015; plan [05_benchmark_plan.md](../05_benchmark_plan.md)

## Question
So far every conclusion came from one object (the toy). Do they hold across objects? In particular, does the combined approach beat reconstruction alone and generation alone, and where?

## Setup
- **Examples** (`bench/examples.json`; data in `~/datasets/ysda_bench`):
  - the toy (21 photos);
  - two CO3D-small phone videos, cup and plant (40 frames each, 3 held out);
  - three GSO objects (sofa, mug, shoe), rendered by `ysda3d/synth.py` as handheld captures: 25 views, of which 20 train, 3 held out between the rings, and 2 low (~10°).
- **Methods** (7): v0, v1.2, generation only (own colours), generation + global colour map, v2.3 (raw mesh + cull), v2.3.1 (repaired, current), and the oracle (true mesh, photo colours).
- **Metrics:** exact renders on held-out photos (IoU, full-mask/covered PSNR, coverage). Synthetic examples add the low views and geometry against the true mesh (Chamfer % of the bounding-box diagonal, F-score, precision/recall).
- **Commands:** `scripts/bench_prepare.py`, `scripts/bench_run.py`. Full output: [assets/0016_benchmark_summary.md](assets/0016_benchmark_summary.md).
- **Time:**
  - preparing an example:
    - masks + DA3: 4–8 min (40 views ~7 min);
    - COLMAP: 1–1.5 min;
    - render: ~40 s;
    - UniForge3D: 21–47 min;
  - first evaluation of 6 examples × 7 methods: 17.5 min with 3 workers;
  - cached rerun: **~5 s**.

## Results
Mean ± std over examples (held-out views):

| method | IoU | PSNR full | PSNR covered |
|---|---|---|---|
| v0 (DA3 cameras) | 0.657 ± 0.192 | 9.17 ± 2.38 | 15.08 ± 2.73 |
| v1.2 (reconstruction only) | 0.939 ± 0.045 | 17.12 ± 3.41 | 21.82 ± 3.34 |
| generation only (own colours) | 0.961 ± 0.007 | 11.30 ± 2.16 | 11.89 ± 2.42 |
| generation + global colour map | 0.961 ± 0.007 | 15.14 ± 1.58 | 16.91 ± 1.79 |
| v2.3 (raw mesh + cull) | 0.959 ± 0.017 | 17.79 ± 2.76 | 22.46 ± 3.88 |
| **v2.3.1 (current)** | **0.961 ± 0.007** | **18.04 ± 2.08** | **22.81 ± 3.24** |

Synthetic examples only (true geometry known):

| method | IoU low | PSNR low | Chamfer % | F@2% | precision@2% | recall@2% |
|---|---|---|---|---|---|---|
| v1.2 | 0.922 | 18.80 | 1.57 | 0.755 | 0.822 | 0.699 |
| generation only | 0.960 | 10.85 | 0.61 | 0.950 | 0.951 | 0.949 |
| v2.3 | 0.965 | 18.39 | **0.56** | **0.954** | 0.961 | 0.947 |
| v2.3.1 | 0.960 | 18.18 | 0.61 | 0.950 | 0.951 | 0.949 |
| *oracle (true mesh)* | *0.995* | *25.93* | *0.15* | *1.000* | | |

Paired per example: current pipeline minus each method (median Δ, wins–losses):

| vs | IoU | PSNR full | PSNR covered | coverage | Chamfer % | F@2% |
|---|---|---|---|---|---|---|
| v1.2 | +0.007 (5–1) | −0.215 (3–3) | +0.513 (5–1) | +0.004 (4–2) | −0.906 (3–0) | +0.186 (3–0) |
| generation + colour map | 0 (same mesh) | +3.094 (6–0) | +5.553 (6–0) | 0 | 0 | 0 |
| v2.3 (raw + cull) | −0.003 (2–4) | +0.015 (3–3) | +0.022 (3–3) | +0.001 (3–3) | +0.051 (0–3) | −0.006 (1–2) |

Per example, full-mask PSNR:

| | toy | cup | plant | sofa | mug | shoe |
|---|---|---|---|---|---|---|
| v1.2 | 16.84 | 15.57 | 10.94 | 20.94 | 20.89 | 17.56 |
| v2.3.1 | 15.80 | 15.95 | 18.88 | 21.61 | 19.25 | 16.75 |

## Findings
1. **Combined vs reconstruction only:**
   - **Geometry and completeness** are robustly better: 3/3 synthetic objects, Chamfer 0.61 vs 1.57%, recall 0.95 vs 0.70.
   - **Outline** is better on 5/6 (IoU), and **colour where the model covers the object** on 5/6 (covered PSNR, median +0.5 dB).
   - **Full-mask PSNR is mixed:** 3–3, median −0.2 dB. The +0.9 dB average is carried by the plant, where carving fails on thin leaves (+7.9 dB).
   - Where combined loses (toy, mug, shoe), it covers fewer object pixels at the edges (toy 0.956 vs 0.970, shoe 0.961 vs 0.978). **Edge coverage and alignment, not colour, is the remaining weakness.**
2. **Photo colours are essential.** Against the generator's own colours: +7.0 dB full (6–0). Against those colours after the best global colour map: +3.1 dB (6–0). This confirms #0015 on all objects.
3. **Repair vs raw + cull: no consistent difference** (3–3 on PSNR). On the clean synthetic generations, the repaired mesh is slightly less accurate (Chamfer +0.05%, 0–3): its surface is offset by one cell and smoothed. Its large win was on the toy, whose generation had by far the most holes. The repair still matters for how the model looks (no see-through holes). Recovering the lost precision is a cheap next step.
4. **Large headroom in geometry.** With the same colouring, the true mesh reaches 27.0 dB on the synthetic objects, against 19.2–19.8 dB for our methods (v1.2 is highest there). Even at 0.6% Chamfer, small surface differences still cost ~7–8 dB of photo fidelity.
5. **The camera stage is the weak point on real captures.** 3 of the 5 CO3D sequences tried failed at COLMAP:
   - `plant/375_42504_85162` and `apple/444_63731_125794`: the orbit split into pieces, 9 of 40 frames kept;
   - `cup/622_102276_205423`: a 26-image model with only 498 points, every view under the 200-point check.
   - They stay in `examples.json` as `disabled` test cases.
6. **The measurement had to change.** The old point-splat checks capped an exact shape at IoU ~0.96, favoured shrunken meshes and hid holes; the benchmark renders meshes exactly. Under the old checks the toy's v2.3 scored 17.2 dB; exactly rendered, its holes cost it 3.6 dB (13.6). Numbers here aren't comparable with journals ≤ 0015.

## What went wrong on the way
1. 20 frames per CO3D video were too sparse for COLMAP: the cup kept 0 views. With 40 it works.
2. Three more CO3D sequences failed at COLMAP (finding 5). One was replaced by a sequence with a textured floor; the other two are disabled.
3. The first synthetic textures were blurry pastel noise, unrealistic and weak for SfM. Terrazzo-style speckles and posters fixed both.
4. The oracle's IoU of 0.96, despite masks agreeing with the truth at 0.996, exposed the point-splat bias (finding 6).
5. Infrastructure:
   - The COLMAP work folder was shared across examples; it's now one folder per example.
   - Tooling pitfalls: zsh's `$path`/glob behaviour, grep output buffering, and editing a running shell script.
6. Noise floor: re-running DA3 + COLMAP on the toy moved registration by ~2° and its PSNR by up to 0.5 dB.

## Conclusion
- The combined approach is the best overall, and clearly the best on geometry, completeness and outline.
- It is not uniformly better on full-mask photo fidelity: objects that reconstruction alone handles well (toy, mug, shoe) still score slightly higher with v1.2, because v1.2 covers their edges more completely.
- Where reconstruction alone fails (thin parts, unseen sides), combined wins by a lot.
- The benchmark is cheap enough to run after every change: ~5 s cached, minutes for a changed stage.

## Artifacts
- **Code:**
  - `ysda3d/synth.py`: synthetic captures;
  - `ysda3d/bench.py`: registry, fetching, caching helpers;
  - `ysda3d/bench_eval.py`: methods, exact renders, metrics, stage and score caches;
  - `scripts/bench_prepare.py`, `scripts/bench_run.py`: the runner reports averages over common examples and a paired table against the current pipeline.
- **Config:** `bench/examples.json`.
- **Output:** `outputs/bench/latest.md`; a snapshot is in [assets/0016_benchmark_summary.md](assets/0016_benchmark_summary.md).
- **Visual report (added 2026-11-12):** `scripts/bench_report.py` writes `~/datasets/ysda_bench/report/index.html` (photos, held-out renders, model views) and every model as a GLB in `~/ComfyUI-Shared/input/3d/ysda_bench/` (ComfyUI Load 3D).

## Next
1. **Edge coverage:** after registration, refine the silhouette fit or apply a small dilation, or move vertices slightly toward photo consistency. Track coverage and full-mask PSNR on held-out views.
2. **Camera robustness:** make the 200-point check relative for low-resolution video frames (keeping the DA3 rotation check), and/or use sequential matching for videos. Then re-enable the 3 disabled CO3D examples.
3. **Repair precision:** project repaired vertices back onto the raw surface (removing the one-cell offset), or repair at 768; re-test against raw + cull.
4. **More examples and noise estimates:** ~20 more GSO objects in shard 0, own captures, and repeated runs or seeds.
