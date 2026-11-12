# 05 — Benchmark: comparing approaches on many examples, quickly

*Written 2026-11-11. Until now every idea was judged on one object (the toy). This benchmark scores all our
approaches on several real and synthetic objects and averages the metrics, fast enough to rerun after
every change. Results and findings go in the journal (0016 onwards).*

## 1. Examples (`bench/examples.json`)
| Kind | Source | Per example | Scored on |
|---|---|---|---|
| **Real** | our toy (21 photos); [CO3D-small](https://huggingface.co/datasets/XeemPad/CO3D-small) phone videos (cup and plant: 40 frames each, 3 held out; three more sequences fail at COLMAP and are disabled) | photos, nothing else | held-out photos |
| **Synthetic** | [Google Scanned Objects](https://huggingface.co/datasets/hugging-3d/google-scanned-objects) meshes (sofa, mug, shoe), rendered as handheld captures by `ysda3d/synth.py` | 25 renders (20 train, 3 held out between the rings, 2 low), true cameras, masks and mesh | held-out photos + low views + geometry |

- **Synthetic captures** exercise the whole pipeline, from masks through COLMAP.
  - The object stands on a speckled floor inside a backdrop with posters; both are realistic and feature-rich, so COLMAP registers the views as it would in a room.
  - Lighting: directional light with shadows, plus ambient. Renders get sensor noise and JPEG compression.
  - Cameras: rings at ~25° and ~50° elevation, plus 2 near-top views.
  - Held out: views between the rings, and 2 at ~10° elevation. The low views test parts the training views saw only obliquely.
- **Data** lives in `$YSDA_BENCH_ROOT` (default `~/datasets/ysda_bench`), never in the repo. The CO3D licence is non-commercial, and the photos are large. The repo holds the recipe: sequence/object ids, seeds, held-out rules.
- **Adding an example** is one line in `examples.json`:
  - a CO3D sequence id;
  - a GSO object name (each 0.5 GB shard has ~24 objects);
  - or your own photos: `{"name": ..., "kind": "real", "photos": "data/<name>/images", "heldout": [...]}`.

## 2. Methods (`ysda3d/bench_eval.py`, `METHODS`)
Every method sees only the training photos and returns a coloured mesh in the example's COLMAP frame.

| Method | What it is |
|---|---|
| `recon_v0` | v0: DA3's own cameras, visual-hull carving, DA3-visibility colours (aligned to COLMAP for scoring) |
| `recon_v1_2` | v1.2: COLMAP cameras, carving, mesh-visibility photo colours (**reconstruction only**) |
| `gen_only` | UniForge3D mesh (repaired), registered, its own colours (**generation only**) |
| `gen_harmonised` | as above + one global colour map fitted on the photos (#0015) |
| `fused_v2_3` | v2.3: raw UniForge3D mesh + hidden-face cull, registered, photo colours |
| `fused_v2_3_1` | v2.3.1, the current pipeline: repaired mesh, registered, photo colours |
| `oracle_gt_mesh` | reference, synthetic only: the true mesh, coloured from the photos the same way. Its gap to the fused methods is what geometry still costs |

## 3. Metrics
- All scores come from **exact renders**: a z-buffer triangle rasteriser with perspective-correct vertex colours and COLMAP's lens distortion. The older `diagnostics.*` checks splat points 2 px wide, which caps an exact shape near IoU 0.96 and favours shrunken meshes. Benchmark numbers therefore differ from journals ≤ 0015.
- **Held-out photos (all examples):**
  - silhouette IoU against the object mask;
  - full-mask PSNR (mask pixels the model doesn't cover count as errors);
  - covered PSNR (only pixels the model covers);
  - coverage (fraction of object pixels the model covers), which explains differences between the two PSNRs.
- **Synthetic only:**
  - the same on the low views;
  - geometry against the true mesh, after aligning COLMAP to the true frame with a similarity fitted on the camera centres:
    - Chamfer distance, as % of the object's bounding-box diagonal;
    - F-score at 1% and 2%;
    - precision (accuracy) and recall (completeness) at 2%.
- **Averages:** mean ± std over examples of per-example means. Only examples where every compared method ran count, so rows stay comparable; a missing generation, for example, drops that example.
- **Diagnostics:** views per split, stage timings, and for synthetic examples the IoU of our BiRefNet masks against the true masks.

## 4. Time budget, and why iterations stay fast
- **Once per example:**
  - capture: CO3D download ~30 s, or GSO render ~1 min;
  - masks + DA3: 4–7 min;
  - COLMAP: 1–2 min;
  - UniForge3D: 21–47 min, queued after all the reconstruction stages.
- **First evaluation run:** 17.5 min for 6 examples × 7 methods, with 3 workers. Most of it is the two registrations per example.
- **Re-runs** recompute only what changed:
  - Pipeline stages are files in `work/<name>/`.
  - Generations are found by `ysda3d/runcache.py`, by input fingerprint, seed and port code.
  - Method stages are cached as `cache/<name>/<stage>_<hash>.pt`, keyed by input-file hashes, parameters and the source of the `ysda3d` modules they use. Editing `generated.py` re-runs registration and colouring, but not the reconstructions.
- **Cached rerun:** ~5 s.
- `--quick` runs one example per kind (toy, co3d_cup, gso_sofa).
- **Paired table:** every run also reports the current pipeline minus each method per example, as median Δ and wins–losses. One example can carry an average; it can't carry a median or a win count.

## 5. Usage
```
PY=~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python
$PY -I scripts/bench_prepare.py --no-generate     # captures, masks, DA3, COLMAP for all examples (~1 h for 7)
$PY -I scripts/bench_prepare.py                   # + the UniForge3D queue (~30 min per example, resumable)
$PY -I scripts/bench_run.py                       # all methods x all prepared examples -> outputs/bench/latest.md
$PY -I scripts/bench_run.py --quick --methods recon_v1_2,fused_v2_3_1
$PY -I scripts/bench_report.py                   # pictures + GLBs of every method's model (~30 s)
```
**Where things are:**
- input photos: `$YSDA_BENCH_ROOT/examples/<name>/photos/` (the toy: `data/toy/images/`);
- for synthetic examples, true masks and mesh: `examples/<name>/gt/`;
- the textured scans: `gso_shard0/extracted/<object>.glb`;
- **report:** `$YSDA_BENCH_ROOT/report/index.html`; the repo copy is [bench/report/](bench/report/README.md) (`--out bench/report --no-glb --no-html`). Per example: input photos (held-out and generator views marked), each held-out photo next to every method's render, four views of every model, metrics;
- **models:** `~/ComfyUI-Shared/input/3d/ysda_bench/<name>/<method>.glb`. Open them in ComfyUI with **Load 3D** (`3d/ysda_bench/...`, press R to refresh) or any GLB viewer. Real examples are in the upright pipeline frame, synthetic ones in the true frame, next to `gt_mesh.glb`.
To test a new idea, add it as a method in `METHODS` (or change a stage). The rerun reuses everything it doesn't touch.

## 6. Known limitations
- **Few examples so far (6).** Adding more costs mostly generation time. One GSO shard has ~24 objects, CO3D-small has ~20 sequences per category, and own captures are best of all.
- **Synthetic renders are cleaner than real photos** (no blur, simple lighting). Look at the real and synthetic groups separately, not only at the overall mean.
- **Held-out photos mostly see what the training photos saw.** Completeness and hallucinated parts are tested only by the low views and the true geometry, on synthetic examples.
- **Evaluation uses the pipeline's own masks.** For synthetic examples their quality is reported.
- **Noise floor.** Re-running DA3 + COLMAP gives near-identical cameras, but registration then settles ~2° differently, which moves the toy's PSNR by up to 0.5 dB. Single-example differences below that aren't meaningful; averages over examples are the point. Repeated runs with different seeds would quantify it.
- **Some captures fail in the pipeline itself.** That is reported, and is itself a robustness result.
  - The CO3D cup at 20 frames: COLMAP kept no photos; it works at 40.
  - Two other sequences (`plant/375_42504_85162`, `apple/444_63731_125794`): COLMAP split the orbit into pieces and only 9 of 40 frames survived.
  - The plant was replaced by a sequence with a more textured floor. The apple stays in `examples.json` as `disabled`, a test case for a more robust camera stage.
