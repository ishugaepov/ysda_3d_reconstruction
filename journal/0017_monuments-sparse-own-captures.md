# 0017 — Own sparse captures: three monuments with 4–9 photos

- **Date:** 2026-11-25
- **Status:** done (report built); the main limitation found is the camera stage
- **Stage(s):** evaluation on new data; camera fallback; input selection
- **Builds on:** #0016 (benchmark), plan [05_benchmark_plan.md](../05_benchmark_plan.md)

## Question
How do the approaches built so far do on our own small captures of outdoor monuments (`~/Downloads/monument_*`)?
- `monument_589`: standing statue on a tall pedestal; 9 photos, including 2 close-ups.
- `monument_616`: seated statue; 4 photos about 90° apart.
- `monument_625`: thin metal arch; 6 photos, including 4 close-ups.

## Setup
- **Benchmark set `monuments`** (`bench/examples.json`, `--set monuments`): the photos are copied into `~/datasets/ysda_bench/examples/`; nothing goes into the repo.
- **Held-out photos:** 589: 004, 008 (chosen by hand, see below); 616: 004; 625: 004 (a close-up).
- **Generator inputs:** full views only. 589: 5 (anchor 007); 616: 3; 625: 2.
- Methods and metrics as in #0016. No ground truth here, so no oracle and no geometry scores.
- **Time:** 10:28 → 11:15, about 50 minutes of wall-clock, against an estimate of 1.5–2 h.
  - UniForge3D took only 10–12 min per object with 2–5 input views (21–47 min with 8).
  - Masks + DA3: 1–1.5 min. Scoring and the report: a few minutes.

## What had to change in the pipeline
1. **COLMAP fails on all three:** 3 of 9, 0 of 4 and 0 of 6 photos registered ("no good initial image pair"). Photos 60–90° apart share too few matching features.
   - New fallback: when COLMAP registers < 60% of the photos, **every method uses DA3's cameras** (`cameras.json` records the source).
2. **Close-ups:**
   - A *full view* (`bench.full_views`) is one where the object covers at most 20% of any image edge. Only full views become generator inputs, so a face close-up never becomes the anchor.
   - For held-out photos, `bench.auto_heldout` holds out the most redundant view. It holds out a close-up instead if a full view would leave fewer than 2 full views for the generator: close-ups are still fair tests inside their frame.
   - On 589, mixing close-ups with distant shots made DA3's poses inconsistent. Estimating the cameras from the 7 full views only (`camera_views: "full"`) raised the carve's held-out IoU from 0.12 to 0.37, and its fit on the training photos from 0.46 to 0.57.
3. **The masks disagree about the pedestal (589).** In the two wide shots, background removal includes the pedestal; in the others only the figure. The automatic rule would have held out exactly the two pedestal views, so the held-out photos were chosen by hand (004, 008).
4. **Camera check** (in every example's diagnostics and in the report): the IoU of the reconstruction-only carve against its *own* training masks.
   - It's about 0.9 with consistent cameras.
   - Here it's **0.57 / 0.69 / 0.51**. Held-out scores therefore partly measure camera error.

## Results (held-out photos; DA3 cameras)
| method | 589 PSNR / IoU | 616 PSNR / IoU | 625 PSNR / IoU | mean PSNR full | mean IoU |
|---|---|---|---|---|---|
| v0 | 9.32 / 0.404 | **13.48** / 0.798 | 7.77 / 0.150 | 10.19 | 0.451 |
| v1.2 (reconstruction only) | 8.17 / 0.369 | 13.44 / 0.759 | 7.94 / 0.155 | 9.85 | 0.427 |
| generation only | 11.80 / 0.280 | 8.80 / 0.727 | 10.89 / 0.603 | 10.50 | 0.537 |
| generation + colour map | 12.62 / 0.280 | 9.60 / 0.727 | **12.29** / 0.603 | 11.50 | 0.537 |
| v2.3 | 12.25 / 0.284 | 10.62 / **0.809** | 12.24 / **0.626** | **11.70** | **0.573** |
| v2.3.1 (current) | **12.71** / 0.280 | 9.56 / 0.727 | 12.21 / 0.603 | 11.49 | 0.537 |

The report has the photos, the held-out comparisons, four views of every model and the GLBs: `~/datasets/ysda_bench/report_monuments/index.html`. The models also appear in ComfyUI's Load 3D as `3d/ysda_bench/monument_*/`.

## Findings
1. **Visually, generation wins on all three, and by a lot.**
   - Reconstruction from 2–5 photos gives blobs (616), fragments (589) or shards (625).
   - UniForge3D gives complete, recognizable models:
     - 616: a seated man with hat, cane, bench and pedestal;
     - 589: the coated figure with crossed arms;
     - 625: a clean horseshoe arch, whose photo-coloured versions carry readable lettering.
2. **The scores only partly agree.**
   - Arch: they agree (IoU 0.60–0.63 vs 0.15; PSNR 12.2 vs 7.9).
   - 589: mixed. PSNR favours generation (12.7 vs 8.2); IoU favours reconstruction (0.37 vs 0.28), with all IoUs low.
   - 616: reconstruction wins on PSNR (13.4 vs 9.6), despite looking far worse.
   - With inconsistent cameras, a sharp, detailed model that's slightly misaligned gets more error than a blurry one. **Held-out metrics need reliable cameras to mean anything.**
3. **Photo colours need good cameras.** With DA3's poses, the projected photo colours come out muddy: grey instead of bronze on 616, streaky on 589.
   - Current pipeline vs generation + colour map: median −0.04 dB, 1–2.
   - On captures this sparse the photo-colour step earns nothing. The generator's colours after the global colour map are as good.
4. **Thin parts leak colour.** The arch's unseen back shows faint lettering: the ring is thin, and visibility lets photo colours through.
5. **Generation cost scales with the number of views:** 10–12 min with 2–5 views.

## Conclusion
- On sparse own captures, the generation-based approaches are clearly the useful ones.
- The **camera stage** is what fails. COLMAP can't register the photos, and DA3's cameras are too inconsistent for the photo-colour and evaluation steps that worked well in #0016.
- So the scores here are indicative only, and the model images are the more reliable comparison.

## Next
1. **Per-photo camera refinement against the generated model.** After the global registration, fit each photo's camera (6-DoF, perhaps focal) so the mesh's silhouette matches that photo's mask.
   - This fixes the colour projection.
   - Applied to the held-out cameras per method (test-time pose alignment), it makes the held-out scores mean something on sparse captures.
2. **Better feed-forward cameras for few views:** VGGT is already inside UniForge3D; MASt3R-style matching is another option.
3. **Trust photo colours less when cameras are uncertain** (camera check), falling back to generator colours + the colour map.
4. **Consistent masks across views** (the pedestal question), e.g. decided once on the generated model.
5. Prevent colour leaking through thin structures (tighter visibility for thin parts).
