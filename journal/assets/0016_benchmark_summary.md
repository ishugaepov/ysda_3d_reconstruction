# Benchmark 2026-11-11 21:10 (4 s)

- `recon_v0`: v0: DA3 cameras, visual-hull carving, DA3-visibility colours
- `recon_v1_2`: v1.2: COLMAP cameras, carving, mesh-visibility photo colours (reconstruction only)
- `gen_only`: UniForge3D (repaired), registered, its own colours (generation only)
- `gen_harmonised`: UniForge3D (repaired), registered, own colours + global colour map
- `fused_v2_3`: v2.3: raw UniForge3D mesh + cull, registered, photo colours
- `fused_v2_3_1`: v2.3.1 (current): repaired mesh, registered, photo colours
- `oracle_gt_mesh`: reference, synthetic only: true mesh + photo colours

Mean ± std across examples of per-example means over held-out views. Bold = best (excluding the oracle).

### all examples: 6 examples (toy, co3d_cup, co3d_plant, gso_sofa, gso_mug, gso_shoe)

| method | IoU | PSNR full | PSNR cov |
|---|---|---|---|
| recon_v0 | 0.657 ± 0.192 | 9.17 ± 2.38 | 15.08 ± 2.73 |
| recon_v1_2 | 0.939 ± 0.045 | 17.12 ± 3.41 | 21.82 ± 3.34 |
| gen_only | **0.961 ± 0.007** | 11.30 ± 2.16 | 11.89 ± 2.42 |
| gen_harmonised | 0.961 ± 0.007 | 15.14 ± 1.58 | 16.91 ± 1.79 |
| fused_v2_3 | 0.959 ± 0.017 | 17.79 ± 2.76 | 22.46 ± 3.88 |
| fused_v2_3_1 | 0.961 ± 0.007 | **18.04 ± 2.08** | **22.81 ± 3.24** |

### real photos: 3 examples (toy, co3d_cup, co3d_plant)

| method | IoU | PSNR full | PSNR cov |
|---|---|---|---|
| recon_v0 | 0.547 ± 0.185 | 7.34 ± 1.08 | 13.91 ± 1.90 |
| recon_v1_2 | 0.910 ± 0.047 | 14.45 ± 2.53 | 19.32 ± 0.96 |
| gen_only | **0.957 ± 0.005** | 11.29 ± 2.68 | 12.07 ± 3.07 |
| gen_harmonised | 0.957 ± 0.005 | 14.89 ± 1.05 | 17.01 ± 1.32 |
| fused_v2_3 | 0.948 ± 0.016 | 16.05 ± 2.07 | 20.10 ± 2.21 |
| fused_v2_3_1 | 0.957 ± 0.005 | **16.88 ± 1.42** | **20.98 ± 1.79** |

### synthetic (with ground truth): 3 examples (gso_sofa, gso_mug, gso_shoe)

| method | IoU | PSNR full | PSNR cov | IoU low | PSNR low | Chamfer % | F@1% | F@2% | Prec@2% | Rec@2% |
|---|---|---|---|---|---|---|---|---|---|---|
| recon_v0 | 0.766 ± 0.125 | 11.00 ± 1.85 | 16.25 ± 2.93 | 0.756 ± 0.073 | 10.60 ± 2.30 | 2.46 ± 0.60 | 0.376 ± 0.099 | 0.634 ± 0.057 | 0.727 ± 0.077 | 0.564 ± 0.054 |
| recon_v1_2 | 0.969 ± 0.003 | 19.79 ± 1.58 | 24.31 ± 2.98 | 0.922 ± 0.018 | 18.80 ± 1.93 | 1.57 ± 0.31 | 0.624 ± 0.059 | 0.755 ± 0.030 | 0.822 ± 0.059 | 0.699 ± 0.014 |
| gen_only | 0.966 ± 0.007 | 11.31 ± 1.47 | 11.72 ± 1.50 | 0.960 ± 0.006 | 10.85 ± 0.98 | 0.61 ± 0.12 | 0.867 ± 0.052 | 0.950 ± 0.040 | 0.951 ± 0.031 | 0.949 ± 0.051 |
| gen_harmonised | 0.966 ± 0.007 | 15.39 ± 1.94 | 16.82 ± 2.15 | 0.960 ± 0.006 | 15.81 ± 1.94 | 0.61 ± 0.12 | 0.867 ± 0.052 | 0.950 ± 0.040 | 0.951 ± 0.031 | 0.949 ± 0.051 |
| fused_v2_3 | 0.971 ± 0.007 | 19.52 ± 2.22 | 24.82 ± 3.74 | 0.965 ± 0.009 | 18.39 ± 2.21 | 0.56 ± 0.12 | 0.885 ± 0.062 | 0.954 ± 0.037 | 0.961 ± 0.031 | 0.947 ± 0.043 |
| fused_v2_3_1 | 0.966 ± 0.007 | 19.20 ± 1.98 | 24.65 ± 3.33 | 0.960 ± 0.006 | 18.18 ± 2.21 | 0.61 ± 0.12 | 0.867 ± 0.052 | 0.950 ± 0.040 | 0.951 ± 0.031 | 0.949 ± 0.051 |
| oracle_gt_mesh | 0.994 ± 0.003 | 26.97 ± 2.24 | 30.24 ± 3.44 | 0.995 ± 0.002 | 25.93 ± 1.70 | 0.15 ± 0.02 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### paired: `fused_v2_3_1` minus each method, per example (median Δ, wins–losses of `fused_v2_3_1`)

Averages can hide one example carrying a method; the median and the win count do not.

| vs method | IoU | PSNR full | PSNR cov | coverage | Chamfer % (synth.) | F@2% (synth.) |
|---|---|---|---|---|---|---|
| recon_v0 | +0.281 (6–0) | +9.266 (6–0) | +7.302 (6–0) | +0.281 (6–0) | -1.652 (3–0) | +0.329 (3–0) |
| recon_v1_2 | +0.007 (5–1) | -0.215 (3–3) | +0.513 (5–1) | +0.004 (4–2) | -0.906 (3–0) | +0.186 (3–0) |
| gen_only | +0.000 (0–0) | +7.033 (6–0) | +9.927 (6–0) | +0.000 (0–0) | +0.000 (0–0) | +0.000 (0–0) |
| gen_harmonised | +0.000 (0–0) | +3.094 (6–0) | +5.553 (6–0) | +0.000 (0–0) | +0.000 (0–0) | +0.000 (0–0) |
| fused_v2_3 | -0.003 (2–4) | +0.015 (3–3) | +0.022 (3–3) | +0.001 (3–3) | +0.051 (0–3) | -0.006 (1–2) |

### per example: PSNR full / IoU (held-out views)

| method | toy | co3d_cup | co3d_plant | gso_sofa | gso_mug | gso_shoe |
|---|---|---|---|---|---|---|
| recon_v0 | 8.44 / 0.710 | 5.88 / 0.288 | 7.69 / 0.644 | 13.15 / 0.809 | 8.63 / 0.597 | 11.21 / 0.894 |
| recon_v1_2 | 16.84 / 0.939 | 15.57 / 0.947 | 10.94 / 0.843 | 20.94 / 0.966 | 20.89 / 0.967 | 17.56 / 0.974 |
| gen_only | 10.69 / 0.951 | 8.34 / 0.956 | 14.83 / 0.964 | 13.39 / 0.969 | 10.23 / 0.971 | 10.30 / 0.956 |
| gen_harmonised | 14.58 / 0.951 | 13.78 / 0.956 | 16.29 / 0.964 | 17.88 / 0.969 | 15.13 / 0.971 | 13.15 / 0.956 |
| fused_v2_3 | 13.61 / 0.926 | 15.87 / 0.952 | 18.66 / 0.965 | 22.25 / 0.974 | 19.52 / 0.977 | 16.81 / 0.960 |
| fused_v2_3_1 | 15.80 / 0.951 | 15.95 / 0.956 | 18.88 / 0.964 | 21.61 / 0.969 | 19.25 / 0.971 | 16.75 / 0.956 |
| oracle_gt_mesh | needs ground truth | needs ground truth | needs ground truth | 29.00 / 0.996 | 28.06 / 0.997 | 23.84 / 0.990 |
