# Real-Haze Dataset Results

Run date: 2026-09-14 (Asia/Shanghai). The fixed candidate grid was
`alpha = [0, .125, .25, .375, .5, .75, 1]`, with
`WD0375 = A0 + .375 * (WDMamba - A0)`. No alpha or checkpoint was selected
after looking at these test results.

The three evaluations use the dataset-specific ConvIR-B and WDMamba
checkpoints already registered in the clean Dehaze route. The NH-HAZE formal
test is the official `51-55` subset; its earlier 55-image run is retained only
as a split-mixed diagnostic. Metrics are mean
per-image RGB float32 PSNR and the evaluator's pooled `pytorch_msssim` SSIM;
delta is relative to the matching ConvIR-B A0 checkpoint.

## Summary

| Dataset | Profile | Count | PSNR (dB) | SSIM | Delta PSNR (dB) | Positive ratio | Severe loss (<= -0.20 dB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NH-HAZE (official test 51-55) | A0 | 5 | 20.6636 | 0.796807 | 0.0000 | 0.000 | 0 |
| NH-HAZE (official test 51-55) | alpha=.125 | 5 | 20.8883 | 0.805705 | +0.2248 | 1.000 | 0 |
| NH-HAZE (official test 51-55) | alpha=.25 | 5 | 21.0624 | 0.813094 | +0.3988 | 1.000 | 0 |
| NH-HAZE (official test 51-55) | **WD0375** | 5 | 21.1794 | 0.818843 | **+0.5158** | 1.000 | 0 |
| NH-HAZE (official test 51-55) | **alpha=.5 (grid best PSNR)** | 5 | **21.2349** | 0.822814 | **+0.5713** | 0.800 | 0 |
| NH-HAZE (official test 51-55) | alpha=.75 | 5 | 21.1540 | 0.824867 | +0.4904 | 0.600 | 1 |
| NH-HAZE (official test 51-55) | WDMamba | 5 | 20.8306 | 0.818217 | +0.1671 | 0.600 | 2 |
| Dense-Haze | A0 | 55 | 22.7525 | 0.801035 | 0.0000 | 0.000 | 0 |
| Dense-Haze | alpha=.125 | 55 | 23.0213 | 0.806165 | +0.2688 | 0.964 | 0 |
| Dense-Haze | alpha=.25 | 55 | 23.1767 | 0.808038 | +0.4242 | 0.927 | 2 |
| Dense-Haze | **WD0375** | 55 | 23.2079 | 0.806622 | **+0.4554** | 0.800 | 5 |
| Dense-Haze | alpha=.5 | 55 | 23.1140 | 0.801895 | +0.3615 | 0.709 | 13 |
| Dense-Haze | alpha=.75 | 55 | 22.5957 | 0.782661 | -0.1568 | 0.400 | 27 |
| Dense-Haze | WDMamba | 55 | 21.7652 | 0.751545 | -0.9873 | 0.218 | 41 |
| O-HAZE | A0 | 45 | 27.3650 | 0.945877 | 0.0000 | 0.000 | 0 |
| O-HAZE | alpha=.125 | 45 | 27.4542 | 0.946376 | +0.0892 | 0.778 | 0 |
| O-HAZE | alpha=.25 | 45 | 27.4348 | 0.945745 | +0.0698 | 0.689 | 4 |
| O-HAZE | **WD0375** | 45 | 27.3106 | 0.944010 | **-0.0544** | 0.444 | 14 |
| O-HAZE | alpha=.5 | 45 | 27.0935 | 0.941208 | -0.2715 | 0.200 | 27 |
| O-HAZE | alpha=.75 | 45 | 26.4461 | 0.932578 | -0.9189 | 0.133 | 38 |
| O-HAZE | WDMamba | 45 | 25.6205 | 0.920230 | -1.7445 | 0.067 | 42 |

## Interpretation

- The fixed WD0375 profile transfers positively to Dense-Haze (`+0.4554 dB`),
  but is below the ConvIR baseline on NH-HAZE (`-0.1336 dB`) and O-HAZE
  (`-0.0544 dB`). It is therefore not a universal real-haze default.
- The most favorable fixed grid point is dataset-dependent: official NH-HAZE
  peaks at alpha=.5, O-HAZE peaks at alpha=.125, while Dense-Haze peaks at
  WD0375. These maxima
  are descriptive test-grid summaries, not post-test tuning evidence.
- WDMamba alone is below the corresponding ConvIR checkpoint on all three
  datasets. Residual shrinkage is materially safer than alpha=1 here.

## Protocol And O-HAZE Capacity Note

The official NH-HAZE subset (51-55) and Dense-Haze use strict FP32, batch 1, whole-image inference with
reflect padding to ConvIR factor 32 and WDMamba factor 4. O-HAZE contains
original high-resolution files (roughly 2.6k-5.5k pixels on an edge), so the
ConvIR whole-image `unfold` path exceeds 24 GiB on the RTX 4090. O-HAZE was
therefore evaluated at native resolution with explicit ConvIR overlap tiles of
core size 1024 and pad 64; WDMamba remained whole-image. The exact tiling
parameters are recorded in `ohaze/manifest.json`. This is a valid capacity
workaround, but its A0 endpoint is not numerically identical to a hypothetical
whole-image ConvIR run.

All three runs use the fixed seven-value alpha grid, no TTA, no output-image
rounding before metrics, and no test-time checkpoint selection. The O-HAZE
whole-image attempts that failed with OOM are retained on the cloud and were
not overwritten.

## Reproducibility And Evidence

| Dataset | Cloud run | Local compact evidence |
| --- | --- | --- |
| NH-HAZE | `/sda/home/wangyuxin/Dehaze/runs/nhhaze-full-57b822d-20260914` | `nhhaze/alpha_grid.csv`, `nhhaze/manifest.json`, `nhhaze/audit.json` |
| Dense-Haze | `/sda/home/wangyuxin/Dehaze/runs/densehaze-full-57b822d-20260914` | `densehaze/alpha_grid.csv`, `densehaze/manifest.json`, `densehaze/audit.json` |
| O-HAZE | `/sda/home/wangyuxin/Dehaze/runs/ohaze-full-tile-4d5c986-20260914` | `ohaze/alpha_grid.csv`, `ohaze/manifest.json`, `ohaze/audit.json` |

The cloud post-run audit passed for every dataset: 5/5 official NH-HAZE test
rows, 55/55 Dense-Haze rows, and 45/45 O-HAZE rows; 7/7 alpha rows each;
finite numeric fields; and
`DEHAZE_REAL_DATASET_RUN_OK` plus `DEHAZE_WDMAMBA_EVAL_OK` status markers.
Per-image CSVs remain on the cloud runtime and are identified by SHA-256 in
the corresponding audit files.

The previously completed RESIDE/SOTS Indoor and Outdoor results remain in
[`../../sots/20260914/SOTS_RESULTS.md`](../../sots/20260914/SOTS_RESULTS.md).
