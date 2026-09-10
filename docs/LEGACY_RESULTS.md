# Historical Results Used By This Route

These records are imported as prior evidence. The original files remain in the
`onenoober/ConvIR-B` repository and are not copied into this project.

## Fixed WD0375

| Scope | Images | Mean dPSNR | Hard dPSNR | Easy dPSNR | Positive | Severe / 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Haze4K development | 600 | +2.512202 | +3.505615 | +1.189484 | 0.973333 | 11.0 |
| Haze4K test reproduction | 1000 | +1.442090 | +1.529767 | +1.182529 | 0.938000 | 25.8 |

The historical checkpoint identities are fixed in
`configs/convir-4090.json`.

## Cross-Expert Finding

The training-derived v2.6 grid found positive/tail-safe ranges of WDMamba
`0.125-0.5`, FSNet+UDP `0.125-0.75`, and MB-Taylor `0.125`. The later Haze4K
test grids reproduced the general residual-shrinkage pattern but are not treated
as independent selection evidence here.

Primary sources:

- <https://github.com/onenoober/ConvIR-B/blob/main/experience_docx/family_summaries/strongexpert_gainmix_family_summary.md>
- <https://github.com/onenoober/ConvIR-B/blob/main/experience_docx/experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616/v26_all_expert_alpha_grid.csv>
- <https://github.com/onenoober/ConvIR-B/blob/main/experience_docx/experiment_logs/haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616/v210_haze4k_locked_wdmamba_alpha_grid_compact_metrics.csv>
- <https://github.com/onenoober/ConvIR-B/blob/main/experience_docx/experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616/v211_haze4k_locked_cross_expert_alpha_grid_combined_alpha_grid_compact_metrics.csv>

## Reusable Cloud Assets

The WD0375 training cache contains 2400 PNG files at:

```text
/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher/train_core
```

Selected historical visual comparisons are at:

```text
/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked/
experience_docx/experiment_logs/
haze4k_visual_exports_alpha_best_over_anchor_expert_20260617/visuals
```

New route images should be regenerated under this repository's own run root.
