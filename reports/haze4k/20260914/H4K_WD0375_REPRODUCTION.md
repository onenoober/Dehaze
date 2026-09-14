# Haze4K WD0375 reproduction

Run date: 2026-09-14. Formal reproduction code commit: `08374fab06a87030f05091485c563ed242743ce6`.

The original Haze4K `test` split was evaluated again on all 1,000 images with
the original ConvIR-B Haze4K checkpoint and WDMamba Haze4K checkpoint. The
candidate grid was fixed to `alpha = [0, .125, .25, .375, .5, .75, 1]`, with
`WD0375 = alpha=.375`.

The formal run uses the historical CUDA numeric profile of the old v2.10
evaluator (`cudnn_deterministic=false`, default backend TF32 settings) and the
historical 32-grid SSIM reference. This is required for exact reproduction;
the independent evaluator still defaults to its stricter deterministic profile
for new datasets.

## Exact comparison

| Profile | Old v2.10 PSNR | New PSNR | Difference | Old SSIM grid32 | New SSIM grid32 | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 / ConvIR-B | 34.1455015774 | 34.1455015774 | 0 | 0.9896190688 | 0.9896190688 | 0 |
| alpha=.125 | 34.6752765255 | 34.6752765255 | 0 | 0.9906085477 | 0.9906085477 | 0 |
| alpha=.25 | 35.1637587214 | 35.1637587214 | 0 | 0.9914327065 | 0.9914327065 | 0 |
| **WD0375 / alpha=.375** | **35.5875913658** | **35.5875913658** | **0** | **0.9920900025** | **0.9920900025** | **0** |
| alpha=.5 | 35.9204599171 | 35.9204599171 | 0 | 0.9925777691 | 0.9925777691 | 0 |
| alpha=.75 | 36.2031388302 | 36.2031388302 | 0 | 0.9930264474 | 0.9930264474 | 0 |
| WDMamba full / alpha=1 | 35.9171470776 | 35.9171470776 | 0 | 0.9927112212 | 0.9927112212 | 0 |

The 1,000-row new per-image CSV was matched by image stem to the historical
v2.2 per-image CSV. For A0, WDMamba, and WD0375, both PSNR and SSIM had
`max_abs=0` and `mean_abs=0` across all 1,000 matched rows. The seven alpha
summary rows also had zero difference against v2.10.

## Audit and interpretation

The formal run has `COMPLETE` status, 1,000 per-image rows, seven alpha rows,
finite numeric fields, and a passing parity/output audit. Checkpoint hashes are
ConvIR Haze4K `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
and WDMamba Haze4K `57ff24c3791e593f0172607fea66252a8ba5475ab0e417f4cf48e72b4c9a36da`.

Two earlier full runs used the new evaluator's strict TF32/deterministic
profile. Their metrics were stable with each other but differed from the old
run on a few numerically sensitive images, including image `508_0.72_1.04`.
Those runs are retained for traceability but are not used as the reproduction
result. A targeted repeat under historical settings matched the old values,
and the final full historical run passed exact parity.

Formal cloud run:

```text
/sda/home/wangyuxin/Dehaze/runs/haze4k-full-r3-08374fa-20260914
```

Compact evidence in this directory: `alpha_grid.csv`, `manifest.json`,
`parity.json`, and `audit.json`. The full per-image CSV and run log remain on
the cloud runtime; model weights and datasets are not committed.
