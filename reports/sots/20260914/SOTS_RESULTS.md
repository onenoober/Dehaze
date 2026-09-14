# SOTS full evaluation: ConvIR-B + WDMamba residual grid

Run date: 2026-09-14 (Asia/Shanghai). Code commit: `eacd5db87de957417454fbc49b2466850971a56f`.

The complete original RESIDE SOTS sets were evaluated: 500 Indoor pairs and
500 Outdoor pairs. Indoor uses the ConvIR ITS checkpoint and a 10-pixel crop on
each edge of the 640x480 GT (no interpolation), producing the native 620x460
evaluation size. Outdoor uses the ConvIR OTS checkpoint at its native size.
WDMamba uses the uploaded RESIDE-6K checkpoint for both splits. The fixed
candidate is `A0 + alpha * (WDMamba - A0)` with alpha in
`[0, .125, .25, .375, .5, .75, 1]`; `WD0375` is alpha=.375.

## Mean results

PSNR and pooled SSIM are means over 500 images. Delta is relative to the
matching ConvIR baseline. The alpha grid was fixed before this test; the best
grid row is descriptive and was not used to tune a model.

| Split | Profile | PSNR (dB) | SSIM | Delta PSNR (dB) | Positive ratio | Severe loss (<= -0.20 dB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Indoor | A0 / ConvIR ITS | 42.7214 | 0.996669 | 0.0000 | 0.000 | 0 |
| Indoor | alpha=.125 | 41.9455 | 0.996296 | -0.7759 | 0.170 | 381 |
| Indoor | **WD0375** | **38.2944** | **0.993708** | **-4.4270** | **0.000** | **500** |
| Indoor | WDMamba (alpha=1) | 31.4276 | 0.978564 | -11.2938 | 0.000 | 500 |
| Outdoor | A0 / ConvIR OTS | 37.4075 | 0.993788 | 0.0000 | 0.000 | 0 |
| Outdoor | **alpha=.125 (grid best PSNR)** | **37.4894** | **0.994009** | **+0.0819** | **0.590** | **144** |
| Outdoor | WD0375 | 36.8528 | 0.993597 | -0.5548 | 0.368 | 287 |
| Outdoor | WDMamba (alpha=1) | 33.2614 | 0.987677 | -4.1461 | 0.130 | 430 |

The remaining grid rows are in `indoor/alpha_grid.csv` and
`outdoor/alpha_grid.csv`. Indoor declines monotonically from the baseline over
this grid. Outdoor has a small improvement at alpha=.125, while the previously
successful Haze4K profile WD0375 is below the OTS baseline. Therefore the
Haze4K WD0375 success does not transfer as a default SOTS profile.

## Reproducibility and audit

- ConvIR ITS SHA-256: `0c71389bb8e4a602e4548e90c0015941b78698dfbfdd6ddfd46cdd2f61cd1fbd`.
- ConvIR OTS SHA-256: `dc28713ad92af0a2594b1964451602614e1b24a1e898647a6cf41c94e9e533e6`.
- WDMamba RESIDE-6K SHA-256: `9dc852540db5c1d2b04cfa689acda994d4dcf3e2cbb820a5da49db1ba4073a35`.
- Upstream source commits: ConvIR `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`; WDMamba `f3b13952f31d30bd945934b845b39cf96152beeb`.
- PSNR: native-resolution RGB float output, per-image MSE floor `1e-12`, then mean PSNR.
- SSIM: the evaluator's documented pooled `pytorch_msssim` protocol; it is not a native-resolution SSIM claim.
- Inference: FP32, batch 1, no TTA/tiling, reflect padding (ConvIR factor 32, WDMamba factor 4), crop to input, clamp to [0,1].
- Output audit: both runs have `COMPLETE` manifests, 500 per-image rows, 7 alpha rows, and 3,500/3,500 RGB PNGs that decode with expected dimensions.

Cloud run roots (images and per-image metrics remain outside Git):

```text
/sda/home/wangyuxin/Dehaze/runs/sots-indoor-full-eacd5db-20260914
/sda/home/wangyuxin/Dehaze/runs/sots-outdoor-full-eacd5db-20260914
```

The compact evidence beside this report contains each alpha summary, manifest,
and output audit. The scores are an evaluation of these available pretrained
weights on SOTS, not a claim that WDMamba RESIDE-6K reproduces an official
SOTS-trained benchmark.
