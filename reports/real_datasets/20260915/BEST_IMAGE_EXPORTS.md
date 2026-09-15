# Best Profile Image Exports

Run date: 2026-09-15 (Asia/Shanghai). These exports reuse the completed fixed
alpha-grid evaluations. A profile is selected from the archived aggregate PSNR
for each dataset; this is an image-export decision, not a new test-time model
or checkpoint selection. `WD0375` is retained as the successful expert route
reference where it is not the aggregate grid maximum.

| Dataset / split | Archived grid-best profile | Profiles saved | Image count | Cloud image root | Protocol |
| --- | --- | --- | ---: | --- | --- |
| Haze4K test | alpha=.75 (36.2031 dB) | alpha0.75, WD0375 | 2,000 | `/sda/home/wangyuxin/Dehaze/runs/haze4k-best-images-0302705-20260915/evaluation/images` | historical full-image evaluator |
| SOTS Indoor | A0 / ConvIR ITS (42.7214 dB) | all 7 grid profiles | 3,500 | `/sda/home/wangyuxin/Dehaze/runs/sots-indoor-full-eacd5db-20260914/evaluation/images` | existing complete export |
| SOTS Outdoor | alpha=.125 (37.4894 dB) | all 7 grid profiles | 3,500 | `/sda/home/wangyuxin/Dehaze/runs/sots-outdoor-full-eacd5db-20260914/evaluation/images` | existing complete export |
| NH-HAZE official test 51-55 | alpha=.5 (21.2349 dB) | alpha0.5, WD0375 | 10 | `/sda/home/wangyuxin/Dehaze/runs/nhhaze-best-images-official-f4f9c05-20260915/evaluation/images` | strict FP32, official 5-image test subset |
| Dense-Haze test | WD0375 (23.2079 dB) | WD0375 | 55 | `/sda/home/wangyuxin/Dehaze/runs/densehaze-best-images-f4f9c05-20260915/evaluation/images` | strict FP32, whole-image |
| O-HAZE test | alpha=.125 (27.4542 dB) | alpha0.125, WD0375 | 90 | `/sda/home/wangyuxin/Dehaze/runs/ohaze-best-images-wdmamba-tile-fallback-54ff735-20260915/evaluation/images` | ConvIR tile 1024/64; WDMamba tile 1024/64 fallback |

The Haze4K, SOTS, NH-HAZE, and Dense-Haze image roots contain outputs from
the same whole-image model protocol as their corresponding metric runs. The
O-HAZE archived metrics used whole-image WDMamba and ConvIR tile 1024/64, but
the current GPU allocation could not hold whole-image WDMamba; its export uses
the explicit WDMamba overlap-tile fallback and is therefore visual-only, not a
replacement for the archived O-HAZE aggregate metrics. The fallback details
are recorded in its `evaluation/manifest.json`.

Only prediction PNGs are stored on the cloud runtime. Input images, GT images,
weights, and raw inference artifacts are not copied into Git.
