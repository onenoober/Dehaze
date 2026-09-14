# SOTS evaluation

Evaluate the complete original SOTS Indoor and Outdoor sets (500 hazy images
each) using ConvIR-B ITS and OTS checkpoints respectively. Use the uploaded
WDMamba RESIDE-6K checkpoint for both. Assets are referenced read-only in
`configs/convir-4090.json`; no training, checkpoint selection, or alpha tuning
is performed on SOTS. The fixed grid is 0, .125, .25, .375, .5, .75, 1.
WD0375 remains the preselected profile. Any best test-grid value is descriptive.

Pair each haze filename to its scene-prefix GT. Indoor has 50 unique GTs for
500 inputs; Outdoor has 492 GTs for 500 inputs. The original Indoor GT is
640x480 and haze is 620x460: crop **10 pixels from each GT edge**, without
resampling. Outdoor uses the original equal-sized pair. All selected pairs
must decode and match dimensions before either model loads; unmatched files,
an incorrect source census, and duplicate output stems are fatal.

The crop is supported by the existing read-only asset audit at
`/sda/home/wangyuxin/ConvIR-B/repos/reside-sots-indoor-pixel-registr-reside-sots-indoor-pixe-da82b6d3be2407d2/experience_docx/experiment_logs/reside-sots-indoor-pixel-registration-qualification-v2/reside_sots_indoor_pixel_registration_qualification_v2_summary.json`:
500/500 pairs and 50/50 scenes supported zero residual displacement after the
fixed crop, with valid shifted and wrong-scene controls. This is geometry
evidence, not evidence of restoration quality. Earlier one-image smoke runs
using `--resize-gt` do not establish valid SOTS Indoor benchmark scores.

Inference uses FP32, batch size one, no TTA or tiling. Reflect-pad ConvIR input
to a multiple of 32 and WDMamba input to a multiple of 4, crop each prediction
back to input dimensions, and clamp both to [0,1] before mixing. Metrics are
computed from float outputs, before saving rounded uint8 PNGs.

- PSNR: native-resolution RGB, per-image MSE with floor 1e-12, then mean PSNR
  over all 500 images; no Y-channel conversion or extra evaluation crop.
- SSIM: the independent evaluator's existing `pytorch_msssim` convention:
  adaptive-average-pool both tensors to `(h//d, w//d)`, where
  `d=max(1, round(min(h,w)/256))`, then default Gaussian RGB SSIM. This is
  **pooled SSIM**, not native-resolution SSIM; do not compare unlabeled SSIM
  protocols directly.
- Delta and positive ratio are relative to the matching ConvIR checkpoint.
  Severe loss means per-image PSNR delta <= -0.20 dB. Hard/easy quartiles are
  defined by baseline PSNR, and are descriptive only.

From a clean cloud checkout, after checking that the chosen GPU is available:

```bash
bash tools/run_sots.sh indoor 5 /sda/home/wangyuxin/Dehaze/runs/sots-indoor-UNIQUE_ID
bash tools/run_sots.sh outdoor 6 /sda/home/wangyuxin/Dehaze/runs/sots-outdoor-UNIQUE_ID
```

Use distinct tmux sessions to detach full runs. A fourth positional argument
limits inference for a smoke run, while still verifying the 500-pair census.
Every launch reserves a fresh root; existing paths are rejected. The root
contains a command script, launcher copy, code commit, GPU inventory, log,
and status. `evaluation/` contains progress/failure status, manifest, hashed
pair list, per-image metrics, alpha-grid summary, and seven image directories.
Commit only the compact text evidence and report after completion.

These runs test the available pretrained models on SOTS. ConvIR ITS/OTS and
WDMamba RESIDE-6K use different training protocols. The filename `32.15` is
not an expected score or an independently reproduced official benchmark.
