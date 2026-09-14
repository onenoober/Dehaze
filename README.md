# Dehaze

Independent dehazing experiment repository. The first route rebuilds the
successful ConvIR-B + WDMamba fixed residual mixture around `WD0375`:

```text
WD0375 = A0 + 0.375 * (WDMamba - A0)
```

This repository owns experiment code, small configurations, reports, and run
metadata. Datasets, checkpoints, upstream model repositories, generated images,
and other large artifacts remain outside Git and are referenced by explicit
paths and hashes.

## Layout

```text
configs/              Cloud resource and route configuration
assets/               Cloud-only dataset/checkpoint symlink manifest
docs/                 Route design and imported historical results
src/dehaze/           Reusable mixture code
tools/                Environment and asset checks
tests/                Lightweight unit tests
```

The clean cloud workspace is:

```text
/sda/home/wangyuxin/Dehaze/
  repo/                Git checkout
  assets/              Read-only links to datasets and checkpoints
  runs/                One directory per experiment run
  cache/               Regenerable local cache
  logs/                Operational logs
```

Model runs use `convir-4090`. Verify the linked assets there before adding an
evaluator or launching an experiment:

```bash
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python \
  tools/check_assets.py --config configs/convir-4090.json
```

The independent WDMamba evaluator accepts a paired input/GT directory and
writes one immutable run directory. For Haze4K, the default `test/haze` and
`test/gt` layout is sufficient:

```bash
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python \
  tools/evaluate_wdmamba.py \
  --data-root /sda/home/wangyuxin/Dehaze/assets/datasets/Haze4K \
  --split test \
  --out-dir /sda/home/wangyuxin/Dehaze/runs/wdmamba-haze4k-test-YYYYMMDD \
  --convir-its-dir /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS \
  --convir-dataset Haze4K \
  --a0-checkpoint /sda/home/wangyuxin/Dehaze/assets/checkpoints/convir/haze4k-base.pkl \
  --wdmamba-repo /sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba \
  --wdmamba-checkpoint /sda/home/wangyuxin/Dehaze/assets/checkpoints/wdmamba/haze4k_35.88.pth \
  --dataset-name Haze4K \
  --save-images
```

For RESIDE or NH-HAZE, pass `--input-dir` and `--gt-dir` explicitly when the
dataset uses a flat or symlinked layout. `--max-images 1` is useful for a
cloud smoke check before a full run.

The asset links and their source paths, sizes, and SHA-256 values are recorded
in `assets/manifest.json` on the cloud workspace. Dense-Haze now includes both
the ConvIR-B and WDMamba checkpoints; DNH-HAZE 2024 remains intentionally
unlisted until a dataset and model-specific checkpoints are acquired.

For complete SOTS Indoor/Outdoor runs, use `tools/run_sots.sh` and follow
[docs/SOTS_PROTOCOL.md](docs/SOTS_PROTOCOL.md). Original SOTS Indoor requires
`--gt-border 10`; do not resize its GT. The launcher captures logs and commands,
requires all 500 source pairs, and refuses to overwrite an existing run.

The completed NH-HAZE, Dense-Haze, and O-HAZE alpha-grid results are archived
in [reports/real_datasets/20260914/REAL_DATASET_RESULTS.md](reports/real_datasets/20260914/REAL_DATASET_RESULTS.md).
O-HAZE's high-resolution source requires the evaluator's explicit ConvIR
overlap-tile options (`--a0-tile-size 1024 --a0-tile-pad 64`); the default
remains whole-image inference.

To reproduce the historical 1,000-image Haze4K test alpha grid, use
`tools/run_haze4k.sh`. It selects the original Haze4K checkpoints and explicitly
enables the v2.10-compatible 32-grid SSIM convention and historical CUDA backend
defaults; the evaluator's default native-size SSIM and strict numeric profile
remain unchanged for other datasets.

See [docs/ROUTE.md](docs/ROUTE.md) for the staged experiment design and
[docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md) for the reusable historical
evidence.
