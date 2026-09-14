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
docs/                 Route design and imported historical results
src/dehaze/           Reusable mixture code
tools/                Environment and asset checks
tests/                Lightweight unit tests
```

The clean cloud workspace is:

```text
/sda/home/wangyuxin/Dehaze/
  repo/                Git checkout
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
  --data-root /sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K \
  --split test \
  --out-dir /sda/home/wangyuxin/Dehaze/runs/wdmamba-haze4k-test-YYYYMMDD \
  --convir-its-dir /sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS \
  --convir-dataset Haze4K \
  --a0-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl \
  --wdmamba-repo /sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba \
  --wdmamba-checkpoint /sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth \
  --dataset-name Haze4K \
  --save-images
```

For RESIDE or NH-HAZE, pass `--input-dir` and `--gt-dir` explicitly when the
dataset uses a flat or symlinked layout. `--max-images 1` is useful for a
cloud smoke check before a full run.

For complete SOTS Indoor/Outdoor runs, use `tools/run_sots.sh` and follow
[docs/SOTS_PROTOCOL.md](docs/SOTS_PROTOCOL.md). Original SOTS Indoor requires
`--gt-border 10`; do not resize its GT. The launcher captures logs and commands,
requires all 500 source pairs, and refuses to overwrite an existing run.

To reproduce the historical 1,000-image Haze4K test alpha grid, use
`tools/run_haze4k.sh`. It selects the original Haze4K checkpoints and explicitly
enables the v2.10-compatible 32-grid SSIM convention; the evaluator's default
native-size SSIM remains unchanged for other datasets.

See [docs/ROUTE.md](docs/ROUTE.md) for the staged experiment design and
[docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md) for the reusable historical
evidence.
