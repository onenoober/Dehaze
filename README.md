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

See [docs/ROUTE.md](docs/ROUTE.md) for the staged experiment design and
[docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md) for the reusable historical
evidence.
