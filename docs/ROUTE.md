# WD0375 Clean Route

## Objective

Build a small, independent and reproducible inference route around ConvIR-B A0
and WDMamba. The primary candidate is fixed before execution:

```text
WD0375 = A0 + 0.375 * (WDMamba - A0)
```

This is a two-model inference mixture, not a new single-model architecture.
Every evaluated image requires both A0 and the selected expert unless endpoint
outputs are already cached.

## Phase 1: WDMamba Reproduction

Render A0 and WDMamba once per image, then derive the fixed alpha grid
`0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0` from those endpoints. WD0375 is the
primary profile; the other rows characterize the gain-risk curve and are not a
post-test search.

Required outputs for each run:

- `manifest.json`: source commits, checkpoint hashes, dataset identity, command;
- `per_image.csv`: A0, expert and every alpha profile metric by image;
- `alpha_grid.csv`: mean, hard/easy, positive ratio and severe-tail summary;
- `parity.json`: comparison with the historical v2.6 development result;
- `resource_summary.json`: elapsed time and peak GPU memory;
- `images/<profile>/`: regenerated images when image export is enabled.

## Phase 2: Cross-Expert Reproduction

Add FSNet+UDP and MB-TaylorFormerV2-L only after Phase 1 has exact A0/WDMamba
identity and metric parity. Keep each expert behind an explicit adapter because
their official loaders differ:

- FSNet+UDP uses official-style `depth2l` PNG input, factor-8 padding, and the
  historical `num_heads=1 -> 2` construction fix required for strict loading.
- MB-TaylorFormerV2-L uses its official configuration, factor-8 reflect padding,
  and the original non-strict checkpoint loading behavior.

The historical development-safe candidate sets are WDMamba `0.125-0.5`,
FSNet+UDP `0.125-0.75`, and MB-Taylor `0.125`. Full endpoint rows remain useful
negative controls.

For the completed real-haze alpha grid, NH-HAZE and Dense-Haze run whole-image
FP32 inference. The original O-HAZE files are high resolution and exceed the
24 GiB ConvIR whole-image memory path, so its reproducible run explicitly uses
ConvIR overlap tiles (`1024` core, `64` context pad). This option is recorded
in the O-HAZE manifest and is not enabled by default for other datasets.

## Evaluation Roles

Historical Haze4K test metrics are already known. A rerun on that split is a
reproduction benchmark, not a new blind confirmation. Candidate development
should use the Haze4K training-derived split; a new confirmation claim requires
a genuinely independent dataset or newly reserved population.

## Cloud Layout

Each launch creates a new immutable run directory:

```text
/sda/home/wangyuxin/Dehaze/runs/<run-id>/
  command.sh
  manifest.json
  status.txt
  runtime.log
  metrics/
  images/
```

External model repositories, datasets, and checkpoints stay read-only at the
paths declared in `configs/convir-4090.json`.
