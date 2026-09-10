# Repository Instructions

- Treat this repository as an independent project. Do not import the ConvIR-B
  experiment governance or route-control framework.
- Use `convir-4090` for model inference, evaluation, training, and image
  generation. Lightweight source inspection, formatting, and unit tests may be
  performed locally.
- Keep cloud outputs under `/sda/home/wangyuxin/Dehaze/`.
- Treat every external dataset, checkpoint, and model repository referenced by
  `configs/convir-4090.json` as read-only.
- Do not commit datasets, checkpoints, generated images, arrays, caches, or raw
  run directories. Commit compact metrics, manifests, and written conclusions.
- Create a new run directory for every launch; never overwrite an existing run.
