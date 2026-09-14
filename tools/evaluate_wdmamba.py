#!/usr/bin/env python3
"""Evaluate ConvIR-B, WDMamba and their fixed residual alpha grid.

The evaluator is intentionally independent from the historical ConvIR-B
experiment scripts.  It loads the two upstream models from explicitly supplied
read-only paths, pairs images by conservative filename rules, and writes all
run artifacts below one caller-owned output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
import torchvision.transforms.functional as TVF

DEFAULT_ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_key(alpha: float) -> str:
    return (f"a{alpha:.6f}".rstrip("0").rstrip(".")).replace(".", "p")


def alpha_label(alpha: float) -> str:
    if abs(alpha - 0.375) < 1e-9:
        return "WD0375"
    if abs(alpha) < 1e-9:
        return "A0"
    if abs(alpha - 1.0) < 1e-9:
        return "WDMamba"
    return f"alpha{alpha:g}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pad_to(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    pad_h = (factor - height % factor) % factor
    pad_w = (factor - width % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height + pad_h, width + pad_w


def metric_pair(pred: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    mse = F.mse_loss(pred, label).clamp_min(1e-12)
    psnr = float((10.0 * torch.log10(1.0 / mse)).item())
    _, _, height, width = pred.shape
    down = max(1, round(min(height, width) / 256))
    target_size = (max(1, height // down), max(1, width // down))
    score = ssim(
        F.adaptive_avg_pool2d(pred, target_size),
        F.adaptive_avg_pool2d(label, target_size),
        data_range=1.0,
        size_average=False,
    ).mean()
    return psnr, float(score.item())


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing image directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def gt_candidates(name: str) -> Iterable[str]:
    path = Path(name)
    stem, suffix = path.stem, path.suffix
    yielded: set[str] = set()

    def add(value: str) -> Iterable[str]:
        if value not in yielded:
            yielded.add(value)
            yield value

    yield from add(name)
    yield from add(f"{stem}.png")
    if "_hazy" in stem.lower():
        yield from add(f"{stem.lower().replace('_hazy', '_GT')}.png")
        yield from add(f"{stem.lower().replace('_hazy', '_gt')}.png")
    if "_gt" in stem.lower():
        yield from add(f"{stem.lower().replace('_gt', '_hazy')}{suffix}")
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        yield from add(f"{prefix}.png")
    # RESIDE ITS/6K haze names are usually <scene>_<variant>_... while GT is
    # <scene>.png.  The first numeric token is therefore the final fallback.
    numeric = stem.split("_", 1)[0]
    if numeric.isdigit():
        yield from add(f"{numeric.zfill(4)}.png")


def pair_paths(input_dir: Path, gt_dir: Path, limit: int = 0) -> list[tuple[Path, Path]]:
    inputs = image_files(input_dir)
    gt_map = {path.name: path for path in image_files(gt_dir)}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for input_path in inputs:
        gt_path = next((gt_map[candidate] for candidate in gt_candidates(input_path.name) if candidate in gt_map), None)
        if gt_path is None:
            missing.append(input_path.name)
        else:
            pairs.append((input_path, gt_path))
        if limit > 0 and len(pairs) >= limit:
            break
    if missing and not pairs:
        raise RuntimeError(f"no input/GT pairs found; first missing={missing[:5]}")
    if missing and len(missing) > max(5, len(inputs) // 2):
        raise RuntimeError(
            f"only {len(pairs)} pairs found from {len(inputs)} inputs; "
            f"first missing={missing[:5]}"
        )
    return pairs


def resolve_dirs(data_root: Path, split: str, input_dir: str | None, gt_dir: str | None) -> tuple[Path, Path]:
    if input_dir and gt_dir:
        return Path(input_dir), Path(gt_dir)
    root = data_root / split
    input_path = next((root / name for name in ("haze", "hazy", "IN") if (root / name).is_dir()), None)
    gt_path = next((root / name for name in ("gt", "GT") if (root / name).is_dir()), None)
    if input_path is None or gt_path is None:
        raise FileNotFoundError(
            f"could not resolve paired dirs below {root}; pass --input-dir and --gt-dir explicitly"
        )
    return input_path, gt_path


def load_convir_builders(convir_its_dir: Path):
    sys.path.insert(0, str(convir_its_dir))
    from models.ConvIR import build_net  # type: ignore

    return build_net


def load_a0(build_net: Any, checkpoint: Path, dataset_name: str, device: torch.device):
    model = build_net("base", dataset_name, "original").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_wdmamba(repo: Path, checkpoint: Path, device: torch.device) -> tuple[Any, int]:
    # WDMamba's vendored basicsr package must be isolated from ConvIR's package.
    # Its mamba_ssm dependency imports two decoder output classes removed from
    # newer Transformers releases; the classes are only used for text
    # generation, so lightweight placeholders preserve image-model loading.
    try:
        import transformers.generation as generation
        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, type(name, (object,), {}))
    except Exception:
        pass
    for key in list(sys.modules):
        if key == "basicsr" or key.startswith("basicsr."):
            del sys.modules[key]
    package = types.ModuleType("basicsr")
    package.__path__ = [str(repo / "basicsr")]  # type: ignore[attr-defined]
    sys.modules["basicsr"] = package
    for sub in ("archs", "utils"):
        child = types.ModuleType(f"basicsr.{sub}")
        child.__path__ = [str(repo / "basicsr" / sub)]  # type: ignore[attr-defined]
        sys.modules[f"basicsr.{sub}"] = child
    load_module("basicsr.utils.registry", repo / "basicsr/utils/registry.py")
    for filename in ("Ublock.py", "detail_enhance_net.py", "wavelet.py"):
        load_module(f"basicsr.archs.{filename[:-3]}", repo / "basicsr/archs" / filename)
    wavemamba = load_module("basicsr.archs.wavemamba_arch", repo / "basicsr/archs/wavemamba_arch.py")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["params"] if isinstance(state, dict) and "params" in state else state
    # The released real-haze NH checkpoint uses DENet(3, 4), while synthetic
    # checkpoints use DENet(3, 6). Infer this architecture detail from the
    # checkpoint instead of silently allowing a partial load.
    de_blocks = 4 if any(key.endswith("DE.g1.gp.4.weight") for key in state) else 6
    if de_blocks != 6:
        default_denet = wavemamba.DENet
        wavemamba.DENet = lambda in_chn, _unused: default_denet(in_chn, de_blocks)
    model = wavemamba.WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, de_blocks


def infer_a0(model: Any, image: torch.Tensor) -> torch.Tensor:
    _, _, height, width = image.shape
    padded, _, _ = pad_to(image, 32)
    output = model(padded)
    if isinstance(output, (list, tuple)):
        output = output[0][2] if isinstance(output[0], (list, tuple)) else output[2]
    else:
        raise TypeError(f"unexpected ConvIR output: {type(output)!r}")
    return torch.clamp(output[:, :, :height, :width], 0, 1)


def infer_wdmamba(model: Any, image: torch.Tensor) -> torch.Tensor:
    _, _, height, width = image.shape
    padded, _, _ = pad_to(image, 4)
    output = model.restoration_network(padded)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return torch.clamp(output[:, :, :height, :width], 0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def summarize(rows: list[dict[str, Any]], alphas: list[float]) -> list[dict[str, Any]]:
    if not rows:
        return []
    baseline = [float(row["A0_PSNR"]) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: baseline[index])
    bucket = max(1, len(rows) // 4)
    summaries: list[dict[str, Any]] = []
    for alpha in alphas:
        key = alpha_key(alpha)
        deltas = [float(row[f"alpha_{key}_dPSNR"]) for row in rows]
        dssim = [float(row[f"alpha_{key}_dSSIM"]) for row in rows]
        summaries.append({
            "alpha": alpha,
            "label": alpha_label(alpha),
            "count": len(rows),
            "mean_PSNR": statistics.mean(float(row[f"alpha_{key}_PSNR"]) for row in rows),
            "mean_SSIM": statistics.mean(float(row[f"alpha_{key}_SSIM"]) for row in rows),
            "mean_dPSNR": statistics.mean(deltas),
            "hard_bottom25_dPSNR": statistics.mean(deltas[index] for index in order[:bucket]),
            "easy_top25_dPSNR": statistics.mean(deltas[index] for index in order[-bucket:]),
            "mean_dSSIM": statistics.mean(dssim),
            "positive_ratio": sum(value > 0 for value in deltas) / len(deltas),
            "severe_loss_count": sum(value <= -0.20 for value in deltas),
            "severe_loss_per_600": sum(value <= -0.20 for value in deltas) / len(deltas) * 600.0,
            "worst_dPSNR": min(deltas),
            "best_dPSNR": max(deltas),
        })
    return summaries


def evaluate(args: argparse.Namespace) -> None:
    output_root = args.out_dir
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir, gt_dir = resolve_dirs(args.data_root, args.split, args.input_dir, args.gt_dir)
    pairs = pair_paths(input_dir, gt_dir, args.max_images)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    build_net = load_convir_builders(args.convir_its_dir)
    a0 = load_a0(build_net, args.a0_checkpoint, args.convir_dataset, device)
    wdmamba, wdmamba_de_blocks = load_wdmamba(args.wdmamba_repo, args.wdmamba_checkpoint, device)
    alphas = sorted({round(float(value), 6) for value in args.alphas})
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, (input_path, gt_path) in enumerate(pairs, 1):
        hazy = TVF.to_tensor(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
        label = TVF.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        label_height, label_width = label.shape[-2:]
        if label.shape[-2:] != hazy.shape[-2:]:
            if not args.resize_gt:
                raise ValueError(
                    f"input/GT size mismatch for {input_path.name}: "
                    f"input={tuple(hazy.shape[-2:])}, gt={tuple(label.shape[-2:])}; "
                    "pass --resize-gt only when the dataset protocol requires it"
                )
            label = F.interpolate(label, size=hazy.shape[-2:], mode="bilinear", align_corners=False)
        with torch.no_grad():
            a0_pred = infer_a0(a0, hazy)
            expert_pred = infer_wdmamba(wdmamba, hazy)
        a0_psnr, a0_ssim = metric_pair(a0_pred, label)
        expert_psnr, expert_ssim = metric_pair(expert_pred, label)
        row: dict[str, Any] = {
            "image_id": input_path.stem,
            "input": str(input_path),
            "gt": str(gt_path),
            "width": hazy.shape[-1],
            "height": hazy.shape[-2],
            "gt_width": label_width,
            "gt_height": label_height,
            "gt_resized": bool((label_height, label_width) != tuple(hazy.shape[-2:])),
            "A0_PSNR": a0_psnr,
            "A0_SSIM": a0_ssim,
            "WDMamba_PSNR": expert_psnr,
            "WDMamba_SSIM": expert_ssim,
            "WDMamba_dPSNR": expert_psnr - a0_psnr,
            "WDMamba_dSSIM": expert_ssim - a0_ssim,
        }
        for alpha in alphas:
            key = alpha_key(alpha)
            prediction = torch.clamp(a0_pred + alpha * (expert_pred - a0_pred), 0, 1)
            psnr, score = metric_pair(prediction, label)
            row[f"alpha_{key}_PSNR"] = psnr
            row[f"alpha_{key}_SSIM"] = score
            row[f"alpha_{key}_dPSNR"] = psnr - a0_psnr
            row[f"alpha_{key}_dSSIM"] = score - a0_ssim
            if args.save_images:
                save_image(prediction, output_root / "images" / alpha_label(alpha) / f"{input_path.stem}.png")
        rows.append(row)
        if index % args.print_freq == 0 or index == len(pairs):
            print(f"progress={index}/{len(pairs)} elapsed={time.time()-started:.1f}s", flush=True)
    write_csv(output_root / "metrics" / "per_image.csv", rows)
    write_csv(output_root / "metrics" / "alpha_grid.csv", summarize(rows, alphas))
    manifest = {
        "mode": "evaluate",
        "dataset": args.dataset_name,
        "split": args.split,
        "input_dir": str(input_dir),
        "gt_dir": str(gt_dir),
        "count": len(rows),
        "alpha_grid": alphas,
        "device": str(device),
        "a0_checkpoint": str(args.a0_checkpoint),
        "a0_sha256": sha256(args.a0_checkpoint),
        "wdmamba_checkpoint": str(args.wdmamba_checkpoint),
        "wdmamba_sha256": sha256(args.wdmamba_checkpoint),
        "wdmamba_de_blocks": wdmamba_de_blocks,
        "convir_its_dir": str(args.convir_its_dir),
        "wdmamba_repo": str(args.wdmamba_repo),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "status.txt").write_text("DEHAZE_WDMAMBA_EVAL_OK\n", encoding="utf-8")
    print(f"DEHAZE_WDMAMBA_EVAL_OK count={len(rows)} out={output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--convir-its-dir", type=Path, required=True)
    parser.add_argument("--convir-dataset", default="Haze4K", help="dataset tag accepted by ConvIR build_net")
    parser.add_argument("--a0-checkpoint", type=Path, required=True)
    parser.add_argument("--wdmamba-repo", type=Path, required=True)
    parser.add_argument("--wdmamba-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--resize-gt",
        action="store_true",
        help="resize GT to input resolution for datasets whose paired files differ in size",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
