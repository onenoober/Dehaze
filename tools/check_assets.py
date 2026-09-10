#!/usr/bin/env python3
"""Validate the external cloud resources declared by a Dehaze config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_path(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    kind = item["kind"]
    exists = path.exists()
    valid = {
        "directory": path.is_dir(),
        "file": path.is_file(),
        "executable": path.is_file() and os.access(path, os.X_OK),
    }.get(kind, False)
    required = bool(item.get("required", True))
    return {
        "id": item["id"],
        "path": str(path),
        "kind": kind,
        "required": required,
        "status": "ok"
        if valid
        else ("missing_optional" if not required and not exists else "error"),
    }


def check_git_source(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    actual = None
    error = None
    try:
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        error = str(exc)
    expected = item["commit"]
    return {
        "id": item["id"],
        "path": str(path),
        "expected_commit": expected,
        "actual_commit": actual,
        "status": "ok" if actual == expected else "error",
        "error": error,
    }


def check_checkpoint(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    actual = sha256(path) if path.is_file() else None
    expected = item["sha256"].lower()
    return {
        "id": item["id"],
        "path": str(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": "ok" if actual == expected else "error",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise SystemExit("unsupported config schema_version")

    checks = {
        "paths": [check_path(item) for item in config.get("paths", [])],
        "git_sources": [
            check_git_source(item) for item in config.get("git_sources", [])
        ],
        "checkpoints": [
            check_checkpoint(item) for item in config.get("checkpoints", [])
        ],
    }
    failures = [
        item
        for group in checks.values()
        for item in group
        if item["status"] == "error"
    ]
    report = {
        "status": "DEHAZE_ASSETS_OK" if not failures else "DEHAZE_ASSETS_ERROR",
        "host": config.get("host"),
        "config": str(args.config),
        "checks": checks,
        "failure_count": len(failures),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
