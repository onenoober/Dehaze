"""Fixed residual blending shared by the WD0375 and alpha-grid routes."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_ALPHA_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    return value


def profile_name(expert_prefix: str, alpha: float) -> str:
    """Return the stable historical profile label, for example WD0375."""
    value = _validate_alpha(alpha)
    prefix = expert_prefix.strip().upper()
    if not prefix or not prefix.isalnum():
        raise ValueError("expert_prefix must contain only letters and digits")
    return f"{prefix}{round(value * 1000):04d}"


def blend_outputs(a0: Any, expert: Any, alpha: float, *, clamp: bool = True) -> Any:
    """Compute A0 + alpha * (expert - A0) for NumPy or Torch tensors."""
    value = _validate_alpha(alpha)
    if getattr(a0, "shape", None) != getattr(expert, "shape", None):
        raise ValueError(
            f"shape mismatch: A0={getattr(a0, 'shape', None)!r}, "
            f"expert={getattr(expert, 'shape', None)!r}"
        )
    mixed = a0 + value * (expert - a0)
    if not clamp:
        return mixed
    clamp_method = getattr(mixed, "clamp", None)
    if callable(clamp_method):
        return clamp_method(min=0.0, max=1.0)
    clip_method = getattr(mixed, "clip", None)
    if callable(clip_method):
        return clip_method(0.0, 1.0)
    raise TypeError("blended output must provide clamp() or clip()")
