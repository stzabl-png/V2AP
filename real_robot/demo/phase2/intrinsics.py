"""Parse camera intrinsics from dexcontrol camera_info or CLI overrides."""

from __future__ import annotations

from typing import Any

import numpy as np

from demo.phase2.constants import CAMERA_FRAME, FALLBACK_INTRINSICS_640x360


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_intrinsic_fields(node: Any, found: dict[str, float]) -> None:
    """Recursively search nested dicts for fx/fy/cx/cy or a 3×3 K."""
    if isinstance(node, dict):
        for key, val in node.items():
            key_l = str(key).lower()
            if key_l in ("fx", "fy", "cx", "cy"):
                f = _as_float(val)
                if f is not None:
                    found[key_l] = f
            elif key_l in ("k", "camera_matrix", "intrinsic_matrix", "intrinsics"):
                k = _parse_k_matrix(val)
                if k is not None:
                    found["fx"] = float(k[0, 0])
                    found["fy"] = float(k[1, 1])
                    found["cx"] = float(k[0, 2])
                    found["cy"] = float(k[1, 2])
            elif key_l in ("left", "left_rgb", "rgb", "calibration", "camera"):
                _find_intrinsic_fields(val, found)
            else:
                _find_intrinsic_fields(val, found)
    elif isinstance(node, (list, tuple)):
        k = _parse_k_matrix(node)
        if k is not None:
            found["fx"] = float(k[0, 0])
            found["fy"] = float(k[1, 1])
            found["cx"] = float(k[0, 2])
            found["cy"] = float(k[1, 2])


def _parse_k_matrix(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


def build_intrinsics_json(
    width: int,
    height: int,
    camera_info: dict[str, Any] | None = None,
    *,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """
    Build calib/intrinsics.json content.

    Priority: explicit CLI overrides > parsed camera_info > resolution-scaled fallback.
    """
    fields: dict[str, float] = {}
    if camera_info is not None:
        _find_intrinsic_fields(camera_info, fields)

    for name, override in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        if override is not None:
            fields[name] = float(override)

    if "cx" not in fields:
        fields["cx"] = (width - 1) / 2.0
    if "cy" not in fields:
        fields["cy"] = (height - 1) / 2.0

    if "fx" not in fields or "fy" not in fields:
        if not allow_fallback:
            raise RuntimeError(
                "Camera intrinsics fx/fy not found in camera_info and fallback disabled. "
                "Pass --fx/--fy or --intrinsics-json."
            )
        scale_x = width / 640.0
        scale_y = height / 360.0
        fields.setdefault("fx", FALLBACK_INTRINSICS_640x360["fx"] * scale_x)
        fields.setdefault("fy", FALLBACK_INTRINSICS_640x360["fy"] * scale_y)

    fx_v = float(fields["fx"])
    fy_v = float(fields["fy"])
    cx_v = float(fields["cx"])
    cy_v = float(fields["cy"])

    return {
        "camera_frame": CAMERA_FRAME,
        "width": int(width),
        "height": int(height),
        "K": [
            [fx_v, 0.0, cx_v],
            [0.0, fy_v, cy_v],
            [0.0, 0.0, 1.0],
        ],
        "distortion_model": "none",
        "dist_coeffs": [],
        "source": "camera_info" if camera_info is not None and fx is None else "cli_or_fallback",
    }
