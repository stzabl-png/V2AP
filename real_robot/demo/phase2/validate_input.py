"""Validate Phase 2 input/ package (schema 1.0)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from demo.phase2.constants import SCHEMA_VERSION
from demo.phase2.table_height import estimate_table_height_m_from_depth


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.ok = self.ok and other.ok


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_input_dir(input_dir: Path, *, check_table_cloud: bool = True) -> ValidationResult:
    """Validate required files and basic consistency."""
    result = ValidationResult(ok=True)
    input_dir = Path(input_dir)

    session_path = input_dir / "session.json"
    if not session_path.is_file():
        result.errors.append(f"Missing {session_path}")
        result.ok = False
        return result

    session = _load_json(session_path)
    schema = session.get("schema_version")
    if schema not in (SCHEMA_VERSION, "1.0"):
        result.errors.append(
            f"session.json schema_version must be {SCHEMA_VERSION!r} (or legacy '1.0'), got {schema!r}"
        )
        result.ok = False
    elif schema == "1.0":
        result.warnings.append(
            "schema_version 1.0 is legacy (ICP-era); re-capture with current capture_session for FP metadata"
        )
    reg_method = session.get("pipeline", {}).get("registration_method")
    if reg_method and reg_method != "foundationpose":
        result.warnings.append(f"Unexpected registration_method={reg_method!r}; expected foundationpose")
    elif reg_method is None and schema == SCHEMA_VERSION:
        result.warnings.append("session.json missing pipeline.registration_method (expected foundationpose)")

    required_rel = [
        "rgb/left_rgb.png",
        "depth/depth.npy",
        "calib/intrinsics.json",
        "calib/K.npy",
        "calib/extrinsics.json",
        "calib/robot_state.json",
        "scene/table.json",
    ]
    for rel in required_rel:
        if not (input_dir / rel).is_file():
            result.errors.append(f"Missing required file: {rel}")
            result.ok = False

    if not result.ok:
        return result

    try:
        import cv2  # noqa: WPS433

        rgb = cv2.imread(str(input_dir / "rgb" / "left_rgb.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            result.errors.append("rgb/left_rgb.png could not be read")
            result.ok = False
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    except Exception as e:
        result.errors.append(f"Failed to read RGB: {e}")
        result.ok = False
        rgb = None

    try:
        depth = np.load(input_dir / "depth" / "depth.npy")
    except Exception as e:
        result.errors.append(f"Failed to load depth.npy: {e}")
        result.ok = False
        depth = None

    intr = _load_json(input_dir / "calib" / "intrinsics.json")
    extr = _load_json(input_dir / "calib" / "extrinsics.json")
    table = _load_json(input_dir / "scene" / "table.json")

    cap = session.get("capture", {})
    if rgb is not None:
        h, w = rgb.shape[:2]
        if cap.get("rgb_width") != w or cap.get("rgb_height") != h:
            result.warnings.append(
                f"session.json rgb size ({cap.get('rgb_width')}×{cap.get('rgb_height')}) "
                f"!= actual PNG ({w}×{h})"
            )

    if depth is not None:
        if depth.ndim != 2:
            result.errors.append(f"depth must be 2D (H,W), got shape {depth.shape}")
            result.ok = False
        elif rgb is not None and depth.shape[:2] != rgb.shape[:2]:
            result.errors.append(
                f"depth shape {depth.shape[:2]} != rgb shape {rgb.shape[:2]} (must be aligned)"
            )
            result.ok = False
        else:
            valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
            frac = float(valid.mean())
            if frac < 0.05:
                result.warnings.append(f"Very few valid depth pixels ({frac * 100:.1f}%)")
            elif frac < 0.2:
                result.warnings.append(f"Low valid depth fraction ({frac * 100:.1f}%)")

    K = np.asarray(intr.get("K"), dtype=np.float64)
    if K.shape != (3, 3):
        result.errors.append(f"intrinsics K must be 3×3, got {K.shape}")
        result.ok = False

    k_npy_path = input_dir / "calib" / "K.npy"
    if k_npy_path.is_file() and result.ok:
        K_npy = np.load(k_npy_path)
        if K_npy.shape != (3, 3):
            result.errors.append(f"calib/K.npy must be 3×3, got {K_npy.shape}")
            result.ok = False
        elif not np.allclose(K, K_npy, rtol=0, atol=1e-6):
            result.errors.append("calib/K.npy does not match intrinsics.json K")
            result.ok = False

    T = np.asarray(extr.get("T_base_cam"), dtype=np.float64)
    if T.shape != (4, 4):
        result.errors.append(f"extrinsics T_base_cam must be 4×4, got {T.shape}")
        result.ok = False

    table_h = table.get("table_height_m")
    if table_h is None:
        result.errors.append("scene/table.json missing table_height_m")
        result.ok = False

    if check_table_cloud and depth is not None and rgb is not None and table_h is not None and result.ok:
        cloud_check = _check_table_height_in_cloud(depth, K, T, float(table_h))
        result.warnings.extend(cloud_check.warnings)
        result.errors.extend(cloud_check.errors)
        result.ok = result.ok and cloud_check.ok

    prompt_path = input_dir / "segment" / "prompt.json"
    if prompt_path.is_file():
        try:
            prompt = _load_json(prompt_path)
            if "prompts" not in prompt:
                result.warnings.append("segment/prompt.json has no 'prompts' key")
        except Exception as e:
            result.warnings.append(f"segment/prompt.json unreadable: {e}")

    return result


def _check_table_height_in_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    table_height_m: float,
    *,
    tol_m: float = 0.05,
) -> ValidationResult:
    """Backproject center-bottom depth pixels; median z in base should be near table."""
    result = ValidationResult(ok=True)
    median_z = estimate_table_height_m_from_depth(depth, K, T_base_cam)
    if median_z is None:
        result.warnings.append(
            "Table height check skipped: too few valid depth pixels in lower-center ROI"
        )
        return result

    err = abs(median_z - table_height_m)
    result.warnings.append(
        f"Table ROI depth → base median z={median_z:.3f} m "
        f"(recorded table_height_m={table_height_m:.3f} m, |Δ|={err:.3f} m)"
    )
    if err > tol_m:
        result.warnings.append(
            f"Table height mismatch: |Δ|={err:.3f} m > {tol_m} m. "
            "Update scene/table.json and planning table_height to the measured value."
        )
    return result
