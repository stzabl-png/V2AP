"""Write Phase 2 session input/ tree to disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M, DEMO_HEAD_JOINT_POS, HEAD_PITCH_DOWN_DEG
from demo.phase2.constants import (
    BASE_FRAME,
    CAMERA_FRAME,
    DEFAULT_TITAN_MAX_CANDIDATES,
    EE_FRAME,
    FP_DEPTH_MM_SCALE,
    FP_FRAME_INDEX,
    REGISTRATION_METHOD,
    SCHEMA_VERSION,
    SESSIONS_DIR,
    TABLE_COLLISION_CENTER_XYZ_M,
    TABLE_COLLISION_SIZE_XYZ_M,
)
from demo.phase2.extrinsics import extrinsics_json, joint_dict_to_component_arrays
from demo.phase2.validate_input import ValidationResult, validate_input_dir


def make_session_id(object_slug: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    ts = when.strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_object_slug(object_slug)
    return f"{ts}_{slug}"


def _sanitize_object_slug(slug: str) -> str:
    s = slug.strip().lower().replace("-", "_").replace(" ", "_")
    out = "".join(c for c in s if c.isalnum() or c == "_")
    if not out:
        raise ValueError("object_slug must contain at least one alphanumeric character")
    return out


def session_input_dir(session_id: str, sessions_root: Path | None = None) -> Path:
    root = sessions_root or SESSIONS_DIR
    return root / session_id / "input"


def write_session_input(
    input_dir: Path,
    *,
    session_id: str,
    object_slug: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    T_base_cam: np.ndarray,
    joint_pos_dict: dict[str, float],
    robot_model: str,
    table_height_m: float = DEFAULT_TABLE_HEIGHT_M,
    notes: str = "",
    segment_prompt: dict[str, Any] | None = None,
    created_at: datetime | None = None,
    left_hand_joint_pos: np.ndarray | None = None,
    right_hand_joint_pos: np.ndarray | None = None,
    capture_pose_source: str | None = None,
    titan_max_candidates: int | None = None,
) -> Path:
    """Write all required input/ files. Returns input_dir."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H,W,3), got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth.shape}")
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"rgb and depth shape mismatch: {rgb.shape[:2]} vs {depth.shape[:2]}")

    h, w = rgb.shape[:2]
    created = created_at or datetime.now(timezone.utc).astimezone()

    # --- rgb ---
    rgb_dir = input_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    _save_rgb_png(rgb_dir / "left_rgb.png", rgb)

    # --- depth ---
    depth_dir = input_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    np.save(depth_dir / "depth.npy", depth)
    _save_depth_colormap_png(depth_dir / "depth_colormap.png", depth)

    # --- calib ---
    calib_dir = input_dir / "calib"
    calib_dir.mkdir(exist_ok=True)
    _write_json(calib_dir / "intrinsics.json", intrinsics)
    _write_json(calib_dir / "extrinsics.json", extrinsics_json(T_base_cam))
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    np.save(calib_dir / "K.npy", K)

    components = joint_dict_to_component_arrays(joint_pos_dict)
    joints_out = {k: v.tolist() for k, v in components.items()}
    if left_hand_joint_pos is not None:
        joints_out["left_hand"] = np.asarray(left_hand_joint_pos, dtype=np.float64).tolist()
    else:
        joints_out["left_hand"] = []
    if right_hand_joint_pos is not None:
        joints_out["right_hand"] = np.asarray(right_hand_joint_pos, dtype=np.float64).tolist()
    else:
        joints_out["right_hand"] = []

    robot_state = {
        "timestamp_iso": created.isoformat(),
        "joints": joints_out,
        "head_pitch_down_deg": HEAD_PITCH_DOWN_DEG,
        "joint_names_flat": {k: float(v) for k, v in sorted(joint_pos_dict.items())},
    }
    if capture_pose_source:
        robot_state["capture_pose_source"] = capture_pose_source
    _write_json(calib_dir / "robot_state.json", robot_state)

    # --- scene ---
    scene_dir = input_dir / "scene"
    scene_dir.mkdir(exist_ok=True)
    cx, cy, _ = TABLE_COLLISION_CENTER_XYZ_M
    table_json = {
        "table_height_m": float(table_height_m),
        "table_frame_note": "World Z up; table top is plane z = table_height_m in base frame",
        "collision_box": {
            "size_xyz_m": list(TABLE_COLLISION_SIZE_XYZ_M),
            "center_xyz_m": [cx, cy, float(table_height_m) - 0.01],
        },
    }
    _write_json(scene_dir / "table.json", table_json)

    # --- optional segment prompt ---
    if segment_prompt is not None:
        seg_dir = input_dir / "segment"
        seg_dir.mkdir(exist_ok=True)
        _write_json(seg_dir / "prompt.json", segment_prompt)

    # --- session.json (last) ---
    session_json = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "object_slug": _sanitize_object_slug(object_slug),
        "created_at_iso": created.isoformat(),
        "capture": {
            "rgb_file": "rgb/left_rgb.png",
            "depth_file": "depth/depth.npy",
            "depth_preview_file": "depth/depth_colormap.png",
            "depth_unit": "meters",
            "depth_invalid_values": [0.0, None],
            "camera_frame": CAMERA_FRAME,
            "rgb_width": w,
            "rgb_height": h,
            "depth_width": w,
            "depth_height": h,
            "depth_aligned_to_rgb": True,
            "pose_source": capture_pose_source,
            "head_joint_pos_rad": joints_out.get("head", DEMO_HEAD_JOINT_POS.tolist()),
        },
        "robot": {
            "model": robot_model,
            "base_frame": BASE_FRAME,
            "ee_frame": EE_FRAME,
            "state_file": "calib/robot_state.json",
            "extrinsics_file": "calib/extrinsics.json",
        },
        "scene": {
            "table_file": "scene/table.json",
        },
        "pipeline": {
            "registration_method": REGISTRATION_METHOD,
            "titan": {
                "max_candidates": int(
                    titan_max_candidates
                    if titan_max_candidates is not None
                    else DEFAULT_TITAN_MAX_CANDIDATES
                ),
            },
            "foundationpose": {
                "fp_scene_layout": "ycbineoat_reader",
                "frame_index": FP_FRAME_INDEX,
                "depth_storage_input": "depth/depth.npy float32 meters",
                "depth_storage_fp_scene": "uint16 PNG millimeters",
                "depth_mm_scale": FP_DEPTH_MM_SCALE,
                "K_files": ["calib/intrinsics.json", "calib/K.npy"],
                "mask_required": True,
                "mask_source": "output/segment/mask.png on Titan (or input/segment if pre-masked)",
                "mesh_file_on_titan": "output/mesh/object_scaled.glb",
                "ucb_reference": "tools/batch_obj_pose_ego.py run_fp() + prepare_scene layout",
            },
        },
        "notes": notes,
    }
    _write_json(input_dir / "session.json", session_json)

    return input_dir


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import cv2  # noqa: WPS433

        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
    except ImportError:
        from PIL import Image  # noqa: WPS433

        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _save_depth_colormap_png(path: Path, depth: np.ndarray) -> None:
    """Human-readable depth preview (TURBO colormap). Titan uses depth.npy only."""
    import cv2  # noqa: WPS433

    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    if np.any(valid):
        d_min = float(np.min(d[valid]))
        d_max = float(np.max(d[valid]))
        span = max(d_max - d_min, 1e-6)
        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid] = ((d[valid] - d_min) / span).astype(np.float32)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    if not cv2.imwrite(str(path), vis):
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def validate_and_report(input_dir: Path) -> ValidationResult:
    result = validate_input_dir(input_dir)
    return result
