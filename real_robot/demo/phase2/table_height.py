"""Table top height (m) in robot ``base`` frame from captured session depth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M
from demo.phase2.session_io import SessionInput, load_session_input


def estimate_table_height_m_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    min_valid_pixels: int = 20,
) -> float | None:
    """
    Backproject lower-center depth ROI → median ``z`` in ``base`` frame.

    Same ROI as ``validate_input._check_table_height_in_cloud`` (not ``scene/table.json``).
    """
    h, w = depth.shape
    v0, v1 = int(h * 0.55), h
    u0, u1 = int(w * 0.25), int(w * 0.75)
    patch = depth[v0:v1, u0:u1]
    valid = np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)
    if int(valid.sum()) < min_valid_pixels:
        return None

    zs = patch[valid]
    us = np.broadcast_to(np.arange(u0, u1), patch.shape)[valid]
    vs = np.broadcast_to(np.arange(v0, v1)[:, None], patch.shape)[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (us - cx) * zs / fx
    y_cam = (vs - cy) * zs / fy
    p_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)
    p_base = (T_base_cam @ p_cam.T).T[:, :3]
    return float(np.median(p_base[:, 2]))


def estimate_table_height_m_from_session(
    session_dir: Path,
    *,
    default: float = DEFAULT_TABLE_HEIGHT_M,
) -> tuple[float, str]:
    """
    Return ``(table_height_m, source_label)`` for OMPL planning.

    Priority: depth ROI median → ``default``.

    For candidate filtering use ``candidate_table_collision.planning_table_height_m_from_session``
    (``max(depth, table.json)``).
    """
    input_dir = Path(session_dir) / "input"
    if not input_dir.is_dir():
        return default, f"default ({default:.3f} m, no input/)"

    try:
        sess = load_session_input(input_dir)
    except Exception:
        return default, f"default ({default:.3f} m, input load failed)"

    measured = estimate_table_height_m_from_depth(sess.depth, sess.K, sess.T_base_cam)
    if measured is None:
        return default, f"default ({default:.3f} m, depth ROI too sparse)"
    return measured, f"depth ROI median ({measured:.3f} m)"


def estimate_table_height_m_from_input(sess: SessionInput) -> float | None:
    """Convenience wrapper when ``SessionInput`` is already loaded."""
    return estimate_table_height_m_from_depth(sess.depth, sess.K, sess.T_base_cam)


def load_table_height_m(session_dir: Path, default: float = DEFAULT_TABLE_HEIGHT_M) -> float:
    """Read ``input/scene/table.json`` ``table_height_m`` (packed at capture)."""
    table_path = Path(session_dir) / "input" / "scene" / "table.json"
    if not table_path.is_file():
        return default
    with table_path.open(encoding="utf-8") as f:
        table = json.load(f)
    return float(table.get("table_height_m", default))
