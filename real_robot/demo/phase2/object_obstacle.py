"""Collision box for Titan-registered object (pre-grasp planning only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from demo.phase2.retarget import TitanSessionOutput


def _mesh_aabb_in_mesh_frame(output: TitanSessionOutput) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Object axis-aligned bounds in ``base_aligned`` mesh coordinates.

    Priority:
      1. ``candidates.json`` ``mesh_aabb_min_m`` / ``mesh_aabb_max_m`` (if present)
      2. Vertex bounds of ``output/mesh/object_base_aligned.glb`` (trimesh)
      3. Symmetric box ``± mesh_span_m / 2`` about mesh origin (legacy Titan export)
    """
    cands = output.candidates
    if "mesh_aabb_min_m" in cands and "mesh_aabb_max_m" in cands:
        aabb_min = np.asarray(cands["mesh_aabb_min_m"], dtype=np.float64)
        aabb_max = np.asarray(cands["mesh_aabb_max_m"], dtype=np.float64)
        return aabb_min, aabb_max, "candidates mesh_aabb_min/max"

    glb_path = Path(output.session_dir) / "output" / "mesh" / "object_base_aligned.glb"
    if glb_path.is_file():
        try:
            import trimesh

            mesh = trimesh.load(glb_path, force="mesh")
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            return bounds[0], bounds[1], f"glb vertex bounds ({glb_path.name})"
        except ImportError:
            pass

    half = 0.5 * np.asarray(output.mesh_span_m, dtype=np.float64)
    return -half, half, "mesh_span_m symmetric about mesh origin"


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4×4 ``T`` to N×3 points."""
    assert T.shape == (4, 4)
    assert points.ndim == 2 and points.shape[1] == 3
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def object_collision_box(
    output: TitanSessionOutput,
    *,
    padding: float = 1.08,
    min_extent_m: float = 0.06,
    name: str = "titan_object",
) -> dict:
    """
    Axis-aligned box in robot **base** frame covering the registered object mesh.

    The mesh AABB is taken in ``base_aligned`` mesh coordinates, optionally padded,
    then its eight corners are transformed by ``T_base_mesh``; the result is the
    tight axis-aligned box in base (handles small non-diagonal ``R`` in ``T_base_mesh``).
    """
    aabb_min, aabb_max, aabb_source = _mesh_aabb_in_mesh_frame(output)
    center_mesh = 0.5 * (aabb_min + aabb_max)
    half_mesh = 0.5 * (aabb_max - aabb_min) * float(padding)
    half_mesh = np.maximum(half_mesh, float(min_extent_m) / 2.0)

    corner_signs = (
        (sx, sy, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    )
    corners_mesh = np.array(
        [
            center_mesh + np.array([sx * half_mesh[0], sy * half_mesh[1], sz * half_mesh[2]])
            for sx, sy, sz in corner_signs
        ],
        dtype=np.float64,
    )
    corners_base = _transform_points(output.T_base_mesh, corners_mesh)
    base_min = corners_base.min(axis=0)
    base_max = corners_base.max(axis=0)
    center_base = 0.5 * (base_min + base_max)
    extents_base = base_max - base_min

    return {
        "name": name,
        "position": center_base,
        "full_extents": extents_base,
        "aabb_source": aabb_source,
        "mesh_aabb_min_m": aabb_min,
        "mesh_aabb_max_m": aabb_max,
    }


def format_object_collision_box(box: dict) -> str:
    """Human-readable summary for logs."""
    pos = np.asarray(box["position"], dtype=np.float64)
    ext = np.asarray(box["full_extents"], dtype=np.float64)
    lo = pos - 0.5 * ext
    hi = pos + 0.5 * ext
    src = box.get("aabb_source", "unknown")
    return (
        f"center={np.round(pos, 4).tolist()} extents={np.round(ext, 4).tolist()} "
        f"ranges xyz=[{np.round(lo, 3).tolist()}, {np.round(hi, 3).tolist()}] "
        f"(aabb_source={src})"
    )
