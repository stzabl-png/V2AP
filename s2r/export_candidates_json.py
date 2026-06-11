#!/usr/bin/env python3
"""
Export A2G-format HDF5 → UCB candidates.json (schema v1.1)
===========================================================
Converts grasp candidates stored in project-native HDF5 files (written by
inference/grasp_pose.py or Baseline2/graspnet/graspnet_to_hdf5.py) into the
portable JSON format expected by the Razor-side run_auto_grasp.py.

The candidates.json schema is defined in:
    https://github.com/stzabl-png/UCB_Project/tree/titan/demo

Usage (standalone):
    python -m s2r.export_candidates_json \
        --hdf5 output/inference/affordance_grasp.hdf5 \
        --T-base-mesh output/register/T_base_mesh.json \
        --T-cam-mesh  output/register/T_cam_mesh.json \
        --mesh        output/mesh/object_scaled.obj \
        --output      output/inference/candidates.json

Typical usage (called from process_razor_session.py):
    from s2r.export_candidates_json import hdf5_to_candidates_json
    hdf5_to_candidates_json(hdf5_path, T_base_mesh, T_cam_mesh, mesh_path, out)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# ──────────────────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────────────────

def _ndarray_to_list(arr) -> list:
    """Recursively convert numpy arrays to nested python lists."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _read_candidates_from_hdf5(hdf5_path: str | Path) -> tuple[list[dict], dict]:
    """Read candidates from any A2G-compatible HDF5 file.

    Supports layouts written by:
    - inference/grasp_pose.py  (candidates/ group with candidate_N subgroups)
    - graspnet_to_hdf5.py      (same layout, source="graspnet_baseline")
    """
    candidates = []
    source_meta = {}

    with h5py.File(str(hdf5_path), "r") as f:
        # Read metadata
        if "metadata" in f:
            mg = f["metadata"]
            source_meta["obj_id"] = mg.attrs.get("obj_id", "unknown")
            source_meta["source"] = mg.attrs.get("source", "a2g_pdm")
            if isinstance(source_meta["obj_id"], bytes):
                source_meta["obj_id"] = source_meta["obj_id"].decode()
            if isinstance(source_meta["source"], bytes):
                source_meta["source"] = source_meta["source"].decode()

        # Read candidates
        if "candidates" in f:
            cg = f["candidates"]
            keys = sorted(
                [k for k in cg.keys() if k.startswith("candidate_")],
                key=lambda k: int(k.split("_")[-1])
                if k.split("_")[-1].isdigit()
                else 0,
            )
            for key in keys:
                ci = cg[key]
                c = {
                    "position": np.array(ci["position"][:], dtype=np.float64),
                    "rotation": np.array(ci["rotation"][:], dtype=np.float64),
                    "gripper_width": float(ci.attrs.get("gripper_width", 0.06)),
                    "score": float(ci.attrs.get("score", 0.0)),
                    "name": ci.attrs.get("name", key),
                    "approach_type": ci.attrs.get("approach_type", ""),
                }
                # Decode bytes if needed
                for str_key in ("name", "approach_type"):
                    if isinstance(c[str_key], bytes):
                        c[str_key] = c[str_key].decode()
                # Grasp point: if present use it, else position IS the grasp point
                if "grasp_point" in ci:
                    c["grasp_point"] = np.array(ci["grasp_point"][:], dtype=np.float64)
                else:
                    c["grasp_point"] = c["position"].copy()
                candidates.append(c)

        elif "grasp" in f:
            # Single-candidate legacy format
            gi = f["grasp"]
            c = {
                "position": np.array(gi["position"][:], dtype=np.float64),
                "rotation": np.array(gi["rotation"][:], dtype=np.float64),
                "gripper_width": float(gi.attrs.get("gripper_width", 0.06)),
                "score": float(gi.attrs.get("score", 0.0)),
                "name": gi.attrs.get("candidate_name", "best"),
                "approach_type": gi.attrs.get("approach_type", ""),
            }
            for str_key in ("name", "approach_type"):
                if isinstance(c[str_key], bytes):
                    c[str_key] = c[str_key].decode()
            if "grasp_point" in gi:
                c["grasp_point"] = np.array(gi["grasp_point"][:], dtype=np.float64)
            else:
                c["grasp_point"] = c["position"].copy()
            candidates.append(c)

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, source_meta


def _compute_mesh_span(mesh_path: str | Path) -> list[float]:
    """Compute mesh bounding box span [sx, sy, sz] in meters."""
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), force="mesh")
        verts = np.array(mesh.vertices)
        span = (verts.max(axis=0) - verts.min(axis=0)).tolist()
        return [round(s, 4) for s in span]
    except Exception:
        return [0.0, 0.0, 0.0]


def hdf5_to_candidates_json(
    hdf5_path: str | Path,
    T_base_mesh: np.ndarray,
    T_cam_mesh: np.ndarray,
    mesh_path: str | Path,
    output_path: str | Path,
    *,
    camera_frame: str = "zed_left_camera",
    T_base_cam_source: str = "input/calib/extrinsics.json",
    max_candidates: int = 10,
) -> dict:
    """Convert A2G HDF5 → UCB candidates.json (schema v1.1).

    Args:
        hdf5_path: Path to affordance_grasp.hdf5 (A2G format).
        T_base_mesh: 4×4 numpy array, base ← mesh transform.
        T_cam_mesh:  4×4 numpy array, camera ← mesh transform.
        mesh_path:   Path to object_scaled.obj (for span computation).
        output_path: Where to write candidates.json.
        camera_frame: Camera frame name.
        T_base_cam_source: Where T_base_cam came from (for provenance).
        max_candidates: Maximum number of candidates to include.

    Returns:
        The candidates dict (same as what's written to JSON).
    """
    candidates, source_meta = _read_candidates_from_hdf5(hdf5_path)
    candidates = candidates[:max_candidates]

    mesh_span = _compute_mesh_span(mesh_path)
    source = source_meta.get("source", "a2g_pdm")

    # Build output dict following UCB schema v1.1
    output = {
        "schema_version": "1.1",
        "mesh_frame": "mesh",
        "base_frame": "base",
        "camera_frame": camera_frame,
        "registration": {
            "method": "foundationpose",
            "T_cam_mesh": _ndarray_to_list(T_cam_mesh),
            "T_base_mesh": _ndarray_to_list(T_base_mesh),
            "T_base_cam_source": T_base_cam_source,
        },
        "T_base_mesh": _ndarray_to_list(T_base_mesh),
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame": "mesh",
            "ucb_tcp_offset_m": 0.105,
            "ucb_tcp_frame": "panda_hand",
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
        },
        "mesh_span_m": mesh_span,
        "n_candidates": len(candidates),
        "source": source,
        "candidates": [],
    }

    for rank, c in enumerate(candidates):
        entry = {
            "rank": rank,
            "name": c["name"],
            "score": round(float(c["score"]), 4),
            "grasp_point": _ndarray_to_list(c["grasp_point"]),
            "rotation": _ndarray_to_list(c["rotation"]),
            "gripper_width_m": round(float(c["gripper_width"]), 4),
            "approach_type": c["approach_type"],
            "source": source,
        }
        output["candidates"].append(entry)

    # Write atomically (tmp + rename)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.rename(out_path)

    print(f"  📦 candidates.json: {len(candidates)} candidates → {out_path}")
    return output


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def _load_transform_json(path: str) -> np.ndarray:
    """Load a 4×4 transform from a JSON file (T_cam_mesh.json or T_base_mesh.json)."""
    with open(path) as f:
        data = json.load(f)
    # Try known keys
    for key in ("T_cam_mesh", "T_base_mesh", "T_base_cam", "transform"):
        if key in data:
            return np.array(data[key], dtype=np.float64)
    # Fallback: first 4×4-looking value
    for v in data.values():
        if isinstance(v, list) and len(v) == 4:
            return np.array(v, dtype=np.float64)
    raise ValueError(f"Cannot find 4×4 transform in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert A2G HDF5 → UCB candidates.json"
    )
    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to affordance_grasp.hdf5")
    parser.add_argument("--T-base-mesh", type=str, required=True,
                        help="Path to T_base_mesh.json")
    parser.add_argument("--T-cam-mesh", type=str, required=True,
                        help="Path to T_cam_mesh.json")
    parser.add_argument("--mesh", type=str, required=True,
                        help="Path to object_scaled.obj")
    parser.add_argument("--output", type=str, required=True,
                        help="Output candidates.json path")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--camera-frame", type=str, default="zed_left_camera")
    args = parser.parse_args()

    T_base_mesh = _load_transform_json(args.T_base_mesh)
    T_cam_mesh = _load_transform_json(args.T_cam_mesh)

    hdf5_to_candidates_json(
        hdf5_path=args.hdf5,
        T_base_mesh=T_base_mesh,
        T_cam_mesh=T_cam_mesh,
        mesh_path=args.mesh,
        output_path=args.output,
        camera_frame=args.camera_frame,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
