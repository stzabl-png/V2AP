#!/usr/bin/env python3
"""
GraspNet-Baseline inference for a Razor session
=================================================
Runs GraspNet on the scaled mesh from a session's output/mesh/ directory,
producing the same HDF5 + candidates.json that the A2G PDM path produces.

This is the GraspNet analog of demo/scripts/T6/run_pdm_grasp.py.

Usage:
    python -m s2r.graspnet_for_session \
        --session-dir s2r/razor_sessions/20260601_143022_chips \
        --device cuda

    python -m s2r.graspnet_for_session \
        --mesh output/mesh/object_scaled.obj \
        --T-base-mesh output/register/T_base_mesh.json \
        --T-cam-mesh  output/register/T_cam_mesh.json \
        --output-dir  output/inference/ \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _load_transform_json(path: str | Path) -> np.ndarray:
    """Load a 4×4 transform from T_cam_mesh.json / T_base_mesh.json."""
    with open(str(path)) as f:
        data = json.load(f)
    for key in ("T_cam_mesh", "T_base_mesh", "T_base_cam", "transform"):
        if key in data:
            return np.array(data[key], dtype=np.float64)
    for v in data.values():
        if isinstance(v, list) and len(v) == 4:
            return np.array(v, dtype=np.float64)
    raise ValueError(f"Cannot find 4×4 transform in {path}")


def _find_mesh_in_session(session_dir: Path) -> Optional[Path]:
    """Find the scaled mesh in a session's output directory."""
    for name in ("object_scaled.obj", "object_scaled.glb", "object_scaled.ply"):
        p = session_dir / "output" / "mesh" / name
        if p.exists():
            return p
    # Fallback: look in the root of session
    for name in ("object_scaled.obj", "mesh.obj", "mesh.ply"):
        p = session_dir / name
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def run_graspnet_on_session(
    session_dir: Optional[str | Path] = None,
    *,
    mesh_path: Optional[str | Path] = None,
    T_base_mesh_path: Optional[str | Path] = None,
    T_cam_mesh_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    device: str = "cuda",
    n_top: int = 50,
    n_points: int = 20000,
    max_candidates_json: int = 10,
) -> Path:
    """Run GraspNet inference and export candidates.json for a session.

    Can be called in two modes:
    1. Session mode: provide session_dir, auto-discover mesh + transforms
    2. Explicit mode: provide mesh_path, T_base_mesh_path, T_cam_mesh_path, output_dir

    Returns:
        Path to the generated candidates.json
    """
    t0 = time.time()

    # ── Resolve paths ──────────────────────────────────────
    if session_dir is not None:
        session_dir = Path(session_dir)
        if mesh_path is None:
            mesh_path = _find_mesh_in_session(session_dir)
            if mesh_path is None:
                raise FileNotFoundError(
                    f"No scaled mesh found in {session_dir}/output/mesh/"
                )
        if T_base_mesh_path is None:
            T_base_mesh_path = session_dir / "output" / "register" / "T_base_mesh.json"
        if T_cam_mesh_path is None:
            T_cam_mesh_path = session_dir / "output" / "register" / "T_cam_mesh.json"
        if output_dir is None:
            output_dir = session_dir / "output" / "inference"
    else:
        if mesh_path is None:
            raise ValueError("Either session_dir or mesh_path must be provided")
        if output_dir is None:
            output_dir = Path(mesh_path).parent.parent / "inference"

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GraspNet-Baseline → Session Inference")
    print(f"{'='*60}")
    print(f"  Mesh:       {mesh_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Device:     {device}")

    # ── Load GraspNet model ────────────────────────────────
    from Baseline2.graspnet.graspnet_infer import (
        load_model,
        mesh_to_pointcloud,
        infer_grasps,
    )
    from Baseline2.graspnet.graspnet_to_hdf5 import (
        graspgroup_to_candidates,
        rerank_for_reachability,
        write_candidate_hdf5,
    )

    if checkpoint is None:
        checkpoint = PROJ / "Baseline2" / "graspnet" / "checkpoints" / "checkpoint-rs.tar"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"GraspNet checkpoint not found: {checkpoint}")

    print(f"  Checkpoint: {checkpoint}")
    net = load_model(str(checkpoint), device=device)

    # ── Generate point cloud from mesh ─────────────────────
    # No scale_factor needed: object_scaled mesh is already in meters
    points, mesh = mesh_to_pointcloud(str(mesh_path), scale_factor=1.0, n_points=n_points)

    # ── Run GraspNet inference ─────────────────────────────
    gg = infer_grasps(net, points, n_top=n_top)

    if len(gg) == 0:
        print("  ⚠️  GraspNet produced 0 grasps")
        # Write empty candidates
        hdf5_path = output_dir / "affordance_grasp.hdf5"
        write_candidate_hdf5(
            candidates=[],
            obj_id="session",
            output_path=str(hdf5_path),
        )
    else:
        # ── Convert to A2G format + rerank ─────────────────
        candidates = graspgroup_to_candidates(gg, scale_factor=1.0)
        candidates = rerank_for_reachability(candidates)
        print(f"  ✅ {len(candidates)} valid candidates after conversion + rerank")

        # ── Write HDF5 ────────────────────────────────────
        hdf5_path = output_dir / "affordance_grasp.hdf5"
        write_candidate_hdf5(
            candidates=candidates,
            obj_id="session",
            output_path=str(hdf5_path),
        )

    # ── Export candidates.json ─────────────────────────────
    # Load registration transforms (may not exist for standalone tests)
    T_base_mesh = np.eye(4)
    T_cam_mesh = np.eye(4)

    if T_base_mesh_path is not None and Path(T_base_mesh_path).exists():
        T_base_mesh = _load_transform_json(T_base_mesh_path)
        print(f"  📐 T_base_mesh loaded from {T_base_mesh_path}")
    else:
        print(f"  ⚠️  T_base_mesh not found, using identity")

    if T_cam_mesh_path is not None and Path(T_cam_mesh_path).exists():
        T_cam_mesh = _load_transform_json(T_cam_mesh_path)
        print(f"  📐 T_cam_mesh loaded from {T_cam_mesh_path}")
    else:
        print(f"  ⚠️  T_cam_mesh not found, using identity")

    from s2r.export_candidates_json import hdf5_to_candidates_json

    candidates_json_path = output_dir / "candidates.json"
    hdf5_to_candidates_json(
        hdf5_path=hdf5_path,
        T_base_mesh=T_base_mesh,
        T_cam_mesh=T_cam_mesh,
        mesh_path=mesh_path,
        output_path=candidates_json_path,
        max_candidates=max_candidates_json,
    )

    elapsed = time.time() - t0
    print(f"\n  ⏱️  Total GraspNet session time: {elapsed:.1f}s")
    print(f"  📁 HDF5:           {hdf5_path}")
    print(f"  📁 candidates.json: {candidates_json_path}")
    print(f"{'='*60}\n")

    return candidates_json_path


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run GraspNet-Baseline on a Razor session's mesh"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir", type=str,
                       help="Session directory (auto-discover mesh + transforms)")
    group.add_argument("--mesh", type=str,
                       help="Direct path to object_scaled mesh")

    parser.add_argument("--T-base-mesh", type=str, default=None,
                        help="Path to T_base_mesh.json")
    parser.add_argument("--T-cam-mesh", type=str, default=None,
                        help="Path to T_cam_mesh.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for HDF5 + candidates.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="GraspNet checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-top", type=int, default=50,
                        help="Top-K grasps from GraspNet")
    parser.add_argument("--n-points", type=int, default=20000,
                        help="Number of surface sample points")
    parser.add_argument("--max-candidates-json", type=int, default=10,
                        help="Max candidates in candidates.json")
    args = parser.parse_args()

    run_graspnet_on_session(
        session_dir=args.session_dir,
        mesh_path=args.mesh,
        T_base_mesh_path=args.T_base_mesh,
        T_cam_mesh_path=args.T_cam_mesh,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        n_top=args.n_top,
        n_points=args.n_points,
        max_candidates_json=args.max_candidates_json,
    )


if __name__ == "__main__":
    main()
