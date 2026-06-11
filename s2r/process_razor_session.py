#!/usr/bin/env python3
"""
process_razor_session.py — Titan-side session processing pipeline
==================================================================
Complete T1→T7 pipeline that processes a Razor session's input/ package
and produces the output/ package ready for rsync back to Razor.

Supports two grasp policies via --policy:
  - a2g_pdm:           Main method (Affordance v6 + PDM grasp generation)
  - graspnet_baseline:  GraspNet-Baseline (direct mesh → grasp)

Pipeline stages:
  T1: Validate input package
  T2: SAM segmentation → mask.png
  T3: SAM3D mesh reconstruction → object_raw.glb
  T4: Metric scale calibration → object_scaled.obj
  T5: FoundationPose registration → T_cam_mesh, T_base_mesh
  T6: Grasp inference (policy-dependent) → HDF5 + candidates.json
  T7: Write status.json

Usage:
    # Full pipeline (GraspNet)
    python -m s2r.process_razor_session \
        --session-dir s2r/razor_sessions/20260601_143022_chips \
        --policy graspnet_baseline \
        --device cuda

    # Skip to T6 (mesh + registration already done)
    python -m s2r.process_razor_session \
        --session-dir s2r/razor_sessions/20260601_143022_chips \
        --policy graspnet_baseline \
        --skip-sam --skip-sam3d --skip-fp

    # T6-only standalone test with an existing mesh
    python -m s2r.process_razor_session \
        --session-dir /path/to/session \
        --policy graspnet_baseline \
        --t6-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# ──────────────────────────────────────────────────────────
# T1: Input validation
# ──────────────────────────────────────────────────────────

def validate_input(session_dir: Path) -> dict:
    """T1: Validate session input package.

    Returns:
        Dict with validation results and parsed session metadata.
    """
    input_dir = session_dir / "input"
    errors = []
    warnings = []

    # Required files
    required_files = {
        "session.json": input_dir / "session.json",
        "left_rgb.png": input_dir / "rgb" / "left_rgb.png",
        "depth.npy": input_dir / "depth" / "depth.npy",
        "intrinsics.json": input_dir / "calib" / "intrinsics.json",
        "extrinsics.json": input_dir / "calib" / "extrinsics.json",
    }

    for name, path in required_files.items():
        if not path.exists():
            errors.append(f"Missing required file: {name} ({path})")

    # Parse session.json if present
    session_meta = {}
    session_json_path = required_files["session.json"]
    if session_json_path.exists():
        try:
            with open(session_json_path) as f:
                session_meta = json.load(f)
            sv = session_meta.get("schema_version", "0.0")
            if not sv.startswith("1."):
                errors.append(f"Unsupported schema_version: {sv} (need 1.x)")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid session.json: {e}")

    # Optional: check K.npy
    k_npy = input_dir / "calib" / "K.npy"
    if not k_npy.exists():
        warnings.append("K.npy not found (will derive from intrinsics.json)")

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "session_meta": session_meta,
    }

    if result["valid"]:
        print("  ✅ T1: Input validation passed")
    else:
        print(f"  ❌ T1: Validation failed with {len(errors)} error(s)")
        for e in errors:
            print(f"       {e}")
    for w in warnings:
        print(f"     ⚠️  {w}")

    return result


# ──────────────────────────────────────────────────────────
# T6: Grasp inference (policy-dependent)
# ──────────────────────────────────────────────────────────

def run_t6_graspnet(session_dir: Path, device: str = "cuda", **kwargs) -> dict:
    """T6 for graspnet_baseline: run GraspNet on scaled mesh."""
    from s2r.graspnet_for_session import run_graspnet_on_session

    candidates_json = run_graspnet_on_session(
        session_dir=session_dir,
        device=device,
        n_top=kwargs.get("n_top", 50),
    )

    return {
        "step": "grasp_pose",
        "status": "ok",
        "candidates_json": str(candidates_json),
        "policy": "graspnet_baseline",
    }


def run_t6_a2g_pdm(session_dir: Path, device: str = "cuda", **kwargs) -> dict:
    """T6 for a2g_pdm: run Affordance v6 + PDM grasp generation.

    Note: This is a placeholder — the actual implementation would call
    inference/grasp_pose.py on the scaled mesh, then export candidates.json.
    """
    from s2r.export_candidates_json import hdf5_to_candidates_json

    output_dir = session_dir / "output" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find mesh
    mesh_path = None
    for name in ("object_scaled.obj", "object_scaled.glb", "object_scaled.ply"):
        p = session_dir / "output" / "mesh" / name
        if p.exists():
            mesh_path = p
            break

    if mesh_path is None:
        return {"step": "grasp_pose", "status": "error",
                "error": "No scaled mesh found"}

    # Run inference/grasp_pose.py
    print(f"  [T6] Running A2G PDM on {mesh_path}...")
    from inference.grasp_pose import main as grasp_pose_main

    hdf5_path = output_dir / "affordance_grasp.hdf5"

    # Use subprocess to avoid import conflicts
    import subprocess
    cmd = [
        sys.executable, "-m", "inference.grasp_pose",
        "--mesh", str(mesh_path),
        "--output", str(hdf5_path),
        "--device", device,
    ]
    ckpt = kwargs.get("ckpt")
    if ckpt:
        cmd += ["--ckpt", str(ckpt)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "step": "grasp_pose", "status": "error",
            "error": result.stderr[-500:] if result.stderr else "Unknown error",
        }

    # Export candidates.json
    T_base_mesh = np.eye(4)
    T_cam_mesh = np.eye(4)
    tbm_path = session_dir / "output" / "register" / "T_base_mesh.json"
    tcm_path = session_dir / "output" / "register" / "T_cam_mesh.json"

    if tbm_path.exists():
        with open(tbm_path) as f:
            T_base_mesh = np.array(json.load(f)["T_base_mesh"], dtype=np.float64)
    if tcm_path.exists():
        with open(tcm_path) as f:
            T_cam_mesh = np.array(json.load(f)["T_cam_mesh"], dtype=np.float64)

    candidates_json_path = output_dir / "candidates.json"
    hdf5_to_candidates_json(
        hdf5_path=hdf5_path,
        T_base_mesh=T_base_mesh,
        T_cam_mesh=T_cam_mesh,
        mesh_path=mesh_path,
        output_path=candidates_json_path,
    )

    return {
        "step": "grasp_pose",
        "status": "ok",
        "candidates_json": str(candidates_json_path),
        "policy": "a2g_pdm",
    }


# ──────────────────────────────────────────────────────────
# T7: Write status.json
# ──────────────────────────────────────────────────────────

def write_status(
    session_dir: Path,
    steps: dict[str, str],
    warnings: list[str],
    errors: list[str],
    policy: str,
):
    """T7: Write output/status.json (atomic write)."""
    output_dir = session_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = session_dir.name

    status = {
        "schema_version": "1.1",
        "session_id": session_id,
        "success": len(errors) == 0,
        "pipeline_version": "s2r.process_razor_session 0.1.0",
        "policy": policy,
        "finished_at_iso": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "warnings": warnings,
        "errors": errors,
    }

    status_path = output_dir / "status.json"
    tmp_path = status_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    tmp_path.rename(status_path)

    tag = "✅" if status["success"] else "❌"
    print(f"  {tag} T7: status.json written → {status_path}")
    return status


# ──────────────────────────────────────────────────────────
# Main pipeline orchestration
# ──────────────────────────────────────────────────────────

def process_session(
    session_dir: str | Path,
    *,
    policy: str = "graspnet_baseline",
    device: str = "cuda",
    skip_sam: bool = False,
    skip_sam3d: bool = False,
    skip_fp: bool = False,
    t6_only: bool = False,
    **kwargs,
) -> dict:
    """Run the full T1→T7 pipeline on a session.

    Args:
        session_dir: Path to session directory.
        policy: "graspnet_baseline" or "a2g_pdm".
        device: CUDA device.
        skip_sam: Skip T2 (mask already exists).
        skip_sam3d: Skip T3 (mesh already exists).
        skip_fp: Skip T5 (registration already exists).
        t6_only: Only run T6 + T7 (all prior steps done).

    Returns:
        Status dict (same as status.json).
    """
    session_dir = Path(session_dir)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Razor Session Processing Pipeline")
    print(f"{'='*60}")
    print(f"  Session:  {session_dir}")
    print(f"  Policy:   {policy}")
    print(f"  Device:   {device}")
    print(f"{'='*60}\n")

    steps = {}
    warnings = []
    errors = []

    # ── T1: Validate ────────────────────────────────────
    if not t6_only:
        t1 = validate_input(session_dir)
        if not t1["valid"]:
            errors.extend(t1["errors"])
            return write_status(session_dir, {"validate": "error"}, warnings, errors, policy)
        steps["validate"] = "ok"
        warnings.extend(t1.get("warnings", []))

    # ── T2-T5: Perception pipeline ──────────────────────
    if not t6_only:
        # T2: SAM segmentation
        if skip_sam:
            mask_path = session_dir / "output" / "segment" / "mask.png"
            if mask_path.exists():
                steps["segment"] = "ok (skipped)"
                print(f"  ⏭️  T2: SAM skipped (mask exists)")
            else:
                errors.append("T2 skipped but mask.png not found")
        else:
            # TODO: Implement SAM segmentation
            print(f"  🔲 T2: SAM segmentation (not yet implemented)")
            steps["segment"] = "not_implemented"
            warnings.append("T2 SAM not implemented — provide mask.png manually")

        # T3: SAM3D reconstruction
        if skip_sam3d:
            steps["sam3d"] = "ok (skipped)"
            print(f"  ⏭️  T3: SAM3D skipped")
        else:
            # TODO: Implement SAM3D
            print(f"  🔲 T3: SAM3D reconstruction (not yet implemented)")
            steps["sam3d"] = "not_implemented"
            warnings.append("T3 SAM3D not implemented — provide mesh manually")

        # T4: Scale calibration
        # TODO: Implement scale calibration
        print(f"  🔲 T4: Scale calibration (not yet implemented)")
        steps["scale"] = "not_implemented"
        warnings.append("T4 scale not implemented — mesh assumed pre-scaled")

        # T5: FoundationPose registration
        if skip_fp:
            steps["foundationpose"] = "ok (skipped)"
            print(f"  ⏭️  T5: FoundationPose skipped")
        else:
            # TODO: Implement FoundationPose
            print(f"  🔲 T5: FoundationPose (not yet implemented)")
            steps["foundationpose"] = "not_implemented"
            warnings.append("T5 FP not implemented — provide T_*_mesh.json manually")

    # ── T6: Grasp inference ─────────────────────────────
    print(f"\n  [T6] Grasp inference (policy={policy})...")
    try:
        if policy == "graspnet_baseline":
            t6_result = run_t6_graspnet(session_dir, device=device, **kwargs)
        elif policy == "a2g_pdm":
            t6_result = run_t6_a2g_pdm(session_dir, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if t6_result["status"] == "ok":
            steps["grasp_pose"] = "ok"
        else:
            steps["grasp_pose"] = "error"
            errors.append(t6_result.get("error", "T6 failed"))
    except Exception as e:
        steps["grasp_pose"] = "error"
        errors.append(f"T6 exception: {e}")
        traceback.print_exc()

    # ── T7: Status ──────────────────────────────────────
    elapsed = time.time() - t0
    status = write_status(session_dir, steps, warnings, errors, policy)

    print(f"\n  ⏱️  Pipeline total: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    return status


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process a Razor session on Titan (T1→T7)"
    )
    parser.add_argument("--session-dir", type=str, required=True,
                        help="Path to session directory")
    parser.add_argument("--policy", type=str, default="graspnet_baseline",
                        choices=["graspnet_baseline", "a2g_pdm"],
                        help="Grasp generation policy")
    parser.add_argument("--device", type=str, default="cuda")

    # Skip flags
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip T2 SAM segmentation")
    parser.add_argument("--skip-sam3d", action="store_true",
                        help="Skip T3 SAM3D reconstruction")
    parser.add_argument("--skip-fp", action="store_true",
                        help="Skip T5 FoundationPose")
    parser.add_argument("--t6-only", action="store_true",
                        help="Only run T6 grasp inference + T7 status")

    # GraspNet-specific
    parser.add_argument("--n-top", type=int, default=50,
                        help="GraspNet: top-K grasps to keep")

    args = parser.parse_args()

    process_session(
        session_dir=args.session_dir,
        policy=args.policy,
        device=args.device,
        skip_sam=args.skip_sam,
        skip_sam3d=args.skip_sam3d,
        skip_fp=args.skip_fp,
        t6_only=args.t6_only,
        n_top=args.n_top,
    )


if __name__ == "__main__":
    main()
