#!/usr/bin/env python3
"""
GraspNet inference for real-robot testing sessions on USB.

For each session in /media/lyh/KINGSTON/testing data/:
  1. Load object_base_aligned.glb (metric, base frame)
  2. Sample surface → point cloud + virtual table
  3. GraspNet forward → collision detection → NMS → Top-5
  4. Convert to V2AP candidates.json (preserving T_base_mesh, conventions)
  5. Write back to USB

Usage:
    conda activate graspnet
    python scripts/graspnet_for_testing.py

    # Then visualize each one:
    python -m graspnet_demo.vis_candidates --session "/media/lyh/KINGSTON/testing data/<session_name>" --n-top 5
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from Baseline2.graspnet.graspnet_infer import (
    load_model,
    mesh_to_pointcloud,
    infer_grasps,
)

USB_TESTING = Path("/media/lyh/KINGSTON/testing data")
N_TOP_FINAL = 5      # Final top-K after sector filter
N_TOP_PREFILT = 50   # Pre-filter pool from GraspNet
DEPTH_SHIFT_M = 0.025  # 2.5cm shift along approach direction (deeper into object)

# ── Dexmate approach sector filter ────────────────────────────
# Same as Titan PDM pipeline: dexmate_right_approach_xy_sector
SECTOR_LO_DEG = 225.0
SECTOR_HI_DEG = 315.0
MIN_HORIZ_NORM = 0.25    # below this → treat as vertical
VERTICAL_Z_MIN = 0.85    # arrival Z must be ≥ this for top-down


def dexmate_approach_sector_filter(gg):
    """Filter GraspNet grasps by Dexmate reachable approach directions.

    Rules (matching Titan pdm_meta.json approach_sector):
      - arrival = -approach (the direction the arm comes from)
      - Side: arrival XY angle ∈ [225°, 315°] (robot's right side)
      - Top-down: ||arrival_xy|| < MIN_HORIZ_NORM AND arrival_z ≥ VERTICAL_Z_MIN
    """
    keep = []
    n_sector_reject = 0
    for i in range(len(gg)):
        # GraspNet: R[:,0] = approach
        approach = gg[i].rotation_matrix[:, 0]
        arrival = -approach  # direction arm comes from

        horiz_norm = np.sqrt(arrival[0]**2 + arrival[1]**2)

        if horiz_norm < MIN_HORIZ_NORM:
            # Near-vertical: allow if arrival is mostly +Z (top-down)
            if arrival[2] >= VERTICAL_Z_MIN:
                keep.append(i)
            else:
                n_sector_reject += 1
        else:
            # Side approach: check XY angle in sector
            angle_deg = np.degrees(np.arctan2(arrival[1], arrival[0])) % 360
            if SECTOR_LO_DEG <= angle_deg <= SECTOR_HI_DEG:
                keep.append(i)
            else:
                n_sector_reject += 1

    print(f"     approach sector [{SECTOR_LO_DEG}°–{SECTOR_HI_DEG}°]: "
          f"{len(keep)} kept, {n_sector_reject} rejected")
    return gg[keep] if keep else gg  # fallback: keep all if filter too strict


def graspnet_grasp_to_v2ap(g, rank: int) -> dict:
    """Convert one GraspNet GraspGroup item to V2AP candidates.json format.

    GraspNet convention:
        R[:, 0] = approach (wrist→fingertip)
        R[:, 1] = finger open/close
        R[:, 2] = binormal

    V2AP convention (same as A2G):
        R[:, 0] = finger open/close
        R[:, 1] = binormal
        R[:, 2] = approach
    """
    R_gn = g.rotation_matrix  # (3,3)
    approach = R_gn[:, 0].copy()
    finger_dir = R_gn[:, 1].copy()
    binormal = R_gn[:, 2].copy()

    # V2AP rotation: [finger, binormal, approach]
    R_v2ap = np.column_stack([finger_dir, binormal, approach])
    if np.linalg.det(R_v2ap) < 0:
        R_v2ap[:, 1] = -R_v2ap[:, 1]

    # GraspNet translation = TCP center (fingertip midpoint)
    # Apply 2.5cm depth shift along approach (deeper into object for reliable contact)
    grasp_point = (g.translation + approach * DEPTH_SHIFT_M).tolist()
    width = float(np.clip(g.width, 0.01, 0.08))

    return {
        "rank": rank,
        "name": f"graspnet_{rank:03d}",
        "score": float(g.score),
        "grasp_point": grasp_point,
        "rotation": R_v2ap.tolist(),
        "gripper_width_m": width,
        "approach_type": "graspnet_baseline",
        "cross_section_width_m": width,
        "position_panda_hand": grasp_point,  # same for identity T_base_mesh
    }


def process_session(session_dir: Path, net) -> bool:
    """Process one session: GLB → GraspNet → candidates.json."""
    name = session_dir.name
    mesh_path = session_dir / "output" / "mesh" / "object_base_aligned.glb"
    t_base_mesh_path = session_dir / "output" / "register" / "T_base_mesh.json"

    if not mesh_path.exists():
        print(f"  ❌ {name}: mesh not found")
        return False

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    # 1. Load mesh → point cloud (scale=1.0, already metric)
    points, mesh = mesh_to_pointcloud(str(mesh_path), scale_factor=1.0, n_points=20000)

    # 2. GraspNet inference (large pool, then sector filter)
    gg = infer_grasps(net, points, n_top=N_TOP_PREFILT, collision_thresh=0.01, z_approach_max=0.3)

    if len(gg) == 0:
        print(f"  ⚠️  No valid grasps for {name}")
        return False

    # 3. Dexmate approach sector filter
    gg = dexmate_approach_sector_filter(gg)
    gg = gg[:N_TOP_FINAL]  # Take top-5 after filtering

    if len(gg) == 0:
        print(f"  ⚠️  No grasps passed sector filter for {name}")
        return False

    # 4. Convert to V2AP format
    candidates = [graspnet_grasp_to_v2ap(gg[i], i) for i in range(len(gg))]

    # 4. Load T_base_mesh from original registration
    if t_base_mesh_path.exists():
        with open(t_base_mesh_path) as f:
            T_base_mesh = json.load(f)
        if isinstance(T_base_mesh, dict):
            T_base_mesh = T_base_mesh.get("T_base_mesh", np.eye(4).tolist())
    else:
        T_base_mesh = np.eye(4).tolist()

    # 5. Build candidates.json
    import trimesh
    scene = trimesh.load(str(mesh_path))
    mesh_tm = trimesh.util.concatenate([g for g in scene.geometry.values()])
    v = np.array(mesh_tm.vertices)
    aabb_min = v.min(0).tolist()
    aabb_max = v.max(0).tolist()
    span = (v.max(0) - v.min(0)).tolist()

    candidates_json = {
        "schema_version": "1.1",
        "mesh_frame": "base_aligned",
        "base_frame": "base",
        "camera_frame": "zed_left_camera",
        "inference_method": "graspnet_baseline",
        "T_base_mesh": T_base_mesh,
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame": "base_aligned",
            "ucb_tcp_offset_m": 0.105,
            "ucb_tcp_frame": "panda_hand",
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
        },
        "mesh_span_m": span,
        "mesh_aabb_min_m": aabb_min,
        "mesh_aabb_max_m": aabb_max,
        "mesh_file": "output/mesh/object_base_aligned.glb",
        "n_candidates": len(candidates),
        "candidates": candidates,
        "exported_at_iso": datetime.now(timezone.utc).isoformat(),
        "source_hdf5": None,
    }

    # 6. Write to USB
    out_path = session_dir / "output" / "inference" / "candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(candidates_json, f, indent=2)

    # 7. Update status.json
    status_path = session_dir / "output" / "status.json"
    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
    else:
        status = {}
    status["graspnet_baseline"] = {
        "success": True,
        "n_candidates": len(candidates),
        "method": "graspnet_baseline",
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "scores": [c["score"] for c in candidates],
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    scores = [c["score"] for c in candidates]
    print(f"  ✅ {len(candidates)} candidates → {out_path}")
    print(f"     scores: {[f'{s:.3f}' for s in scores]}")
    return True


def main():
    sessions = sorted([
        d for d in USB_TESTING.iterdir()
        if d.is_dir() and (d / "output" / "mesh" / "object_base_aligned.glb").exists()
    ])

    print(f"{'='*60}")
    print(f"  GraspNet Baseline for Real Robot Testing")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Top-K: {N_TOP_FINAL}")
    print(f"{'='*60}")

    net = load_model(str(PROJ / "Baseline2" / "graspnet" / "checkpoints" / "checkpoint-rs.tar"))

    ok = 0
    for sess in sessions:
        if process_session(sess, net):
            ok += 1

    print(f"\n{'='*60}")
    print(f"  Done! {ok}/{len(sessions)} sessions processed")
    print(f"{'='*60}")
    print(f"\n  To visualize each session:")
    print(f"  conda activate graspnet")
    for sess in sessions:
        print(f'  python -m graspnet_demo.vis_candidates --session "{sess}" --n-top {N_TOP_FINAL}')


if __name__ == "__main__":
    main()
