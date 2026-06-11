#!/usr/bin/env python3
"""
s2r/graspnet_scene_infer.py
============================
Scene-level GraspNet inference (official approach).

This script uses the FULL workspace point cloud (not just the object mask),
which gives the collision detector access to the table plane and surrounding
geometry. This is the recommended mode.

Pipeline:
  1. Backproject full depth → camera frame
  2. Transform to base frame using T_base_cam
  3. Workspace filter: Z range, XY crop
  4. Detect table height (RANSAC percentile on Z)
  5. Pad/downsample to 20,000 pts (GraspNet requirement)
  6. Run GraspNet baseline
  7. Collision filter (ModelFreeCollisionDetector)
  8. Filter candidates on object region (XY + height)
  9. Write V2AP-compatible candidates.json + status.json

Usage:
    python -m s2r.graspnet_scene_infer \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips

    # With explicit workspace and object region:
    python -m s2r.graspnet_scene_infer \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips \\
        --ws-x 0.2 1.2 --ws-y -0.4 0.4 \\
        --obj-x 0.35 0.75 --obj-y -0.15 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


# ─────────────────────────────────────────────────────────────
# Core scene-level inference
# ─────────────────────────────────────────────────────────────

def process_scene_session(
    session_dir: str | Path,
    *,
    device: str = "cuda",
    # Workspace filter (base frame, meters)
    ws_x: tuple[float, float] = (0.15, 1.30),
    ws_y: tuple[float, float] = (-0.45, 0.45),
    ws_z_above_table: float = 0.02,      # include pts ≥ table_height + this
    ws_z_max: float = 0.60,              # max height above table
    # Object region filter for candidate selection
    obj_x: tuple[float, float] | None = None,
    obj_y: tuple[float, float] | None = None,
    obj_z_min_above_table: float = 0.02,
    # GraspNet params
    n_top: int = 50,
    max_candidates_json: int = 10,
    target_points: int = 20000,
    # Collision detector
    collision_voxel: float = 0.01,
    # Table height
    table_height_m: float | None = None,
    table_percentile: float = 10.0,
) -> Path:
    """Scene-level GraspNet inference. Returns path to candidates.json."""

    t0 = time.time()
    session_dir = Path(session_dir)
    input_dir   = session_dir / "input"
    output_dir  = session_dir / "output" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GraspNet Scene-Level Inference")
    print(f"{'='*60}")
    print(f"  Session: {session_dir}")

    # ── 1. Load inputs ────────────────────────────────────
    depth = np.load(str(input_dir / "depth" / "depth.npy")).astype(np.float32)
    K     = np.load(str(input_dir / "calib" / "K.npy")).astype(np.float32)
    with open(input_dir / "calib" / "extrinsics.json") as f:
        ext = json.load(f)
    T_base_cam = np.array(ext["T_base_cam"], dtype=np.float32)

    if table_height_m is None:
        table_path = input_dir / "scene" / "table.json"
        if table_path.exists():
            with open(table_path) as f:
                table_height_m = float(json.load(f).get("table_height_m", 0.85))
        else:
            table_height_m = None  # will estimate from depth below

    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    print(f"  Image: {W}×{H}")

    # ── 2. Backproject + transform to base frame ──────────
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    valid = depth > 0.01
    z = depth[valid]
    pts_cam = np.column_stack([(u[valid]-cx)*z/fx,
                               (v[valid]-cy)*z/fy, z]).astype(np.float32)
    R_bc = T_base_cam[:3, :3].astype(np.float32)
    t_bc = T_base_cam[:3, 3].astype(np.float32)
    pts_base = (R_bc @ pts_cam.T).T + t_bc
    print(f"  Full cloud: {len(pts_base):,} pts")

    # ── 3. Estimate table height from depth ───────────────
    if table_height_m is None:
        # Use low-percentile of Z (table is lowest stable surface)
        table_height_m = float(np.percentile(pts_base[:, 2], table_percentile))
        print(f"  Table height (p{table_percentile:.0f}): {table_height_m:.3f}m (estimated)")
    else:
        print(f"  Table height: {table_height_m:.3f}m (from session)")

    # ── 4. Workspace filter ───────────────────────────────
    z_min = table_height_m + ws_z_above_table
    z_max = table_height_m + ws_z_max
    mask = (
        (pts_base[:, 0] >= ws_x[0]) & (pts_base[:, 0] <= ws_x[1]) &
        (pts_base[:, 1] >= ws_y[0]) & (pts_base[:, 1] <= ws_y[1]) &
        (pts_base[:, 2] >= z_min)   & (pts_base[:, 2] <= z_max)
    )
    scene_pts = pts_base[mask]
    print(f"  Workspace [{ws_x[0]:.2f},{ws_x[1]:.2f}]x[{ws_y[0]:.2f},{ws_y[1]:.2f}] "
          f"z=[{z_min:.2f},{z_max:.2f}]: {len(scene_pts):,} pts")

    if len(scene_pts) < 200:
        _write_status(session_dir, success=False, error="Too few points in workspace")
        return output_dir / "candidates.json"

    # ── 5. Center + resample to 20,000 pts ───────────────
    centroid = scene_pts.mean(axis=0)
    pts_c = scene_pts - centroid

    if len(pts_c) >= target_points:
        idx = np.random.choice(len(pts_c), target_points, replace=False)
        pts_input = pts_c[idx].astype(np.float32)
    else:
        n_need = target_points - len(pts_c)
        idx = np.random.choice(len(pts_c), n_need, replace=True)
        noise = (np.random.randn(n_need, 3) * 0.001).astype(np.float32)
        pts_input = np.concatenate([pts_c, pts_c[idx] + noise]).astype(np.float32)

    print(f"  GraspNet input: {len(pts_input):,} pts  centroid={centroid.round(3)}")

    # ── 6. GraspNet inference ─────────────────────────────
    sys.path.insert(0, str(_PROJ / "Baseline2" / "graspnet"))
    from Baseline2.graspnet.graspnet_infer import load_model, infer_grasps

    ckpt = str(_PROJ / "Baseline2" / "graspnet" / "checkpoints" / "checkpoint-rs.tar")
    print(f"  Loading checkpoint: {Path(ckpt).name}")
    net = load_model(ckpt, device=device)
    gg  = infer_grasps(net, pts_input, n_top=n_top)
    print(f"  Raw grasps: {len(gg)}")

    if len(gg) == 0:
        _write_empty_candidates(output_dir)
        _write_status(session_dir, success=True, warning="No grasps produced")
        return output_dir / "candidates.json"

    # ── 7. Collision filter ───────────────────────────────
    try:
        from graspnetAPI.utils.collision_detector import ModelFreeCollisionDetector
        mfcdetector = ModelFreeCollisionDetector(scene_pts, voxel_size=collision_voxel)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]
        print(f"  After collision filter: {len(gg)} grasps")
    except Exception as e:
        print(f"  ⚠️  Collision filter skipped: {e}")

    gg.sort_by_score()

    if len(gg) == 0:
        _write_empty_candidates(output_dir)
        _write_status(session_dir, success=True, warning="All grasps colliding")
        return output_dir / "candidates.json"

    # ── 8. Convert to base frame + object region filter ───
    all_candidates = []
    for g in gg:
        t_base = (g.translation + centroid).astype(np.float64)
        R_g    = g.rotation_matrix.astype(np.float64)
        all_candidates.append({
            "t_base": t_base,
            "R": R_g,
            "score": float(g.score),
            "width": float(g.width),
        })

    # Object region filter (optional)
    filtered = []
    if obj_x is not None or obj_y is not None:
        obj_x_range = obj_x or ws_x
        obj_y_range = obj_y or ws_y
        obj_z_min   = table_height_m + obj_z_min_above_table
        for c in all_candidates:
            t = c["t_base"]
            if (obj_x_range[0] <= t[0] <= obj_x_range[1] and
                obj_y_range[0] <= t[1] <= obj_y_range[1] and
                t[2] >= obj_z_min):
                filtered.append(c)
        print(f"  Object filter x{obj_x_range} y{obj_y_range} z≥{obj_z_min:.2f}: "
              f"{len(filtered)} candidates")
        if not filtered:
            print(f"  ⚠️  Object filter too strict, using all candidates")
            filtered = all_candidates
    else:
        filtered = all_candidates

    # Sort by score, take top N
    filtered.sort(key=lambda c: -c["score"])
    top = filtered[:max_candidates_json]

    candidates = []
    extents = scene_pts.max(axis=0) - scene_pts.min(axis=0)
    object_slug = session_dir.name.split("_")[-1]

    for rank, c in enumerate(top):
        t = c["t_base"]
        R = c["R"]
        candidates.append({
            "rank": rank,
            "name": f"graspnet_{rank}",
            "score": round(c["score"], 4),
            # grasp_point in base frame (T_base_mesh=I)
            "grasp_point": [round(float(x), 5) for x in t],
            # rotation 3x3: col2 = approach direction
            "rotation": [[round(float(x), 5) for x in row] for row in R],
            "gripper_width_m": round(c["width"], 4),
            "approach_type": "graspnet_scene",
            "position_panda_hand": None,
        })

    # ── 9. Write candidates.json (V2AP-compatible) ────────
    output_json = {
        "schema_version": "1.1",
        "mesh_frame": "base_aligned",
        "base_frame": "base",
        "camera_frame": ext.get("camera_frame", "zed_left_camera"),
        "registration": {
            "method": "graspnet_scene",
            "T_cam_mesh": np.eye(4).tolist(),
            "T_base_mesh": np.eye(4).tolist(),
            "T_base_cam": T_base_cam.tolist(),
            "T_base_cam_source": "input/calib/extrinsics.json",
            "object_centroid_base": centroid.tolist(),
            "notes": "Scene-level GraspNet; no FoundationPose; T_base_mesh=I",
        },
        "T_base_mesh": np.eye(4).tolist(),
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame": "base",
            "ucb_tcp_offset_m": 0.105,
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
        },
        "workspace": {
            "x": list(ws_x), "y": list(ws_y),
            "z": [float(z_min), float(z_max)],
        },
        "mesh_span_m": [round(float(x), 4) for x in extents],
        "table_height_m": round(float(table_height_m), 4),
        "n_candidates": len(candidates),
        "titan": {
            "n_candidates": len(candidates),
            "object_slug": object_slug,
            "policy": "graspnet_scene",
            "table_height_m": float(table_height_m),
        },
        "candidates": candidates,
    }

    out_path = output_dir / "candidates.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    tmp_path.rename(out_path)

    _write_status(session_dir, success=True, n_candidates=len(candidates))

    elapsed = time.time() - t0
    print(f"\n  ✅ {len(candidates)} candidates → {out_path}")
    print(f"  ⏱️  {elapsed:.1f}s")
    if candidates:
        best = candidates[0]
        gp = best["grasp_point"]
        print(f"  Best [rank=0]: score={best['score']:.3f} "
              f"pt=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}]")
    print(f"{'='*60}\n")

    return out_path


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _write_empty_candidates(output_dir: Path):
    out = {"schema_version": "1.1", "T_base_mesh": np.eye(4).tolist(),
           "n_candidates": 0, "candidates": []}
    with open(output_dir / "candidates.json", "w") as f:
        json.dump(out, f, indent=2)


def _write_status(session_dir: Path, *, success: bool,
                  error: str = "", warning: str = "", n_candidates: int = 0):
    session_dir = Path(session_dir)
    object_slug = session_dir.name.split("_")[-1]
    status = {
        "schema_version": "1.1",
        "session_id": session_dir.name,
        "success": success,
        "titan": {
            "object_slug": object_slug,
            "n_candidates": n_candidates,
            "policy": "graspnet_scene",
        },
        "steps": {
            "segment": "skip",
            "grasp_pose": "ok" if success else "error",
        },
        "warnings": [warning] if warning else [],
        "errors": [error] if error else [],
    }
    (session_dir / "output").mkdir(parents=True, exist_ok=True)
    tmp = session_dir / "output" / "status.json.tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.rename(session_dir / "output" / "status.json")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Scene-level GraspNet inference")
    p.add_argument("--session-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ws-x", type=float, nargs=2, default=[0.15, 1.30])
    p.add_argument("--ws-y", type=float, nargs=2, default=[-0.45, 0.45])
    p.add_argument("--obj-x", type=float, nargs=2, default=None)
    p.add_argument("--obj-y", type=float, nargs=2, default=None)
    p.add_argument("--table-height", type=float, default=None)
    p.add_argument("--n-top", type=int, default=50)
    p.add_argument("--max-candidates", type=int, default=10)
    args = p.parse_args()

    process_scene_session(
        session_dir=args.session_dir,
        device=args.device,
        ws_x=tuple(args.ws_x),
        ws_y=tuple(args.ws_y),
        obj_x=tuple(args.obj_x) if args.obj_x else None,
        obj_y=tuple(args.obj_y) if args.obj_y else None,
        table_height_m=args.table_height,
        n_top=args.n_top,
        max_candidates_json=args.max_candidates,
    )


if __name__ == "__main__":
    main()
