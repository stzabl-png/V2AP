#!/usr/bin/env python3
"""
graspnet_demo/mesh_infer.py
============================
OFFLINE: Mesh → GraspNet → G_object.json (mesh/object-local frame)

This is step 1 of the official GraspNet real-robot pipeline:
  Mesh sample 20k pts + virtual table → high-quality grasp candidates
  stored in object-local (mesh) coordinate frame.

These candidates (G_object) are fixed for a given object. At execution
time, compose_grasp.py transforms them using the live T_cam_object from
ArUco detection.

Usage:
    # Generate for Pringles can cylinder mesh
    python -m graspnet_demo.mesh_infer \\
        --mesh graspnet_demo/objects/chips_can/mesh.obj \\
        --object-id chips_can \\
        --n-top 20

    # Custom mesh + scale
    python -m graspnet_demo.mesh_infer \\
        --mesh /path/to/mesh.obj \\
        --scale 0.85 \\
        --object-id my_object \\
        --output /path/to/G_object.json
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

_CKPT_DEFAULT = _PROJ / "Baseline2" / "graspnet" / "checkpoints" / "checkpoint-rs.tar"


def run_mesh_infer(
    mesh_path: str | Path,
    *,
    object_id: str,
    scale: float = 1.0,
    n_top: int = 20,
    n_points: int = 20000,
    device: str = "cuda",
    checkpoint: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Run GraspNet on object mesh → save G_object.json.

    Returns path to G_object.json.
    """
    from Baseline2.graspnet.graspnet_infer import load_model, mesh_to_pointcloud, infer_grasps

    t0 = time.time()
    mesh_path  = Path(mesh_path)
    ckpt       = Path(checkpoint or _CKPT_DEFAULT)

    print(f"\n{'='*60}")
    print(f"  GraspNet Mesh Inference (OFFLINE)")
    print(f"{'='*60}")
    print(f"  Mesh:      {mesh_path}")
    print(f"  Object ID: {object_id}")
    print(f"  Scale:     {scale}")

    # ── 1. Sample surface points from mesh ────────────────────
    net    = load_model(str(ckpt), device=device)
    points, mesh = mesh_to_pointcloud(str(mesh_path), scale_factor=scale, n_points=n_points)

    # ── 2. Run GraspNet (includes virtual table + collision) ───
    # graspnet_infer.infer_grasps() already adds generate_table_plane() internally
    gg = infer_grasps(net, points, n_top=n_top)

    if len(gg) == 0:
        print("  ❌ No grasps produced")
        return None

    # ── 3. Export G_object.json (mesh/object-local frame) ─────
    candidates = []
    for i, g in enumerate(gg):
        candidates.append({
            "rank": i,
            "score": round(float(g.score), 5),
            # grasp_point in OBJECT LOCAL (mesh) frame — Z-up, base at z=0
            "grasp_point": [round(float(x), 6) for x in g.translation],
            # rotation 3x3: col0=finger_open, col1=y_body, col2=approach
            "rotation": [[round(float(x), 6) for x in row]
                         for row in g.rotation_matrix],
            "gripper_width_m": round(float(g.width), 5),
            "approach_type": "graspnet_mesh",
            "position_panda_hand": None,  # not used for Dexmate
        })

    # Mesh extents for reference
    extents = (points.max(axis=0) - points.min(axis=0)).tolist()

    g_object = {
        "schema_version": "1.1",
        "object_id": object_id,
        # IMPORTANT: grasp_point/rotation are in object-local (mesh) frame
        # compose_grasp.py will transform using T_base_cam @ T_cam_object
        "frame": "object_local",
        "mesh_file": str(mesh_path.resolve()),
        "scale_factor": scale,
        "mesh_extents_m": [round(x, 4) for x in extents],
        "n_candidates": len(candidates),
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
            "z_axis": "up (object local: z=0 at base, z=height at top)",
        },
        "inference": {
            "n_raw_grasps": len(gg),
            "score_min": round(float(min(c["score"] for c in candidates)), 4),
            "score_max": round(float(max(c["score"] for c in candidates)), 4),
            "score_mean": round(float(np.mean([c["score"] for c in candidates])), 4),
            "checkpoint": str(ckpt.name),
        },
        "candidates": candidates,
    }

    # Determine output path
    if output_path is None:
        out_dir = mesh_path.parent
        output_path = out_dir / "G_object.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(g_object, f, indent=2, ensure_ascii=False)
    tmp.rename(output_path)

    elapsed = time.time() - t0
    print(f"\n  ✅ {len(candidates)} candidates → {output_path}")
    print(f"  Score: [{g_object['inference']['score_min']:.3f},"
          f"{g_object['inference']['score_max']:.3f}]"
          f" mean={g_object['inference']['score_mean']:.3f}")
    print(f"  ⏱️  {elapsed:.1f}s")

    # Print top 5
    print(f"\n  Top candidates (mesh/object-local frame):")
    for c in candidates[:5]:
        gp = c["grasp_point"]
        R  = np.array(c["rotation"])
        app = R[:, 2]
        print(f"    rank={c['rank']} score={c['score']:.3f} "
              f"pt=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}] "
              f"approach=[{app[0]:.2f},{app[1]:.2f},{app[2]:.2f}] "
              f"w={c['gripper_width_m']*100:.1f}cm")

    print(f"{'='*60}\n")
    return output_path


def main():
    p = argparse.ArgumentParser(description="Offline GraspNet mesh inference → G_object.json")
    p.add_argument("--mesh", required=True, help="Path to object mesh (.obj/.ply)")
    p.add_argument("--object-id", default=None, help="Object identifier (default: mesh stem)")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor (meters)")
    p.add_argument("--n-top", type=int, default=20, help="Top-N grasps to keep")
    p.add_argument("--n-points", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None, help="Output G_object.json path")
    args = p.parse_args()

    mesh_path = Path(args.mesh)
    object_id = args.object_id or mesh_path.parent.name

    run_mesh_infer(
        mesh_path,
        object_id=object_id,
        scale=args.scale,
        n_top=args.n_top,
        n_points=args.n_points,
        device=args.device,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
