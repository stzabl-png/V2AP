#!/usr/bin/env python3
"""
GraspNet → A2G HDF5 Coordinate Converter
==========================================
Converts GraspNet GraspGroup output to project-compatible candidate HDF5 files
that can be read by `evaluation/policies/a2g_pdm.py:load_candidate_hdf5()`.

Coordinate system mapping:
  GraspNet (graspnetAPI):
    R[:, 0] = approach反方向 (gripper base → fingertip direction)
    R[:, 1] = 夹爪开合 (finger_dir)
    R[:, 2] = binormal
    center  = 夹爪底板中心 (in metric frame)

  A2G Project:
    R[:, 0] = finger_dir  (夹爪开合, x)
    R[:, 1] = up           (cross product, y)
    R[:, 2] = approach     (接近方向, z)
    position = 接触中点    (in mesh/unscaled frame)

Usage:
    # Standalone (after running graspnet_infer.py --dump-npy):
    python Baseline2/graspnet/graspnet_to_hdf5.py \
        --grasp-npy /tmp/test_grasps.npy \
        --obj-id A01001 \
        --scale-factor 1.0 \
        --output output/graspnet_candidates/A01001_grasp.hdf5

    # Typically called from batch_graspnet.py (no manual invocation needed)
"""

import os
import sys
import argparse
import json
import numpy as np
import h5py

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def graspnet_grasp_to_a2g(
    rotation_gn: np.ndarray,
    translation_gn: np.ndarray,
    width: float,
    depth: float,
    score: float,
    scale_factor: float = 1.0,
    z_approach_max: float = 0.3,
) -> dict | None:
    """Convert a single GraspNet grasp to A2G format.

    Args:
        rotation_gn: (3, 3) GraspNet rotation matrix (metric frame).
        translation_gn: (3,) GraspNet translation (metric frame).
        width: Gripper width (metric).
        depth: Grasp depth (metric).
        score: GraspNet score.
        scale_factor: Mesh scale factor (metric → mesh coords: divide by this).
        z_approach_max: Filter grasps approaching from below table
                        (approach_z > this → reject).

    Returns:
        Dict with position, rotation, gripper_width, approach, finger_dir, score
        in mesh coordinate frame, or None if filtered out.
    """
    # GraspNet column vectors (in rotated_mesh metric frame)
    # R_gn[:, 0] = approach (from gripper toward object, same as A2G convention)
    # Do NOT negate — A2G approach also points wrist→TCP (into object).
    approach = rotation_gn[:, 0].copy()
    finger_dir = rotation_gn[:, 1]  # gripper open/close axis
    binormal = rotation_gn[:, 2]    # third axis (cross product)

    # Filter: reject approach from below table (approach_z > 0 means upward)
    if approach[2] > z_approach_max:
        return None

    # A2G rotation: [finger_dir, binormal, approach]
    R_a2g = np.column_stack([finger_dir, binormal, approach])

    # Ensure right-handed coordinate system
    if np.linalg.det(R_a2g) < 0:
        R_a2g[:, 1] = -R_a2g[:, 1]

    # GraspNet's translation IS the TCP (finger center) in metric.
    # A2G convention: position = TCP/finger-center point (see pose_codec.py L5).
    # Keep in metric rotated_mesh frame — matches the USD object frame.
    position = translation_gn.astype(np.float32)

    gw = float(np.clip(width, 0.01, 0.08))

    return {
        "position": position,
        "rotation": R_a2g.astype(np.float32),
        "gripper_width": gw,
        "approach": approach.astype(np.float32),
        "finger_dir": finger_dir.astype(np.float32),
        "score": float(score),
    }


def graspgroup_to_candidates(
    gg,
    scale_factor: float = 1.0,
    z_approach_max: float = 0.3,
) -> list[dict]:
    """Convert GraspGroup (graspnetAPI) to list of A2G candidate dicts.

    Accepts either a GraspGroup object or a raw (N, 17) numpy array.
    """
    if hasattr(gg, "__len__") and len(gg) == 0:
        return []

    # Support raw numpy array (N, 17) from .npy dump
    if isinstance(gg, np.ndarray):
        candidates = []
        for row in gg:
            score = row[0]
            width = row[1]
            # height = row[2]  # unused (always 0.02)
            depth = row[3]
            rotation = row[4:13].reshape(3, 3)
            center = row[13:16]
            c = graspnet_grasp_to_a2g(
                rotation, center, width, depth, score,
                scale_factor=scale_factor,
                z_approach_max=z_approach_max,
            )
            if c is not None:
                candidates.append(c)
        return candidates

    # GraspGroup object
    candidates = []
    for i in range(len(gg)):
        g = gg[i]
        c = graspnet_grasp_to_a2g(
            g.rotation_matrix,
            g.translation,
            g.width,
            g.depth,
            g.score,
            scale_factor=scale_factor,
            z_approach_max=z_approach_max,
        )
        if c is not None:
            candidates.append(c)
    return candidates


def rerank_for_reachability(candidates: list[dict]) -> list[dict]:
    """Re-rank candidates to prefer top-down approaches reachable by Franka.

    GraspNet score only measures grasp quality; it doesn't know the robot's
    workspace.  Franka reaches objects best from above (approach_z < 0).
    Side approaches push the wrist to the arm's reach limit → IK failure.

    Composite score = graspnet_score × approach_bonus:
        approach_z < -0.5  →  bonus = 2.0  (strong top-down, ideal)
        approach_z < -0.2  →  bonus = 1.5  (angled from above, good)
        approach_z < 0.0   →  bonus = 1.0  (slightly sideways, neutral)
        approach_z >= 0.0  →  bonus = 0.3  (upward / below-table, bad)
    """
    if not candidates:
        return candidates
    for c in candidates:
        az = float(c["approach"][2])
        if az < -0.5:
            bonus = 2.0
        elif az < -0.2:
            bonus = 1.5
        elif az < 0.0:
            bonus = 1.0
        else:
            bonus = 0.3
        c["score_original"] = c["score"]
        c["score"] = c["score"] * bonus
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def write_candidate_hdf5(
    candidates: list[dict],
    obj_id: str,
    output_path: str,
    dataset: str = "",
    scale_factor: float = 1.0,
):
    """Write candidates in format compatible with a2g_pdm.py load_candidate_hdf5().

    HDF5 layout:
        metadata/
            attrs: obj_id, dataset, no_rotation, source, scale_factor
        candidates/
            candidate_0/
                position:     (3,) float32
                rotation:     (3, 3) float32
                attrs: gripper_width, score, name, approach_type
            candidate_1/
                ...
            attrs: n_candidates
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Metadata group (triggers L152 branch in a2g_pdm.py)
        meta = f.create_group("metadata")
        meta.attrs["obj_id"] = obj_id
        meta.attrs["dataset"] = dataset
        meta.attrs["no_rotation"] = True
        meta.attrs["source"] = "graspnet_baseline"
        meta.attrs["scale_factor"] = float(scale_factor)

        # Candidates group (triggers L167 branch in a2g_pdm.py)
        cg = f.create_group("candidates")
        for i, c in enumerate(candidates):
            gi = cg.create_group(f"candidate_{i}")
            gi.create_dataset("position", data=c["position"])
            gi.create_dataset("rotation", data=c["rotation"])
            gi.attrs["gripper_width"] = c["gripper_width"]
            gi.attrs["score"] = c["score"]
            gi.attrs["name"] = f"graspnet_{i}"
            gi.attrs["approach_type"] = "graspnet_baseline"

        cg.attrs["n_candidates"] = len(candidates)

    print(
        f"  💾 HDF5: {len(candidates)} candidates → {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GraspNet output to A2G candidate HDF5"
    )
    parser.add_argument("--grasp-npy", type=str, required=True,
                        help="Path to dumped GraspGroup .npy (N, 17)")
    parser.add_argument("--obj-id", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--z-approach-max", type=float, default=0.3,
                        help="Filter: max approach z-component (reject from-below)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    args = parser.parse_args()

    raw = np.load(args.grasp_npy)
    print(f"  Loaded {len(raw)} raw grasps from {args.grasp_npy}")

    candidates = graspgroup_to_candidates(
        raw,
        scale_factor=args.scale_factor,
        z_approach_max=args.z_approach_max,
    )
    print(f"  After conversion + filtering: {len(candidates)} valid candidates")

    write_candidate_hdf5(
        candidates,
        obj_id=args.obj_id,
        output_path=args.output,
        dataset=args.dataset,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
