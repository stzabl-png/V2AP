"""
Retarget: candidates.json (mesh frame) → robot-executable EE poses (base frame).

Converts grasp candidates from the 5090 processing pipeline into
4×4 homogeneous EE target poses that the Dexmate Vega arm can reach.

Usage (standalone test):
    python demo/phase2/retarget.py \
        --candidates demo/phase2/sessions/<id>/output/inference/candidates.json \
        --calib demo/phase2/calib/ee_retarget.yaml

Called from run_auto_grasp.py — no direct hardware access here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.grasp_geometry import (
    approach_dir_from_pose,
    compute_lift_pose,
    compute_pre_grasp_pose,
    format_pose,
    normalize,
)


# ── Data classes ───────────────────────────────────────────

@dataclass
class RetargetedGrasp:
    """One grasp candidate fully resolved in robot base frame."""

    rank: int
    name: str
    score: float
    gripper_width_m: float

    # Mesh-frame originals (for debugging / visualization)
    grasp_point_mesh: np.ndarray  # (3,)
    rotation_mesh: np.ndarray     # (3, 3)

    # Base-frame targets (for IK / execution)
    T_base_ee: np.ndarray         # 4×4: EE target pose
    T_base_pre: np.ndarray        # 4×4: pre-grasp (retreat along -approach)
    T_base_lift: np.ndarray       # 4×4: lift (+Z world)
    approach_base: np.ndarray     # (3,): approach direction in base frame


@dataclass
class RetargetResult:
    """All retargeted grasps + metadata from one session."""

    session_id: str
    policy: str
    candidates: list[RetargetedGrasp] = field(default_factory=list)
    T_base_mesh: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_ee_pinch: np.ndarray = field(default_factory=lambda: np.eye(4))


# ── Calibration ────────────────────────────────────────────

_DEFAULT_CALIB_PATH = Path(__file__).parent / "calib" / "ee_retarget.yaml"


def load_ee_retarget(
    calib_path: str | Path | None = None,
) -> np.ndarray:
    """Load T_ee_pinch from ee_retarget.yaml.

    Returns:
        4×4 numpy array: transform from pinch (thumb–index midpoint) to R_ee.
    """
    path = Path(calib_path or _DEFAULT_CALIB_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run pose_tuner.py to calibrate T_ee_pinch first."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    T = np.array(data["T_ee_pinch"], dtype=np.float64)
    assert T.shape == (4, 4), f"T_ee_pinch must be 4×4, got {T.shape}"
    return T


# ── Core retarget ──────────────────────────────────────────

def retarget_candidate(
    cand: dict,
    T_base_mesh: np.ndarray,
    T_ee_pinch: np.ndarray,
    *,
    pre_grasp_offset_m: float = 0.15,
    lift_height_m: float = 0.15,
) -> RetargetedGrasp:
    """Convert one candidates.json entry → RetargetedGrasp in base frame.

    Coordinate chain:
        mesh frame → T_base_mesh → base frame (pinch) → inv(T_ee_pinch) → EE frame
    """
    grasp_point = np.asarray(cand["grasp_point"], dtype=np.float64)
    rotation = np.asarray(cand["rotation"], dtype=np.float64)

    # Build 4×4 pinch pose in mesh frame
    T_mesh_pinch = np.eye(4)
    T_mesh_pinch[:3, :3] = rotation
    T_mesh_pinch[:3, 3] = grasp_point

    # Mesh → base
    T_base_pinch = T_base_mesh @ T_mesh_pinch

    # Pinch → EE (undo the calibrated offset)
    T_base_ee = T_base_pinch @ np.linalg.inv(T_ee_pinch)

    # Approach direction in base frame
    # approach = rotation column 2 (UCB convention)
    approach_mesh = rotation[:, 2]
    approach_base = T_base_mesh[:3, :3] @ approach_mesh
    approach_base = normalize(approach_base)

    # Pre-grasp: retreat along -approach in base frame
    T_base_pre = compute_pre_grasp_pose(
        T_base_ee,
        offset_m=pre_grasp_offset_m,
        approach_in_base=approach_base,
    )

    # Lift: +Z world from grasp pose
    T_base_lift = compute_lift_pose(T_base_ee, lift_height_m)

    return RetargetedGrasp(
        rank=int(cand.get("rank", 0)),
        name=str(cand.get("name", "unknown")),
        score=float(cand.get("score", 0.0)),
        gripper_width_m=float(cand.get("gripper_width_m", 0.06)),
        grasp_point_mesh=grasp_point,
        rotation_mesh=rotation,
        T_base_ee=T_base_ee,
        T_base_pre=T_base_pre,
        T_base_lift=T_base_lift,
        approach_base=approach_base,
    )


def retarget_session(
    candidates_json_path: str | Path,
    *,
    calib_path: str | Path | None = None,
    pre_grasp_offset_m: float = 0.15,
    lift_height_m: float = 0.15,
) -> RetargetResult:
    """Load candidates.json and retarget all candidates to base-frame EE poses.

    Args:
        candidates_json_path: Path to candidates.json from 5090 pipeline.
        calib_path: Path to ee_retarget.yaml (default: calib/ee_retarget.yaml).
        pre_grasp_offset_m: Pre-grasp retreat distance.
        lift_height_m: Lift height.

    Returns:
        RetargetResult with all candidates retargeted.
    """
    path = Path(candidates_json_path)
    with open(path) as f:
        data = json.load(f)

    assert data.get("schema_version", "").startswith("1."), (
        f"Unsupported schema: {data.get('schema_version')}"
    )

    T_base_mesh = np.array(data["T_base_mesh"], dtype=np.float64)
    T_ee_pinch = load_ee_retarget(calib_path)

    result = RetargetResult(
        session_id=path.parent.parent.parent.name,
        policy=data.get("source", "unknown"),
        T_base_mesh=T_base_mesh,
        T_ee_pinch=T_ee_pinch,
    )

    for cand in data.get("candidates", []):
        rg = retarget_candidate(
            cand,
            T_base_mesh,
            T_ee_pinch,
            pre_grasp_offset_m=pre_grasp_offset_m,
            lift_height_m=lift_height_m,
        )
        result.candidates.append(rg)

    return result


# ── CLI ────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retarget candidates.json → EE poses")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.json")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to ee_retarget.yaml")
    args = parser.parse_args()

    result = retarget_session(args.candidates, calib_path=args.calib)
    print(f"\nSession: {result.session_id}")
    print(f"Policy:  {result.policy}")
    print(f"T_ee_pinch diagonal: {np.diag(result.T_ee_pinch)}")
    print(f"\n{len(result.candidates)} candidates retargeted:\n")

    for rg in result.candidates:
        print(f"  [{rg.rank}] {rg.name:>16s}  score={rg.score:.3f}  "
              f"width={rg.gripper_width_m*100:.1f}cm")
        print(f"       approach_base = [{rg.approach_base[0]:+.3f}, "
              f"{rg.approach_base[1]:+.3f}, {rg.approach_base[2]:+.3f}]")
        print(format_pose(rg.T_base_ee, "       T_base_ee"))
        print()


if __name__ == "__main__":
    main()
