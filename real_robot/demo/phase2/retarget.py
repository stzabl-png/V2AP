"""Titan candidates.json → Dexmate open-grip grasp targets in base frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from demo.phase1.grasp_geometry import (
    approach_dir_from_pose,
    compute_lift_pose,
    compute_pre_grasp_pose,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pinch_pose_in_mesh_frame(candidate: dict[str, Any]) -> np.ndarray:
    """4x4 pinch frame in ``base_aligned`` mesh coordinates."""
    R = np.asarray(candidate["rotation"], dtype=np.float64)
    t = np.asarray(candidate["grasp_point"], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pinch_pose_in_base(T_base_mesh: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    return np.asarray(T_base_mesh, dtype=np.float64) @ pinch_pose_in_mesh_frame(candidate)


def placeholder_ee_poses_from_pinch(
    T_base_pinch: np.ndarray,
    *,
    pre_grasp_offset_m: float,
    lift_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Placeholder ``R_ee`` poses until open-grip IK sync.

    Arm IK uses ``T_base_pinch`` directly; these are only for config fields and lift
    (lift is overwritten after grasp IK sync).
    """
    T_base_pinch = np.asarray(T_base_pinch, dtype=np.float64)
    grasp = np.eye(4, dtype=np.float64)
    grasp[:3, 3] = T_base_pinch[:3, 3]
    approach = approach_dir_from_pose(T_base_pinch)
    pre = compute_pre_grasp_pose(
        grasp, pre_grasp_offset_m, approach_in_base=approach
    )
    lift = compute_lift_pose(grasp, lift_height_m)
    return grasp, pre, lift


@dataclass
class TitanGraspPoses:
    rank: int
    name: str
    score: float
    T_base_mesh: np.ndarray
    T_base_pinch: np.ndarray
    grasp_pose: np.ndarray
    pre_grasp_pose: np.ndarray
    lift_pose: np.ndarray
    gripper_width_m: float | None


@dataclass
class TitanSessionOutput:
    session_id: str
    session_dir: Path
    status: dict[str, Any]
    candidates: dict[str, Any]
    T_base_mesh: np.ndarray
    mesh_span_m: np.ndarray


def load_titan_output(session_dir: Path) -> TitanSessionOutput:
    session_dir = Path(session_dir)
    status = _load_json(session_dir / "output" / "status.json")
    if not status.get("success"):
        raise RuntimeError(
            f"output/status.json success=false for {session_dir.name}: "
            f"{status.get('errors', [])}"
        )
    candidates = _load_json(session_dir / "output" / "inference" / "candidates.json")
    Tb = np.asarray(candidates["T_base_mesh"], dtype=np.float64)
    span = np.asarray(candidates.get("mesh_span_m", [0.15, 0.15, 0.15]), dtype=np.float64)
    return TitanSessionOutput(
        session_id=str(status.get("session_id", session_dir.name)),
        session_dir=session_dir,
        status=status,
        candidates=candidates,
        T_base_mesh=Tb,
        mesh_span_m=span,
    )


def candidate_poses(
    output: TitanSessionOutput,
    candidate: dict[str, Any],
    *,
    pre_grasp_offset_m: float | None = None,
    lift_height_m: float | None = None,
) -> TitanGraspPoses:
    """Titan candidate → ``T_base_pinch`` + placeholder EE poses (synced after open-grip IK)."""
    conv = output.candidates.get("conventions", {})
    pre_off = float(
        pre_grasp_offset_m
        if pre_grasp_offset_m is not None
        else conv.get("pre_grasp_offset_m", 0.15)
    )
    lift_h = float(
        lift_height_m if lift_height_m is not None else conv.get("lift_height_m", 0.15)
    )

    T_base_pinch = pinch_pose_in_base(output.T_base_mesh, candidate)
    grasp, pre, lift = placeholder_ee_poses_from_pinch(
        T_base_pinch, pre_grasp_offset_m=pre_off, lift_height_m=lift_h
    )
    return TitanGraspPoses(
        rank=int(candidate["rank"]),
        name=str(candidate.get("name", f"rank_{candidate['rank']}")),
        score=float(candidate.get("score", 0.0)),
        T_base_mesh=output.T_base_mesh.copy(),
        T_base_pinch=T_base_pinch,
        grasp_pose=grasp,
        pre_grasp_pose=pre,
        lift_pose=lift,
        gripper_width_m=candidate.get("gripper_width_m"),
    )


def iter_ranked_candidates(output: TitanSessionOutput) -> list[dict[str, Any]]:
    cands = list(output.candidates.get("candidates", []))
    return sorted(cands, key=lambda c: int(c.get("rank", 9999)))


def log_retarget_sanity(
    poses: TitanGraspPoses,
    *,
    forward_offset_m: float = 0.015,
    right_arm_q: np.ndarray | None = None,
    left_arm_q: np.ndarray | None = None,
    right_hand_open: np.ndarray | None = None,
) -> None:
    """Log open-grip FK alignment vs Titan after candidate IK."""
    from demo.phase1.constants import DEFAULT_JOINT_POS  # noqa: WPS433
    from demo.phase2.open_grip_retarget_geometry import (  # noqa: WPS433
        open_grip_retarget_errors as _open_err,
        target_open_grip_align_point,
    )
    from teleop.robot_descriptions import build_full_robot  # noqa: WPS433

    pinch = poses.T_base_pinch[:3, 3]
    ee = poses.grasp_pose[:3, 3]
    logger.info(f"Open-grip forward_offset_m={forward_offset_m:.3f}")
    target_mid = target_open_grip_align_point(poses.T_base_pinch, forward_offset_m=forward_offset_m)
    logger.info(
        f"Open-grip target align (p_titan + {forward_offset_m:.3f}m·approach): "
        f"{target_mid.round(3).tolist()}"
    )
    if right_arm_q is not None and left_arm_q is not None and right_hand_open is not None:
        robot, assemble, _ = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
        q = assemble(
            {
                "left_arm": left_arm_q,
                "right_arm": right_arm_q,
                "left_hand": np.zeros(22),
                "right_hand": right_hand_open,
            }
        )
        err = _open_err(
            robot.model, robot.data, q, poses.T_base_pinch, forward_offset_m=forward_offset_m
        )
        logger.info(
            f"Open-grip FK at IK arm q: align={err.midpoint.round(3).tolist()}, "
            f"align_err={err.pos_err_m:.4f} m, parallel_err={err.parallel_err:.4f}, "
            f"plane_err={err.plane_err:.4f}, thumb_axis·approach={err.thumb_approach_dot:.4f} "
            f"(hard ≤90°), y_robot·y_titan={err.palm_lateral_dot:.4f} "
            f"(palm_err={err.palm_err:.4f}), finger_open·={err.finger_open_dot:.4f}"
        )
        if err.thumb_approach_dot < -0.01:
            logger.warning(
                "Open-grip thumb axis vs approach > 90° — candidate should be rejected"
            )
        elif err.palm_err > 0.01:
            logger.warning(
                "Open-grip dorsum flip? y_robot·y_titan low (soft metric only)"
            )
    logger.info(f"R_ee grasp_pose xyz (after IK sync): {ee.round(3).tolist()}")
    logger.info(f"Titan pinch origin xyz: {pinch.round(3).tolist()}")


def log_titan_pinch_in_base(poses: TitanGraspPoses) -> None:
    """Log Titan candidate pinch frame origin and axes in robot base frame."""
    p = poses.T_base_pinch[:3, 3]
    R = poses.T_base_pinch[:3, :3]
    logger.info(
        f"=== Titan pinch (robot base) rank={poses.rank} ({poses.name}), "
        f"score={poses.score:.1f} ==="
    )
    logger.info(f"  origin xyz (m):   {p.round(4).tolist()}")
    logger.info(f"  finger_open axis: {R[:, 0].round(4).tolist()}")
    logger.info(f"  y_body axis:      {R[:, 1].round(4).tolist()}")
    logger.info(f"  approach axis:    {R[:, 2].round(4).tolist()}")
    if poses.gripper_width_m is not None:
        logger.info(f"  gripper_width_m:  {float(poses.gripper_width_m):.4f}")


def log_live_fingertips_vs_titan(
    tips: dict[str, np.ndarray],
    titan: TitanGraspPoses,
) -> None:
    """Log measured thumb/index tips at grasp vs Titan pinch origin."""
    titan_origin = titan.T_base_pinch[:3, 3]
    thumb = tips["thumb_tip"]
    index = tips["index_tip"]
    mid = tips["pinch_mid"]
    gap = float(np.linalg.norm(index - thumb))
    mid_err = float(np.linalg.norm(mid - titan_origin))
    logger.info("=== Live fingertips at grasp (robot base frame) ===")
    logger.info(f"  thumb tip xyz (m):  {thumb.round(4).tolist()}")
    logger.info(f"  index tip xyz (m):  {index.round(4).tolist()}")
    logger.info(f"  tip-line align xyz (m): {mid.round(4).tolist()}  (2/3 from thumb)")
    logger.info(f"  thumb-index gap (m): {gap:.4f}")
    logger.info(
        f"  vs Titan pinch origin (rank={titan.rank}): "
        f"align_err={mid_err:.4f} m, "
        f"thumb_err={float(np.linalg.norm(thumb - titan_origin)):.4f} m, "
        f"index_err={float(np.linalg.norm(index - titan_origin)):.4f} m"
    )
