"""Open-grip retarget IK for Phase 2 Titan grasps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pinocchio as pin
from loguru import logger

from demo.phase1.constants import DEFAULT_JOINT_POS
from demo.phase1.grasp_geometry import approach_dir_from_pose
from demo.phase2.open_grip_retarget_geometry import (
    DEFAULT_OPEN_PINCH_FORWARD_M,
    DEFAULT_PALM_LATERAL_COS_MIN,
    DEFAULT_THUMB_APPROACH_DOT_MIN,
    open_grip_error_jacobian_fd,
    open_grip_error_vector,
    open_grip_midpoint_jacobian,
    open_grip_retarget_errors,
)
from teleop.robot_descriptions import DEXMATE_ARM_JOINT_ORDER, build_full_robot

DEFAULT_POS_WEIGHT = 100.0
DEFAULT_PARALLEL_WEIGHT = 8.0
DEFAULT_PLANE_WEIGHT = 8.0
DEFAULT_PALM_WEIGHT = 6.0
DEFAULT_REG_WEIGHT = 0.05
DEFAULT_MAX_ITERS = 120
DEFAULT_POS_TOL_M = 0.005
DEFAULT_PALM_ACCEPT_TOL = 1e-4
DEFAULT_DAMPING = 0.05


def _pos_converged(metrics, *, pos_tol_m: float) -> bool:
    """Soft IK stops when the translation target is met."""
    return metrics.pos_err_m <= pos_tol_m


def open_grip_orientation_acceptable(
    metrics,
    *,
    thumb_approach_dot_min: float = DEFAULT_THUMB_APPROACH_DOT_MIN,
    err_tol: float = DEFAULT_PALM_ACCEPT_TOL,
) -> bool:
    """Hard post-IK filter: open-hand thumb MC→DP vs approach angle ≤ 90° (dot ≥ 0)."""
    return float(metrics.thumb_approach_dot) >= float(thumb_approach_dot_min) - float(err_tol)


def palm_orientation_acceptable(
    metrics,
    *,
    palm_cos_min: float = DEFAULT_PALM_LATERAL_COS_MIN,
    err_tol: float = DEFAULT_PALM_ACCEPT_TOL,
) -> bool:
    """Legacy lateral check (``--ik-palm-soft`` only; not the post-IK hard gate)."""
    return float(metrics.palm_lateral_dot) >= float(palm_cos_min) - float(err_tol)


@dataclass(frozen=True)
class PinchIkResult:
    left_arm: np.ndarray
    right_arm: np.ndarray
    pos_err_m: float
    parallel_err: float
    plane_err: float
    finger_open_dot: float
    palm_lateral_dot: float
    palm_err: float
    thumb_approach_dot: float
    converged: bool
    palm_acceptable: bool


@lru_cache(maxsize=1)
def _full_robot_assemble():
    robot, assemble, disassemble = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
    model = robot.model
    right_arm_v_indices = np.array(
        [
            model.joints[model.getJointId(f"R_{name}")].idx_v
            for name in DEXMATE_ARM_JOINT_ORDER
        ],
        dtype=int,
    )
    q_lo = model.lowerPositionLimit.copy()
    q_hi = model.upperPositionLimit.copy()
    return robot, assemble, disassemble, right_arm_v_indices, q_lo, q_hi


def pinch_target_for_move(
    T_base_pinch: np.ndarray,
    *,
    label: str,
    pre_grasp_offset_m: float,
    lift_height_m: float,
) -> np.ndarray:
    """Map motion phase label → Titan ``T_base_pinch`` (pinch origin retreated for pre-grasp)."""
    T = np.asarray(T_base_pinch, dtype=np.float64).copy()
    if label in ("pre_grasp",):
        approach = approach_dir_from_pose(T)
        T[:3, 3] = T[:3, 3] - approach * float(pre_grasp_offset_m)
        return T
    if label in ("grasp", "grasp_approach"):
        return T
    if label == "lift":
        T[:3, 3] = T[:3, 3] + np.array([0.0, 0.0, float(lift_height_m)])
        return T
    return T


def solve_open_grip_retarget_ik(
    T_base_pinch_target: np.ndarray,
    left_arm: np.ndarray,
    right_arm_init: np.ndarray,
    right_hand_open: np.ndarray,
    *,
    pin_robot: Any | None = None,
    assemble: Any | None = None,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
    pos_weight: float = DEFAULT_POS_WEIGHT,
    parallel_weight: float = DEFAULT_PARALLEL_WEIGHT,
    plane_weight: float = DEFAULT_PLANE_WEIGHT,
    palm_weight: float = DEFAULT_PALM_WEIGHT,
    ik_palm_soft: bool = False,
    palm_cos_min: float = DEFAULT_PALM_LATERAL_COS_MIN,
    reg_weight: float = DEFAULT_REG_WEIGHT,
    max_iters: int = DEFAULT_MAX_ITERS,
    pos_tol_m: float = DEFAULT_POS_TOL_M,
    damping: float = DEFAULT_DAMPING,
) -> PinchIkResult:
    """
    Right-arm IK with **open-hand** FK.

    Translation: open tip-line align point (2/3 from thumb on thumb→index segment)
    ``= p_titan + forward_offset * approach``; tip line ``∥`` Titan ``finger_open``.

    Rotation (soft): plane contains approach; optionally palm-side ``y`` aligns Titan
    ``y_body`` (``ik_palm_soft``). Hard gate: thumb MC→DP vs approach ≤ 90°.
    """
    if pin_robot is None or assemble is None:
        pin_robot, assemble, disassemble, right_arm_v_indices, q_lo, q_hi = _full_robot_assemble()
    else:
        _, _, disassemble, right_arm_v_indices, q_lo, q_hi = _full_robot_assemble()
        q_lo = pin_robot.model.lowerPositionLimit.copy()
        q_hi = pin_robot.model.upperPositionLimit.copy()

    model = pin_robot.model
    data = pin_robot.data
    T_base_pinch_target = np.asarray(T_base_pinch_target, dtype=np.float64)

    left_arm = np.asarray(left_arm, dtype=np.float64)
    right_arm = np.asarray(right_arm_init, dtype=np.float64).copy()
    right_hand_open = np.asarray(right_hand_open, dtype=np.float64)

    q = assemble(
        {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_hand": np.zeros(22, dtype=np.float64),
            "right_hand": right_hand_open,
        }
    )

    w_parts = [pos_weight] * 3 + [parallel_weight] * 3 + [plane_weight]
    if ik_palm_soft:
        w_parts.append(palm_weight)
    w = np.array(w_parts, dtype=np.float64)

    metrics = open_grip_retarget_errors(
        model, data, q, T_base_pinch_target, forward_offset_m=forward_offset_m
    )
    pos_err = metrics.pos_err_m

    for _ in range(max_iters):
        metrics = open_grip_retarget_errors(
            model, data, q, T_base_pinch_target, forward_offset_m=forward_offset_m
        )
        pos_err = metrics.pos_err_m
        if _pos_converged(metrics, pos_tol_m=pos_tol_m):
            break

        J_fd, e = open_grip_error_jacobian_fd(
            model,
            data,
            q,
            T_base_pinch_target,
            right_arm_v_indices,
            forward_offset_m=forward_offset_m,
            include_palm_soft=ik_palm_soft,
        )
        J = w[:, None] * J_fd
        e_w = w * e

        J_reg = np.sqrt(reg_weight) * np.eye(len(right_arm_v_indices))
        e_reg = np.sqrt(reg_weight) * (right_arm - right_arm_init)
        J_aug = np.vstack([J, J_reg])
        e_aug = np.concatenate([e_w, e_reg])

        dq_arm, *_ = np.linalg.lstsq(
            J_aug.T @ J_aug + damping * np.eye(len(right_arm_v_indices)),
            -J_aug.T @ e_aug,
            rcond=None,
        )
        dq_full = np.zeros(model.nv)
        dq_full[right_arm_v_indices] = dq_arm
        q = np.clip(pin.integrate(model, q, dq_full), q_lo, q_hi)
        right_arm = disassemble(q)["right_arm"]

    metrics = open_grip_retarget_errors(
        model, data, q, T_base_pinch_target, forward_offset_m=forward_offset_m
    )
    converged = _pos_converged(metrics, pos_tol_m=pos_tol_m)
    palm_ok = open_grip_orientation_acceptable(metrics)
    return PinchIkResult(
        left_arm=left_arm.copy(),
        right_arm=right_arm.copy(),
        pos_err_m=metrics.pos_err_m,
        parallel_err=metrics.parallel_err,
        plane_err=metrics.plane_err,
        finger_open_dot=metrics.finger_open_dot,
        palm_lateral_dot=metrics.palm_lateral_dot,
        palm_err=metrics.palm_err,
        thumb_approach_dot=metrics.thumb_approach_dot,
        converged=converged,
        palm_acceptable=palm_ok,
    )


# Backward-compatible alias used by executor.
def solve_pinch_first_ik(
    T_base_pinch_target: np.ndarray,
    left_arm: np.ndarray,
    right_arm_init: np.ndarray,
    right_hand_open: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
    **kwargs: Any,
) -> PinchIkResult:
    return solve_open_grip_retarget_ik(
        T_base_pinch_target,
        left_arm,
        right_arm_init,
        right_hand_open,
        forward_offset_m=forward_offset_m,
        **kwargs,
    )


def log_pinch_ik_result(result: PinchIkResult, *, label: str) -> None:
    logger.info(
        f"Open-grip IK ({label}): align_err={result.pos_err_m:.4f} m, "
        f"parallel_err={result.parallel_err:.4f}, plane_err={result.plane_err:.4f}, "
        f"thumb_axis·approach={result.thumb_approach_dot:.4f} "
        f"(hard gate ≥0, ≤90°), y_robot·y_titan={result.palm_lateral_dot:.4f}, "
        f"finger_open·={result.finger_open_dot:.4f}, "
        f"converged={result.converged}, orientation_ok={result.palm_acceptable}"
    )
