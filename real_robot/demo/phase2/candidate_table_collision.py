"""Candidate-stage table checks (Pinocchio box + explicit fingertip / pinch height)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin

from demo.phase1.config_io import GraspObjectConfig
from demo.phase2.open_grip_retarget_geometry import open_grip_finger_kinematics
from teleop.robot_descriptions import RIGHT_EE_FRAME
from demo.phase2.pinch_ik import pinch_target_for_move
from demo.phase2.table_height import (
    estimate_table_height_m_from_session,
    load_table_height_m,
)

# Matches teleop/robot_descriptions.add_env_obstacles table box (center z = h - 0.01, half_z = 0.04).
_TABLE_BOX_CENTER_Z_OFFSET_M = -0.01
_TABLE_BOX_HALF_Z_M = 0.04


def table_obstacle_top_z_m(table_height_m: float, *, height_offset_m: float = 0.0) -> float:
    """Upper face of the Pinocchio table collision box."""
    h = float(table_height_m) + float(height_offset_m)
    return h + _TABLE_BOX_CENTER_Z_OFFSET_M + _TABLE_BOX_HALF_Z_M


def table_clearance_floor_z_m(table_height_m: float, *, clearance_m: float) -> float:
    """Nominal table plane + clearance (for fingertip / pinch z checks)."""
    return float(table_height_m) + float(clearance_m)


def planning_table_height_m_from_session(
    session_dir,
    *,
    default: float,
) -> tuple[float, str]:
    """
    Table height for candidate filtering: ``max(depth ROI, scene/table.json)``.

    Depth-only estimate can sit below the packed table.json / Titan vis plane.
    """
    measured, src_meas = estimate_table_height_m_from_session(session_dir, default=default)
    json_h = load_table_height_m(session_dir, default=measured)
    h = float(max(measured, json_h))
    if h > measured + 1e-6:
        return h, f"max(depth={measured:.3f}, table.json={json_h:.3f}) m"
    return measured, src_meas


def _motion_label_for_phase(label: str) -> str:
    return "grasp_approach" if label == "grasp" else label


def _open_grip_fk(
    left_q: np.ndarray,
    right_q: np.ndarray,
    left_hand: np.ndarray,
    right_hand_open: np.ndarray,
    pin_robot: Any,
    assemble: Any,
) -> tuple[pin.Model, pin.Data]:
    q = assemble(
        {
            "left_arm": np.asarray(left_q, dtype=np.float64),
            "right_arm": np.asarray(right_q, dtype=np.float64),
            "left_hand": np.asarray(left_hand, dtype=np.float64),
            "right_hand": np.asarray(right_hand_open, dtype=np.float64),
        }
    )
    model = pin_robot.model
    data = pin_robot.data
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return model, data


def candidate_open_grip_height_samples(
    cfg: GraspObjectConfig,
    start_config: GraspObjectConfig,
    left_q: np.ndarray,
    right_q: np.ndarray,
    *,
    label: str,
    pin_robot: Any,
    assemble: Any,
) -> dict[str, float]:
    """Key z values in base frame after open-grip IK (for logging / clearance)."""
    model, data = _open_grip_fk(
        left_q,
        right_q,
        start_config.left_hand_joint_pos,
        cfg.hand_open_joint_pos,
        pin_robot,
        assemble,
    )
    kin = open_grip_finger_kinematics(model, data)
    ee_z = float(data.oMf[model.getFrameId(RIGHT_EE_FRAME)].translation[2])
    out = {
        "thumb_tip_z": float(kin.thumb_tip[2]),
        "index_tip_z": float(kin.index_tip[2]),
        "align_z": float(kin.midpoint[2]),
        "R_ee_z": ee_z,
    }
    if cfg.titan_T_base_pinch is not None:
        motion_label = _motion_label_for_phase(label)
        T_pinch = pinch_target_for_move(
            cfg.titan_T_base_pinch,
            label=motion_label,
            pre_grasp_offset_m=cfg.pre_grasp_offset_m,
            lift_height_m=cfg.lift_height_m,
        )
        out["pinch_z"] = float(T_pinch[2, 3])
    return out


def candidate_table_z_clearance_ok(
    cfg: GraspObjectConfig,
    start_config: GraspObjectConfig,
    left_q: np.ndarray,
    right_q: np.ndarray,
    *,
    label: str,
    pin_robot: Any,
    assemble: Any,
    clearance_m: float = 0.0,
) -> tuple[bool, str]:
    """
    Reject if any pinch / fingertip / EE z is below table plane + clearance.

    With ``clearance_m=0``, matches execution OMPL table plane.
    """
    samples = candidate_open_grip_height_samples(
        cfg,
        start_config,
        left_q,
        right_q,
        label=label,
        pin_robot=pin_robot,
        assemble=assemble,
    )
    floor_z = table_clearance_floor_z_m(cfg.table_height, clearance_m=clearance_m)
    min_key = min(samples, key=lambda k: samples[k])
    min_z = float(samples[min_key])
    if min_z + 1e-6 < floor_z:
        return (
            False,
            f"{label}: {min_key} z={min_z:.3f} < table+clearance z={floor_z:.3f} "
            f"(samples={{{', '.join(f'{k}={v:.3f}' for k, v in samples.items())}}})",
        )
    return True, ""
