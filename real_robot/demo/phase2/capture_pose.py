"""Move robot to Phase 1 demo start pose before RGB-D capture."""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from demo.hand_close import read_hand_joint_pos
from demo.hardware import HardwareHandles
from demo.phase1.config_io import GraspObjectConfig, grasp_config_path, load_grasp_config
from demo.phase1.constants import (
    DEFAULT_START_OBJECT_NAME,
    DEMO_HEAD_JOINT_POS,
    HEAD_PITCH_DOWN_DEG,
    PHASE1_CONFIG_DIR,
)
from demo.phase1.executor import GraspExecutor
from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

# arm_j3 is index 2 in [j1..j7]; rotate outward (mirror sign L/R) from start.yaml.
CAPTURE_ARM_J3_INDEX = 2
DEFAULT_CAPTURE_ARM_J3_OUTWARD_DEG = 0.0

# Optional slow stepped move (~0.16 rad/s). Default is normal set_joint_pos trajectory.
DEFAULT_CAPTURE_ARM_STEP_RAD = 0.008
DEFAULT_CAPTURE_ARM_POLL_S = 0.05
DEFAULT_CAPTURE_ARM_MAX_DURATION_S = 120.0
DEFAULT_CAPTURE_ARM_REACH_TOL_RAD = 0.025

# Dexcontrol arm default max_vel is 0.5 rad/s; capture uses half for safer moves.
DEFAULT_CAPTURE_ARM_MAX_VEL_RAD_S = 0.25
DEFAULT_CAPTURE_ARM_WAIT_S = 10.0


def load_start_config_for_capture(
    start_object_name: str = DEFAULT_START_OBJECT_NAME,
    config_dir=None,
) -> GraspObjectConfig:
    """Load start.yaml and apply the same demo head override as Phase 1 run_grasp."""
    path = grasp_config_path(start_object_name, config_dir or PHASE1_CONFIG_DIR)
    if not path.is_file():
        raise FileNotFoundError(
            f"Start pose config required for capture: {path}\n"
            f"Tune with: python demo/phase1/pose_tuner.py --object-name {start_object_name}"
        )
    cfg = load_grasp_config(path)
    cfg.head_joint_pos = DEMO_HEAD_JOINT_POS.copy()
    GraspExecutor.apply_hand_profile(cfg)
    return cfg


def build_capture_arm_joint_pos(
    start_config: GraspObjectConfig,
    *,
    j3_outward_deg: float = DEFAULT_CAPTURE_ARM_J3_OUTWARD_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Start arms with arm_j3 rotated outward so hands clear the head camera.

    Left arm_j3 −= outward; right arm_j3 += outward (mirror).
    """
    delta = float(np.deg2rad(j3_outward_deg))
    left = np.asarray(start_config.left_arm_joint_pos, dtype=np.float64).copy()
    right = np.asarray(start_config.right_arm_joint_pos, dtype=np.float64).copy()
    left[CAPTURE_ARM_J3_INDEX] -= delta
    right[CAPTURE_ARM_J3_INDEX] += delta
    return left, right


def _start_body_goal(start_config: GraspObjectConfig) -> dict[str, np.ndarray]:
    return {
        "torso": np.asarray(start_config.torso_joint_pos, dtype=np.float64),
        "head": np.asarray(start_config.head_joint_pos, dtype=np.float64),
        "left_arm": np.asarray(start_config.left_arm_joint_pos, dtype=np.float64),
        "right_arm": np.asarray(start_config.right_arm_joint_pos, dtype=np.float64),
    }


def _move_robot_normal(
    handles: HardwareHandles,
    goal: dict[str, np.ndarray],
    *,
    wait_s: float,
    label: str,
    max_vel_rad_s: float = DEFAULT_CAPTURE_ARM_MAX_VEL_RAD_S,
) -> None:
    """Dexcontrol trajectory at reduced joint speed (default half of arm nominal)."""
    logger.info(
        f"Moving {label} (max_vel={max_vel_rad_s:.2f} rad/s, wait_time={wait_s:.1f}s)..."
    )
    handles.robot.set_joint_pos(
        {k: v.tolist() for k, v in goal.items()},
        relative=False,
        wait_time=wait_s,
        wait_kwargs={"max_vel": float(max_vel_rad_s)},
    )


def _component_arrays_from_dict(
    joint_pos_dict: dict[str, float],
    components: tuple[str, ...],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for comp in components:
        names = DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES[comp]
        out[comp] = np.array([float(joint_pos_dict[n]) for n in names], dtype=np.float64)
    return out


def _read_robot_components(handles: HardwareHandles, components: tuple[str, ...]) -> dict[str, np.ndarray]:
    joint_pos_dict = handles.robot.get_joint_pos_dict(component=list(components))
    return _component_arrays_from_dict(joint_pos_dict, components)


def _components_reached(
    current: dict[str, np.ndarray],
    goal: dict[str, np.ndarray],
    tol_rad: float,
) -> bool:
    for comp, q_goal in goal.items():
        if float(np.max(np.abs(current[comp] - q_goal))) > tol_rad:
            return False
    return True


def move_robot_components_stepped(
    handles: HardwareHandles,
    goal: dict[str, np.ndarray],
    *,
    step_rad: float = DEFAULT_CAPTURE_ARM_STEP_RAD,
    poll_s: float = DEFAULT_CAPTURE_ARM_POLL_S,
    reach_tol_rad: float = DEFAULT_CAPTURE_ARM_REACH_TOL_RAD,
    max_duration_s: float = DEFAULT_CAPTURE_ARM_MAX_DURATION_S,
) -> None:
    """
    Step torso/head/arms toward ``goal`` with a per-joint cap on each command.

    Press **Ctrl+C** to stop if something looks wrong.
    """
    components = tuple(goal.keys())
    logger.info(
        f"Slow capture move: max {step_rad:.4f} rad/step, "
        f"~{step_rad / poll_s:.2f} rad/s — Ctrl+C to abort"
    )
    t0 = time.monotonic()
    steps = 0
    try:
        while time.monotonic() - t0 < max_duration_s:
            current = _read_robot_components(handles, components)
            if _components_reached(current, goal, reach_tol_rad):
                logger.info(f"Slow capture move reached goal in {steps} steps")
                return

            cmd: dict[str, list[float]] = {}
            for comp in components:
                q_cur = current[comp]
                q_goal = goal[comp]
                delta = np.clip(q_goal - q_cur, -step_rad, step_rad)
                cmd[comp] = (q_cur + delta).tolist()

            handles.robot.set_joint_pos(cmd, relative=False, wait_time=0)
            steps += 1
            time.sleep(poll_s)
    except KeyboardInterrupt:
        logger.error("Capture move aborted (Ctrl+C). Robot left at last commanded step.")
        raise SystemExit(1) from None

    raise RuntimeError(
        f"Slow capture move timed out after {max_duration_s:.0f}s ({steps} steps)"
    )


def move_to_capture_start_pose(
    handles: HardwareHandles,
    start_config: GraspObjectConfig,
    *,
    arm_wait_s: float = DEFAULT_CAPTURE_ARM_WAIT_S,
    settle_s: float = 1.0,
    j3_outward_deg: float = DEFAULT_CAPTURE_ARM_J3_OUTWARD_DEG,
    arm_step_rad: float = DEFAULT_CAPTURE_ARM_STEP_RAD,
    arm_poll_s: float = DEFAULT_CAPTURE_ARM_POLL_S,
    arm_max_duration_s: float = DEFAULT_CAPTURE_ARM_MAX_DURATION_S,
    use_slow_arm_move: bool = False,
) -> None:
    """
    Capture sequence:

      1. Torso / head / **both arms** → ``start.yaml`` (normal speed)
      2. Optional arm_j3 ± outward when ``j3_outward_deg > 0`` (legacy camera clearance)
      3. Hands are **not** commanded (stay at current pose, typically zero)
    """
    start_body = _start_body_goal(start_config)

    logger.info(
        f"Capture sequence from start '{start_config.object_name}' "
        f"(head_j1 pitch down {HEAD_PITCH_DOWN_DEG:.0f}°)..."
    )

    if use_slow_arm_move:
        move_robot_components_stepped(
            handles,
            start_body,
            step_rad=arm_step_rad,
            poll_s=arm_poll_s,
            max_duration_s=arm_max_duration_s,
        )
    else:
        _move_robot_normal(handles, start_body, wait_s=arm_wait_s, label="to start pose")

    if j3_outward_deg > 0:
        left_arm_capture, right_arm_capture = build_capture_arm_joint_pos(
            start_config,
            j3_outward_deg=j3_outward_deg,
        )
        capture_arms = {
            "left_arm": left_arm_capture,
            "right_arm": right_arm_capture,
        }
        if use_slow_arm_move:
            move_robot_components_stepped(
                handles,
                capture_arms,
                step_rad=arm_step_rad,
                poll_s=arm_poll_s,
                max_duration_s=arm_max_duration_s,
            )
        else:
            _move_robot_normal(
                handles,
                capture_arms,
                wait_s=arm_wait_s,
                label=f"arm_j3 outward {j3_outward_deg:.0f}° (L−/R+)",
            )
        arm_log = (
            f"left_arm={np.round(left_arm_capture, 3).tolist()}, "
            f"right_arm={np.round(right_arm_capture, 3).tolist()}"
        )
    else:
        arm_log = "arms at start.yaml (no j3 spread)"

    if settle_s > 0:
        time.sleep(settle_s)

    logger.info(f"Capture pose reached: {arm_log} (hands unchanged)")


def return_to_start_pose_after_capture(
    handles: HardwareHandles,
    start_config: GraspObjectConfig,
    *,
    arm_wait_s: float = DEFAULT_CAPTURE_ARM_WAIT_S,
    arm_step_rad: float = DEFAULT_CAPTURE_ARM_STEP_RAD,
    arm_poll_s: float = DEFAULT_CAPTURE_ARM_POLL_S,
    arm_max_duration_s: float = DEFAULT_CAPTURE_ARM_MAX_DURATION_S,
    use_slow_arm_move: bool = False,
) -> None:
    """After capture: torso/head/arms → start.yaml; **do not command hands**."""
    start_body = _start_body_goal(start_config)
    logger.info("Returning to start pose after capture (arms/torso/head only; hands unchanged)")

    if use_slow_arm_move:
        move_robot_components_stepped(
            handles,
            start_body,
            step_rad=arm_step_rad,
            poll_s=arm_poll_s,
            max_duration_s=arm_max_duration_s,
        )
    else:
        _move_robot_normal(handles, start_body, wait_s=arm_wait_s, label="back to start pose")

    logger.info("Start pose restored (hands still at initial start-pose finger state)")


def read_capture_robot_state(handles: HardwareHandles) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Flat dexmate joint dict + left/right hand q (22-DOF each)."""
    joint_pos_dict = handles.robot.get_joint_pos_dict(
        component=["torso", "head", "left_arm", "right_arm"]
    )
    left_hand_q = read_hand_joint_pos(handles.left_hand)
    right_hand_q = read_hand_joint_pos(handles.right_hand)
    return joint_pos_dict, left_hand_q, right_hand_q


def apply_demo_head_to_config(cfg: GraspObjectConfig) -> None:
    """Same helper as GraspExecutor.apply_demo_head_pose for a single config."""
    GraspExecutor.apply_demo_head_pose(cfg)
