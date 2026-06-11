"""Phase 1 demo constants (grasp configs, motion, tuning)."""

from pathlib import Path

import numpy as np

_PHASE1_DIR = Path(__file__).resolve().parent
PHASE1_CONFIG_DIR = _PHASE1_DIR / "configs"
DEFAULT_START_OBJECT_NAME = "start"

# Matches UCB sim/run_grasp_sim.py (pre-grasp retreat along -approach_dir).
PRE_GRASP_OFFSET_M = 0.15
LIFT_HEIGHT_M = 0.15

COMMAND_HZ = 30.0
ARM_ACTION_HZ = 300.0
RESET_DOF_ERR_TOL = 0.3
# Collision-planned moves (pre-grasp, lift, start, grasp joint targets). Teleop default is 0.6 rad/s.
PLANNED_MOTION_JOINT_SPEED_RAD_S = 0.3

LEFT_ARM_DEFAULT_JOINT_POS = np.array(
    [0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03], dtype=np.float64
)
RIGHT_ARM_DEFAULT_JOINT_POS = np.array(
    [-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03], dtype=np.float64
)
TORSO_DEFAULT_JOINT_POS = np.array([1.2, 2.27, 0.5], dtype=np.float64)
# head_j1 pitch only (demo look-down); j2/j3 yaw/roll stay 0 (Vega URDF head_j1 axis +Y).
HEAD_PITCH_DOWN_DEG = 20.0
HEAD_PITCH_DOWN_RAD = float(np.deg2rad(HEAD_PITCH_DOWN_DEG))
HEAD_DEFAULT_JOINT_POS = np.array([HEAD_PITCH_DOWN_RAD, 0.0, 0.0], dtype=np.float64)
# Applied to start + object configs on every run_sequence (overrides YAML head).
DEMO_HEAD_JOINT_POS = HEAD_DEFAULT_JOINT_POS.copy()

DEFAULT_JOINT_POS = {
    "left_arm": LEFT_ARM_DEFAULT_JOINT_POS,
    "right_arm": RIGHT_ARM_DEFAULT_JOINT_POS,
    "head": HEAD_DEFAULT_JOINT_POS,
    "torso": TORSO_DEFAULT_JOINT_POS,
}

DEFAULT_TABLE_HEIGHT_M = 0.85
DEFAULT_HOLD_TIME_S = 3.0

# Small keyboard nudges in pose_tuner (meters / radians).
TUNER_TRANSLATION_STEP_M = 0.005
TUNER_ROTATION_STEP_RAD = 0.03
TUNER_HAND_JOINT_STEP_RAD = 0.0067
TUNER_ARM_JOINT_STEP_RAD = 0.0067
