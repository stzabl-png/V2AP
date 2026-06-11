"""Phase 2 session capture constants."""

from pathlib import Path

import numpy as np

from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M

_PHASE2_DIR = Path(__file__).resolve().parent
PHASE2_DIR = _PHASE2_DIR
SESSIONS_DIR = _PHASE2_DIR / "sessions"
CALIB_DIR = _PHASE2_DIR / "calib"
DEFAULT_HEAD_INTRINSICS_JSON = CALIB_DIR / "head_zed_left_intrinsics.json"
DEFAULT_HEAD_INTRINSICS_K_NPY = CALIB_DIR / "K.npy"
DEFAULT_OPEN_GRIP_YAML = CALIB_DIR / "open_grip.yaml"

SCHEMA_VERSION = "1.1"
REGISTRATION_METHOD = "foundationpose"
DEFAULT_TITAN_MAX_CANDIDATES = 50
DEFAULT_GRASP_ATTEMPTS = 1
# FoundationPose scene depth: uint16 PNG, millimeters (UCB batch_obj_pose / YcbineoatReader).
FP_DEPTH_MM_SCALE = 1000.0
FP_FRAME_INDEX = "000000"
CAMERA_FRAME = "zed_left_camera"
BASE_FRAME = "base"
EE_FRAME = "R_ee"

# ZED left RGB optical frame convention (see phase2/README.md).
DEPTH_UNIT = "meters"
DEPTH_INVALID_VALUES = (0.0, float("nan"), float("inf"), float("-inf"))

# Table obstacle matches teleop/robot_descriptions.add_env_obstacles (Phase 1 planner).
TABLE_COLLISION_SIZE_XYZ_M = (2.0, 4.0, 0.08)
TABLE_COLLISION_CENTER_XYZ_M = (1.1, 0.0, None)  # z filled from table_height
# Candidate IK uses the same table_height as execution OMPL (no extra offset).

DEFAULT_CAPTURE_COMPONENTS = ("torso", "head", "left_arm", "right_arm")

# Fallback intrinsics if dexsensor info lacks calibration (640×360 nominal ZED X Mini).
FALLBACK_INTRINSICS_640x360 = {
    "fx": 350.0,
    "fy": 350.0,
    "cx": 320.0,
    "cy": 180.0,
}
