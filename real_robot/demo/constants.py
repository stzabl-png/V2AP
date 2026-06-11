"""Shared constants for V2AP demo (hand hardware, stall close, profiles)."""

import os
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_HAND_PROFILE_PATH = _DEMO_DIR / "right_hand_profile.yaml"

HAND_JOINT_COUNT = 22
HAND_INTERPOLATE = True
# Wait for Sharpa interpolate to finish (apply-closed used to exit too early).
HAND_APPLY_WAIT_S = 2.0
HAND_TRACK_TOL_RAD = 0.03

# Stall-detecting close (thumb+index pinch joints 0-8 monitored).
STALL_CLOSE_STEP_RAD = 0.0083
STALL_CLOSE_POLL_S = 0.015
STALL_CLOSE_VEL_RAD = 0.002
STALL_CLOSE_HOLD_S = 0.1
STALL_CLOSE_MAX_S = 9.0
STALL_CLOSE_REACH_TOL_RAD = 0.01
STALL_CLOSE_MONITOR_JOINTS = tuple(range(9))
# Ignore stall until hand has time to start moving (avoids instant false stall).
STALL_CLOSE_GRACE_S = 0.3
STALL_CLOSE_MIN_STEPS = 8
STALL_CLOSE_MIN_PROGRESS_RAD = 0.02
# After pose_tuner ']' preview close (stall or target), hold closed then reopen.
PREVIEW_CLOSE_HOLD_S = 1.5

# Override via env on Razor if serials differ from main_teleop.py defaults.
LEFT_HAND_SERIAL = os.environ.get("V2AP_LEFT_HAND_SERIAL", "CD51973BCD51")
RIGHT_HAND_SERIAL = os.environ.get("V2AP_RIGHT_HAND_SERIAL", "C65E9038C65E")
