"""Compute T_base_cam from Vega URDF FK at capture joint positions."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from dexmate_urdf import robots

from demo.phase2.constants import BASE_FRAME, CAMERA_FRAME
from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES


def _vega_model_wrapper() -> pin.RobotWrapper:
    urdf = robots.humanoid.vega_1.vega_1.urdf
    package = robots.humanoid.vega_1.vega_1._parent_dir
    return pin.RobotWrapper.BuildFromURDF(str(urdf), [str(package)])


def joint_dict_to_component_arrays(joint_pos_dict: dict[str, float]) -> dict[str, np.ndarray]:
    """Group flat joint-name dict into component arrays (torso/head/arms)."""
    out: dict[str, np.ndarray] = {}
    for component, names in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES.items():
        if component not in ("torso", "head", "left_arm", "right_arm"):
            continue
        out[component] = np.array([float(joint_pos_dict[name]) for name in names], dtype=np.float64)
    return out


def compute_T_base_cam(
    joint_pos_dict: dict[str, float],
    camera_frame: str = CAMERA_FRAME,
) -> np.ndarray:
    """
    Forward kinematics: return 4×4 T_base_cam (p_base = T @ p_cam).

    Uses full Vega URDF with all provided joint positions; unspecified joints stay at neutral.
    """
    robot = _vega_model_wrapper()
    model = robot.model
    q = pin.neutral(model)

    for jname, value in joint_pos_dict.items():
        if not model.existJointName(jname):
            continue
        jid = model.getJointId(jname)
        if model.joints[jid].nq != 1:
            continue
        q[model.joints[jid].idx_q] = float(value)

    pin.forwardKinematics(model, robot.data, q)
    pin.updateFramePlacements(model, robot.data)

    if not model.existFrame(camera_frame):
        known = [f.name for f in model.frames if "zed" in f.name.lower() or "camera" in f.name.lower()]
        raise RuntimeError(
            f"Frame {camera_frame!r} not in URDF. Camera-like frames: {known[:12]}"
        )

    fid = model.getFrameId(camera_frame)
    T = robot.data.oMf[fid].homogeneous.copy()
    return T


def extrinsics_json(
    T_base_cam: np.ndarray,
    *,
    method: str = "urdf_fk",
    notes: str = "",
) -> dict:
    T = np.asarray(T_base_cam, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_cam must be 4×4, got {T.shape}")
    return {
        "base_frame": BASE_FRAME,
        "camera_frame": CAMERA_FRAME,
        "T_base_cam": T.tolist(),
        "method": method,
        "notes": notes or "FK from capture joints + URDF zed_left_camera mount",
    }
