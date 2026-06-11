"""Open-grip virtual pinch retarget geometry (thumb/index DP + thumb MC).

At the arm target (hand **open** profile):

  Translation:
    - Tip line ``p_index - p_thumb`` parallel to Titan ``finger_open`` (col 0).
    - Tip alignment point on thumb→index segment (default: 2/3 from thumb,
      i.e. 1/3 of segment length from index tip) equals
      ``p_titan + forward_offset * approach``.

  Rotation (soft IK):
    - ``finger_open ⊥ approach`` (follows from Titan frame if ``d ∥ finger_open``).
    - Plane ``(thumb tip, thumb root, index tip)`` contains approach vector.
    - Palm side (not dorsum): robot ``cross(approach, finger_open)`` aligns Titan ``y_body``.

Frames: ``right_thumb_DP``, ``right_index_DP``, ``right_thumb_MC`` (thumb root).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pinocchio as pin

from demo.phase1.constants import DEFAULT_JOINT_POS
from demo.phase1.grasp_geometry import approach_dir_from_pose, normalize
from teleop.robot_descriptions import RIGHT_EE_FRAME, build_full_robot

DEFAULT_OPEN_PINCH_FORWARD_M = 0.015
# On thumb→index segment: 2/3 from thumb ≈ 1/3 line length back from index tip.
DEFAULT_TIP_ALIGN_ALPHA_FROM_THUMB = 2.0 / 3.0
# Hard gate (open hand): thumb root→tip vs approach angle ≤ 90° ⇒ dot ≥ 0.
DEFAULT_THUMB_APPROACH_DOT_MIN = 0.0
# Optional soft (--ik-palm-soft): y_robot·y_titan ≥ cos_min.
DEFAULT_PALM_LATERAL_COS_MIN = 0.5
# Backward-compatible alias
DEFAULT_PALM_APPROACH_COS_MIN = DEFAULT_PALM_LATERAL_COS_MIN

_THUMB_TIP = "right_thumb_DP"
_INDEX_TIP = "right_index_DP"
_THUMB_ROOT = "right_thumb_MC"
_EE = RIGHT_EE_FRAME


@dataclass(frozen=True)
class OpenGripFingerKinematics:
    thumb_tip: np.ndarray
    index_tip: np.ndarray
    thumb_root: np.ndarray
    midpoint: np.ndarray  # tip-line alignment point (not geometric midpoint)
    tip_line: np.ndarray
    tip_line_dir: np.ndarray


@dataclass(frozen=True)
class OpenGripRetargetErrors:
    midpoint: np.ndarray  # tip-line alignment point (not geometric midpoint)
    target_mid: np.ndarray
    pos_err_m: float
    parallel_err: float
    plane_err: float
    finger_open_dot: float
    palm_lateral_dot: float
    palm_err: float
    thumb_approach_dot: float


def finger_open_dir_from_pose(T_base_pinch: np.ndarray) -> np.ndarray:
    T = np.asarray(T_base_pinch, dtype=np.float64)
    return normalize(T[:3, 0])


def y_body_dir_from_pose(T_base_pinch: np.ndarray) -> np.ndarray:
    """Titan column 1: ``normalize(approach × finger_open)``."""
    a = approach_dir_from_pose(T_base_pinch)
    f = finger_open_dir_from_pose(T_base_pinch)
    return normalize(np.cross(a, f))


def aligned_robot_finger_open(
    kin: OpenGripFingerKinematics,
    T_base_pinch: np.ndarray,
) -> np.ndarray:
    """Robot thumb→index direction, sign-matched to Titan ``finger_open``."""
    f_tgt = finger_open_dir_from_pose(T_base_pinch)
    f_r = kin.tip_line_dir.copy()
    if float(np.dot(f_r, f_tgt)) < 0.0:
        f_r = -f_r
    return f_r


def robot_palm_side_dir(
    kin: OpenGripFingerKinematics,
    T_base_pinch: np.ndarray,
) -> np.ndarray:
    """
    Robot analogue of Titan ``y_body = normalize(approach × finger_open)``.

    Same formula as Titan but with sign-aligned open-hand ``finger_open`` FK.
    """
    a_tgt = approach_dir_from_pose(T_base_pinch)
    f_r = aligned_robot_finger_open(kin, T_base_pinch)
    y_r = np.cross(a_tgt, f_r)
    n_norm = float(np.linalg.norm(y_r))
    if n_norm < 1e-9:
        return y_body_dir_from_pose(T_base_pinch)
    return y_r / n_norm


def palm_lateral_alignment(
    kin: OpenGripFingerKinematics,
    T_base_pinch: np.ndarray,
    *,
    cos_min: float = DEFAULT_PALM_LATERAL_COS_MIN,
) -> tuple[float, float]:
    """
    Palm-side check: ``dot(y_robot, y_titan)`` (dorsum flip ⇒ ≈ −1).

    ``approach`` lies **in** the pinch plane (plane soft constraint), so testing
    ``dot(n_palm, approach)`` is the wrong axis — use lateral ``y_body`` instead.
    """
    y_tgt = y_body_dir_from_pose(T_base_pinch)
    y_r = robot_palm_side_dir(kin, T_base_pinch)
    dot = float(np.dot(y_r, y_tgt))
    return dot, max(0.0, float(cos_min) - dot)


def thumb_root_to_tip_dir(kin: OpenGripFingerKinematics) -> np.ndarray:
    """Unit vector thumb MC → thumb DP (open-hand FK)."""
    v = kin.thumb_tip - kin.thumb_root
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def thumb_axis_approach_alignment(
    kin: OpenGripFingerKinematics,
    T_base_pinch: np.ndarray,
) -> float:
    """``dot(normalize(thumb_root→thumb_tip), approach)``; hard gate requires ≥ 0 (≤ 90°)."""
    a_tgt = approach_dir_from_pose(T_base_pinch)
    return float(np.dot(thumb_root_to_tip_dir(kin), a_tgt))


def tip_line_alignment_point(
    p_thumb: np.ndarray,
    p_index: np.ndarray,
    *,
    alpha_from_thumb: float = DEFAULT_TIP_ALIGN_ALPHA_FROM_THUMB,
) -> np.ndarray:
    """
    Point on the thumb→index tip segment used for translation retarget.

    ``alpha_from_thumb=2/3`` places the point one third of the segment length
    inward from the index tip (biased toward index vs geometric midpoint).
    """
    p_t = np.asarray(p_thumb, dtype=np.float64)
    p_i = np.asarray(p_index, dtype=np.float64)
    return p_t + float(alpha_from_thumb) * (p_i - p_t)


def target_open_grip_align_point(
    T_base_pinch: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
) -> np.ndarray:
    """``p = p_titan + forward_offset * approach`` (open-hand alignment target)."""
    T = np.asarray(T_base_pinch, dtype=np.float64)
    p_titan = T[:3, 3]
    approach = approach_dir_from_pose(T)
    return p_titan + float(forward_offset_m) * approach


def target_open_grip_midpoint(
    T_base_pinch: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
) -> np.ndarray:
    """Backward-compatible alias for :func:`target_open_grip_align_point`."""
    return target_open_grip_align_point(T_base_pinch, forward_offset_m=forward_offset_m)


def _frame_origin(model: pin.Model, data: pin.Data, fid: int) -> np.ndarray:
    return data.oMf[fid].translation.copy()


def open_grip_finger_kinematics(model: pin.Model, data: pin.Data) -> OpenGripFingerKinematics:
    p_t = _frame_origin(model, data, model.getFrameId(_THUMB_TIP))
    p_i = _frame_origin(model, data, model.getFrameId(_INDEX_TIP))
    p_r = _frame_origin(model, data, model.getFrameId(_THUMB_ROOT))
    d = p_i - p_t
    d_norm = float(np.linalg.norm(d))
    d_dir = d / d_norm if d_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    align = tip_line_alignment_point(p_t, p_i)
    return OpenGripFingerKinematics(
        thumb_tip=p_t,
        index_tip=p_i,
        thumb_root=p_r,
        midpoint=align,
        tip_line=d,
        tip_line_dir=d_dir,
    )


def open_grip_retarget_errors(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    T_base_pinch: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
) -> OpenGripRetargetErrors:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    kin = open_grip_finger_kinematics(model, data)
    target_mid = target_open_grip_align_point(T_base_pinch, forward_offset_m=forward_offset_m)
    f_tgt = finger_open_dir_from_pose(T_base_pinch)
    a_tgt = approach_dir_from_pose(T_base_pinch)

    e_pos = kin.midpoint - target_mid
    parallel = np.cross(kin.tip_line_dir, f_tgt)
    v_tr = kin.thumb_tip - kin.thumb_root
    v_ir = kin.index_tip - kin.thumb_root
    n = np.cross(v_tr, v_ir)
    n_norm = float(np.linalg.norm(n))
    plane = float(np.dot(a_tgt, n / n_norm)) if n_norm > 1e-9 else 0.0
    palm_dot, palm_short = palm_lateral_alignment(kin, T_base_pinch)
    thumb_app_dot = thumb_axis_approach_alignment(kin, T_base_pinch)

    return OpenGripRetargetErrors(
        midpoint=kin.midpoint,
        target_mid=target_mid,
        pos_err_m=float(np.linalg.norm(e_pos)),
        parallel_err=float(np.linalg.norm(parallel)),
        plane_err=plane,
        finger_open_dot=float(np.dot(kin.tip_line_dir, f_tgt)),
        palm_lateral_dot=palm_dot,
        palm_err=palm_short,
        thumb_approach_dot=thumb_app_dot,
    )


def open_grip_error_vector(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    T_base_pinch: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
    include_palm_soft: bool = True,
) -> tuple[np.ndarray, OpenGripRetargetErrors]:
    """Stack ``[e_pos(3), e_parallel(3), e_plane(1), e_palm(1)?]`` for IK."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    kin = open_grip_finger_kinematics(model, data)
    target_mid = target_open_grip_align_point(T_base_pinch, forward_offset_m=forward_offset_m)
    f_tgt = finger_open_dir_from_pose(T_base_pinch)
    a_tgt = approach_dir_from_pose(T_base_pinch)

    e_pos = kin.midpoint - target_mid
    e_par = np.cross(kin.tip_line_dir, f_tgt)
    v_tr = kin.thumb_tip - kin.thumb_root
    v_ir = kin.index_tip - kin.thumb_root
    n = np.cross(v_tr, v_ir)
    n_norm = float(np.linalg.norm(n))
    e_plane = np.array(
        [float(np.dot(a_tgt, n / n_norm)) if n_norm > 1e-9 else 0.0],
        dtype=np.float64,
    )
    palm_dot, palm_short = palm_lateral_alignment(kin, T_base_pinch)
    thumb_app_dot = thumb_axis_approach_alignment(kin, T_base_pinch)
    e_palm = np.array([palm_short], dtype=np.float64)
    err = OpenGripRetargetErrors(
        midpoint=kin.midpoint,
        target_mid=target_mid,
        pos_err_m=float(np.linalg.norm(e_pos)),
        parallel_err=float(np.linalg.norm(e_par)),
        plane_err=float(e_plane[0]),
        finger_open_dot=float(np.dot(kin.tip_line_dir, f_tgt)),
        palm_lateral_dot=palm_dot,
        palm_err=palm_short,
        thumb_approach_dot=thumb_app_dot,
    )
    parts = [e_pos, e_par, e_plane]
    if include_palm_soft:
        parts.append(e_palm)
    return np.concatenate(parts), err


def open_grip_midpoint_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    right_arm_v_indices: np.ndarray,
) -> np.ndarray:
    fid_t = model.getFrameId(_THUMB_TIP)
    fid_i = model.getFrameId(_INDEX_TIP)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    ref = pin.LOCAL_WORLD_ALIGNED
    Jt = pin.getFrameJacobian(model, data, fid_t, ref)[:3, :]
    Ji = pin.getFrameJacobian(model, data, fid_i, ref)[:3, :]
    alpha = DEFAULT_TIP_ALIGN_ALPHA_FROM_THUMB
    return ((1.0 - alpha) * Jt + alpha * Ji)[:, right_arm_v_indices]


def open_grip_error_jacobian_fd(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    T_base_pinch: np.ndarray,
    right_arm_v_indices: np.ndarray,
    *,
    forward_offset_m: float = DEFAULT_OPEN_PINCH_FORWARD_M,
    include_palm_soft: bool = True,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-diff Jacobian of full open-grip error vector w.r.t. right arm."""
    e0, _ = open_grip_error_vector(
        model,
        data,
        q,
        T_base_pinch,
        forward_offset_m=forward_offset_m,
        include_palm_soft=include_palm_soft,
    )
    n_arm = len(right_arm_v_indices)
    J = np.zeros((len(e0), n_arm))
    for i, vidx in enumerate(right_arm_v_indices):
        dq = np.zeros(model.nv)
        dq[vidx] = eps
        q_plus = pin.integrate(model, q, dq)
        pin.forwardKinematics(model, data, q_plus)
        pin.updateFramePlacements(model, data)
        e1, _ = open_grip_error_vector(
            model,
            data,
            q_plus,
            T_base_pinch,
            forward_offset_m=forward_offset_m,
            include_palm_soft=include_palm_soft,
        )
        J[:, i] = (e1 - e0) / eps
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return J, e0


def fk_R_ee_in_base(model: pin.Model, data: pin.Data) -> np.ndarray:
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(_EE)].homogeneous.copy()


def base_ee_from_open_grip_fk(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    right_hand_open: np.ndarray,
    *,
    pin_robot: Any | None = None,
    assemble: Any | None = None,
) -> np.ndarray:
    """``T_base_ee`` from FK after open-grip arm IK (for grasp_pose logging / OMPL seed)."""
    if pin_robot is None or assemble is None:
        pin_robot, assemble, _ = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
    q = assemble(
        {
            "left_arm": np.asarray(left_arm, dtype=np.float64),
            "right_arm": np.asarray(right_arm, dtype=np.float64),
            "left_hand": np.zeros(22, dtype=np.float64),
            "right_hand": np.asarray(right_hand_open, dtype=np.float64),
        }
    )
    pin.forwardKinematics(pin_robot.model, pin_robot.data, q)
    return fk_R_ee_in_base(pin_robot.model, pin_robot.data)


def right_fingertip_positions_in_base(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    right_hand: np.ndarray,
    *,
    pin_robot: Any | None = None,
    assemble: Any | None = None,
) -> dict[str, np.ndarray]:
    """Thumb/index distal positions and tip-line alignment point in robot base frame."""
    if pin_robot is None or assemble is None:
        pin_robot, assemble, _ = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
    q = assemble(
        {
            "left_arm": np.asarray(left_arm, dtype=np.float64),
            "right_arm": np.asarray(right_arm, dtype=np.float64),
            "left_hand": np.zeros(22, dtype=np.float64),
            "right_hand": np.asarray(right_hand, dtype=np.float64),
        }
    )
    model = pin_robot.model
    data = pin_robot.data
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_thumb = data.oMf[model.getFrameId(_THUMB_TIP)].translation.copy()
    p_index = data.oMf[model.getFrameId(_INDEX_TIP)].translation.copy()
    return {
        "thumb_tip": p_thumb,
        "index_tip": p_index,
        "pinch_mid": tip_line_alignment_point(p_thumb, p_index),
    }
