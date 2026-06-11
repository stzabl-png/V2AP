"""R_arm_l8 distal (hand-mount face) geometry vs Pink IK frame R_ee.

Vega URDF (``R_ee_j0``): ``R_ee`` is fixed on ``R_arm_l8`` at the **same origin** with
``rpy = (0, 0, π/2)`` — axes differ from ``R_arm_l8``. The physical hand-mount face
(distal end of the last arm link) is **+5 cm along R_ee +Z** (``SHARPA_HAND_MOUNT_IN_EE``),
which equals **+5 cm along R_arm_l8 +Z** (Rz preserves Z).

Debug retarget uses a **constructed** arm-distal frame (+Y = Titan approach), not native
``R_ee`` / ``R_arm_l8`` axes. URDF hand mount remains +5 cm along **R_ee +Z** (unchanged).
"""

from __future__ import annotations

import numpy as np

from demo.phase1.grasp_geometry import approach_dir_from_pose, normalize

RIGHT_ARM_L8_FRAME = "R_arm_l8"
RIGHT_EE_FRAME = "R_ee"

# URDF R_ee_j0: parent R_arm_l8, xyz=0, rpy="0 0 1.57079"
R_EE_IN_L8_ROT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Hand-mount / arm-distal face in R_ee (same translation as Sharpa mount; rotation = I).
ARM_DISTAL_IN_EE_TRANSLATION_M = np.array([0.0, 0.0, 0.05], dtype=np.float64)


def T_ee_in_l8_homogeneous() -> np.ndarray:
    """``T_l8_ee`` from URDF (rotation only)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_EE_IN_L8_ROT
    return T


def T_arm_distal_in_ee_homogeneous() -> np.ndarray:
    """Physical arm-distal (mount face) in ``R_ee``: +5 cm along EE +Z."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = ARM_DISTAL_IN_EE_TRANSLATION_M
    return T


def rotation_with_ee_z_along(
    z_in_base: np.ndarray,
    *,
    x_hint_in_base: np.ndarray | None = None,
) -> np.ndarray:
    """
    ``R_ee`` rotation (same as ``right_hand_C_MC``) with **column 2 (+Z) = z_in_base**.

    Completes a right-hand frame; ``x_hint`` breaks ambiguity when ``z`` is vertical.
    """
    z = normalize(np.asarray(z_in_base, dtype=np.float64))
    hint = (
        np.asarray(x_hint_in_base, dtype=np.float64)
        if x_hint_in_base is not None
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    hint = normalize(hint)
    if abs(float(np.dot(hint, z))) > 0.95:
        hint = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x = hint - np.dot(hint, z) * z
    x = normalize(x)
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])


def T_base_ee_hand_z_along(
    z_in_base: np.ndarray,
    position_in_base: np.ndarray,
    *,
    x_hint_in_base: np.ndarray | None = None,
) -> np.ndarray:
    """``T_base_R_ee`` with ``R_ee`` +Z (hand +Z) aligned to ``z_in_base``."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_with_ee_z_along(z_in_base, x_hint_in_base=x_hint_in_base)
    T[:3, 3] = np.asarray(position_in_base, dtype=np.float64)
    return T


BASE_AXIS_SHOWCASE: tuple[tuple[str, np.ndarray], ...] = (
    ("base+Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ("base+X", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("base+Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
)


def T_arm_distal_in_l8_homogeneous() -> np.ndarray:
    """Arm-distal face in ``R_arm_l8`` (FK/URDF chain)."""
    return T_ee_in_l8_homogeneous() @ T_arm_distal_in_ee_homogeneous()


def _orthonormal_rotation_y_approach(
    approach: np.ndarray,
    finger_open_hint: np.ndarray,
) -> np.ndarray:
    """Right-hand frame columns [X, Y, Z] with Y = approach."""
    y = normalize(np.asarray(approach, dtype=np.float64))
    hint = normalize(np.asarray(finger_open_hint, dtype=np.float64))
    if abs(float(np.dot(hint, y))) > 0.95:
        hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(hint, y))) > 0.95:
            hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = hint - np.dot(hint, y) * y
    x = normalize(x)
    z = normalize(np.cross(x, y))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])


def T_base_arm_distal_pregrasp_from_pinch(
    T_base_pinch: np.ndarray,
    *,
    pre_grasp_offset_m: float = 0.15,
) -> np.ndarray:
    """
    Constructed arm-distal frame at pre-grasp.

    Origin = pinch - approach * offset (arm-distal **mount face**, not R_ee origin).
    +Y = Titan approach; +X ≈ Titan finger_open.
    """
    T_base_pinch = np.asarray(T_base_pinch, dtype=np.float64)
    approach = approach_dir_from_pose(T_base_pinch)
    finger_open = normalize(T_base_pinch[:3, 0])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _orthonormal_rotation_y_approach(approach, finger_open)
    T[:3, 3] = T_base_pinch[:3, 3] - approach * float(pre_grasp_offset_m)
    return T


def base_ee_from_arm_distal(T_base_arm_distal: np.ndarray) -> np.ndarray:
    """Pink IK target: ``T_base_ee = T_base_arm_distal @ inv(T_arm_distal_in_ee)``."""
    T_ad = np.asarray(T_base_arm_distal, dtype=np.float64)
    return T_ad @ np.linalg.inv(T_arm_distal_in_ee_homogeneous())


def native_R_ee_at_arm_distal_target(T_base_arm_distal: np.ndarray) -> np.ndarray:
    """
    Where URDF ``R_ee`` origin would be if arm-distal reached ``T_base_arm_distal``.

    ``R_ee`` is 5 cm **proximal** to the mount face along **R_ee +Z** (URDF mount).
    """
    T_ad = np.asarray(T_base_arm_distal, dtype=np.float64)
    ee_z = normalize(T_ad[:3, 2])
    T = T_ad.copy()
    T[:3, 3] = T_ad[:3, 3] - ee_z * ARM_DISTAL_IN_EE_TRANSLATION_M[2]
    return T


def arm_distal_alignment_checks(
    T_base_pinch: np.ndarray,
    T_base_arm_distal: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    pre_grasp_offset_m: float,
) -> dict[str, float]:
    approach = approach_dir_from_pose(T_base_pinch)
    finger_open = normalize(T_base_pinch[:3, 0])
    ad_x = normalize(T_base_arm_distal[:3, 0])
    ad_y = normalize(T_base_arm_distal[:3, 1])
    ad_z = normalize(T_base_arm_distal[:3, 2])
    R_ee_x = normalize(T_base_ee[:3, 0])
    R_ee_y = normalize(T_base_ee[:3, 1])
    R_ee_z = normalize(T_base_ee[:3, 2])
    pinch = T_base_pinch[:3, 3]
    ad_origin = T_base_arm_distal[:3, 3]
    ee_origin = T_base_ee[:3, 3]
    expected_ad = pinch - approach * pre_grasp_offset_m
    T_native_ee = native_R_ee_at_arm_distal_target(T_base_arm_distal)
    mount_offset = ee_origin - ad_origin
    return {
        "arm_distal_y_dot_approach": float(np.dot(ad_y, approach)),
        "arm_distal_x_dot_finger_open": float(np.dot(ad_x, finger_open)),
        "arm_distal_z_dot_approach": float(np.dot(ad_z, approach)),
        "R_ee_y_dot_approach": float(np.dot(R_ee_y, approach)),
        "R_ee_x_dot_approach": float(np.dot(R_ee_x, approach)),
        "R_ee_z_dot_approach": float(np.dot(R_ee_z, approach)),
        "arm_distal_origin_err_m": float(np.linalg.norm(ad_origin - expected_ad)),
        "ee_behind_arm_distal_m": float(np.linalg.norm(mount_offset)),
        "mount_offset_along_ee_z_m": float(np.dot(mount_offset, R_ee_z)),
        "ik_ee_matches_native_m": float(np.linalg.norm(T_base_ee[:3, 3] - T_native_ee[:3, 3])),
    }
