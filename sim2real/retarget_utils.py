"""
sim2real/retarget_utils.py

Franka→Dexmate+Sharpa retarget utilities.

Converts a grasp candidate (UCB / A2G format, object-mesh frame) into:
  - Franka EE world pose  (TCP_OFFSET compensation applied)
  - Dexmate R_ee world pose (via T_ee_pinch retarget)

All math mirrors V2AP-demo/demo/phase2/retarget.py and
V2AP-demo/demo/phase2/hand_retarget_geometry.py, re-implemented here
without depending on the V2AP-demo package being installed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ── constants (must match V2AP-demo) ─────────────────────────────────────────
FRANKA_TCP_OFFSET_M   = 0.105   # panda_hand → grasp_point along approach
PRE_GRASP_OFFSET_M    = 0.15    # retreat before grasping
LIFT_HEIGHT_M         = 0.15    # vertical lift after grasp
# Sharpa hand-mount in R_ee: 5 cm along R_ee +Z (from robot_descriptions.py)
SHARPA_MOUNT_Z_M      = 0.05


# ── coordinate helpers ────────────────────────────────────────────────────────

def make_transform(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    T[:3, 3] = pos
    return T


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ── object-mesh → world transform ────────────────────────────────────────────

# Rotation adapter matching curobo_executor.py line 429-437
_R_ADAPT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)


def candidate_to_world(
    grasp_pos_obj: np.ndarray,      # (3,) in object-mesh frame
    grasp_rot_obj: np.ndarray,      # (3,3) rotation in object-mesh frame
    T_world_obj:   np.ndarray,      # (4,4) object pose in world
    object_scale:  float = 1.0,
    mesh_prerotation_euler: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pos_world, rot_world, quat_wxyz_world) for the *pinch point*
    (= Franka TCP location after TCP_OFFSET compensation has been removed).
    """
    pos_scaled = np.asarray(grasp_pos_obj, dtype=np.float64) * object_scale

    if mesh_prerotation_euler and any(abs(float(e)) > 0.5 for e in mesh_prerotation_euler):
        Rp = Rotation.from_euler("xyz", mesh_prerotation_euler, degrees=True).as_matrix()
        T_eff = T_world_obj.copy()
        T_eff[:3, :3] = T_world_obj[:3, :3] @ Rp.T
    else:
        T_eff = T_world_obj

    pos_w = (T_eff @ np.append(pos_scaled, 1.0))[:3]
    rot_w = T_eff[:3, :3] @ np.asarray(grasp_rot_obj, dtype=np.float64) @ _R_ADAPT
    quat_w = rot_to_quat_wxyz(rot_w)
    return pos_w, rot_w, quat_w


# ── Franka target ─────────────────────────────────────────────────────────────

def franka_ee_world(
    grasp_pos_world: np.ndarray,
    grasp_rot_world: np.ndarray,
    table_top_z: float,
    min_z_margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Franka panda_hand position = grasp_point - approach * TCP_OFFSET.
    Returns (pos_w, rot_w, quat_wxyz_w).
    """
    approach = normalize(grasp_rot_world[:, 2])
    pos_w = grasp_pos_world - approach * FRANKA_TCP_OFFSET_M
    min_z = table_top_z + min_z_margin
    if pos_w[2] < min_z:
        pos_w[2] = min_z
    return pos_w, grasp_rot_world, rot_to_quat_wxyz(grasp_rot_world)


# ── Dexmate retarget ──────────────────────────────────────────────────────────

def _default_T_ee_pinch(sharpa_mount_z: float = SHARPA_MOUNT_Z_M) -> np.ndarray:
    """
    Bootstrap T_ee_pinch from geometry when ee_retarget.yaml is Identity
    (i.e. the virtual closed-pinch centre is estimated from the hand mount
    offset + finger reach).

    This is a *geometric approximation*.  Run calibrate_ee_retarget.py on
    the real robot for a precise value; then load that yaml here.

    Approximate: pinch centre ≈ mount_z + 0.10 m along +Z (finger reach),
    with a small lateral offset (thumb toward +X of hand).
    """
    # Position of pinch midpoint in R_ee frame (column-vector convention)
    # These numbers come from the closed-hand FK default in V2AP-demo:
    #   thumb_DP ≈ [0.02, 0.11, 0.10] in hand-base frame
    #   index_DP ≈ [-0.02, 0.10, 0.10] in hand-base frame
    #   midpoint ≈ [0.0, 0.105, 0.10]
    #   hand-base is +5 cm from R_ee along +Z, so in R_ee frame:
    #   midpoint ≈ [0.0, 0.105, 0.15]
    t_pinch_in_ee = np.array([0.0, 0.105, 0.15])

    # Rotation: UCB pinch frame axis alignment relative to R_ee
    # finger_open ≈ +X_ee, y_body ≈ +Y_ee, approach ≈ +Z_ee (hand pushes forward)
    T = np.eye(4)
    T[:3, 3] = t_pinch_in_ee
    return T


def load_T_ee_pinch(ee_retarget_yaml: str | None = None) -> np.ndarray:
    """
    Load T_ee_pinch_closed from a V2AP-demo ee_retarget.yaml, or fall back
    to the geometric approximation.
    """
    if ee_retarget_yaml is not None:
        try:
            import yaml
            with open(ee_retarget_yaml, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            T = np.asarray(raw["T_ee_pinch"], dtype=np.float64)
            if T.shape == (4, 4) and not np.allclose(T, np.eye(4), atol=1e-6):
                return T
        except Exception as exc:
            print(f"[retarget] Could not load {ee_retarget_yaml}: {exc}; using geometric default.")
    return _default_T_ee_pinch()


def dexmate_ee_world(
    grasp_pos_world: np.ndarray,
    grasp_rot_world: np.ndarray,
    T_ee_pinch: np.ndarray,
    robot_base_position: list[float],
    robot_base_yaw_deg: float,
    table_top_z: float,
    min_z_margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retarget grasp_point (world) → Dexmate R_ee pose (world).

    T_base_ee = T_base_pinch @ inv(T_ee_pinch)
    where T_base_pinch has origin at grasp_point and rotation = grasp_rot.

    Returns (pos_w, rot_w, quat_wxyz_w) for R_ee.
    """
    # Build T_base_pinch (world frame)
    T_base_pinch = np.eye(4)
    T_base_pinch[:3, :3] = grasp_rot_world
    T_base_pinch[:3, 3]  = grasp_pos_world

    # Retarget
    T_base_ee = T_base_pinch @ np.linalg.inv(T_ee_pinch)

    pos_w = T_base_ee[:3, 3]
    rot_w = T_base_ee[:3, :3]

    min_z = table_top_z + min_z_margin
    if pos_w[2] < min_z:
        pos_w[2] = min_z

    return pos_w, rot_w, rot_to_quat_wxyz(rot_w)


# ── pre-grasp & lift helpers (shared) ────────────────────────────────────────

def pre_grasp_pos(grasp_pos_world: np.ndarray, grasp_rot_world: np.ndarray) -> np.ndarray:
    """Retreat 15 cm along approach direction."""
    approach = normalize(grasp_rot_world[:, 2])
    return grasp_pos_world - approach * PRE_GRASP_OFFSET_M


def lift_pos(grasp_pos_world: np.ndarray) -> np.ndarray:
    """Lift 15 cm vertically."""
    p = grasp_pos_world.copy()
    p[2] += LIFT_HEIGHT_M
    return p
