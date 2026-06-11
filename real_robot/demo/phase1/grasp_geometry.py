"""Grasp frame geometry aligned with UCB sim/run_grasp_sim.py."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def approach_dir_from_pose(T: np.ndarray) -> np.ndarray:
    """Approach direction = third column of rotation (into object in sim convention)."""
    assert T.shape == (4, 4)
    return normalize(T[:3, 2].astype(np.float64))


def compute_pre_grasp_pose(
    grasp_pose: np.ndarray,
    offset_m: float = 0.15,
    *,
    approach_in_base: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pre-grasp TCP pose: retreat along -approach from grasp pose (same orientation).

    Matches sim/run_grasp_sim.py:
        pre_grasp_pos = pos_world - approach_dir * pre_grasp_offset

    If ``approach_in_base`` is given (e.g. Titan pinch column 2), use it instead of
    ``grasp_pose`` rotation column 2 — needed when R_ee Z ≠ grasp approach.
    """
    grasp_pose = np.asarray(grasp_pose, dtype=np.float64)
    assert grasp_pose.shape == (4, 4)
    T = grasp_pose.copy()
    if approach_in_base is not None:
        approach = normalize(np.asarray(approach_in_base, dtype=np.float64))
    else:
        approach = approach_dir_from_pose(T)
    T[:3, 3] = T[:3, 3] - approach * offset_m
    return T


def compute_lift_pose(
    grasp_pose: np.ndarray,
    lift_height_m: float = 0.15,
) -> np.ndarray:
    """
    Lift pose: grasp pose shifted +Z in world frame (sim LIFT_HEIGHT convention).
    """
    grasp_pose = np.asarray(grasp_pose, dtype=np.float64)
    assert grasp_pose.shape == (4, 4)
    T = grasp_pose.copy()
    T[2, 3] += lift_height_m
    return T


def homogeneous_to_se3(T: np.ndarray):
    """Convert 4x4 homogeneous matrix to pinocchio SE3 (requires pinocchio)."""
    import pinocchio as pin

    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4)
    return pin.SE3(T[:3, :3], T[:3, 3])


def se3_to_homogeneous(se3) -> np.ndarray:
    return se3.homogeneous.copy()


def format_pose(T: np.ndarray, name: str = "pose") -> str:
    T = np.asarray(T, dtype=np.float64)
    t = T[:3, 3]
    approach = approach_dir_from_pose(T)
    lines = [
        f"{name}:",
        f"  translation: [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]",
        f"  approach:    [{approach[0]:+.4f}, {approach[1]:+.4f}, {approach[2]:+.4f}]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Sanity check: 15 cm retreat along approach (numpy only; no pinocchio/scipy).
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    grasp = np.eye(4)
    grasp[:3, 3] = [0.4, -0.1, 0.95]
    grasp[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pre = compute_pre_grasp_pose(grasp)
    lift = compute_lift_pose(grasp)
    print(format_pose(grasp, "grasp"))
    print(format_pose(pre, "pre_grasp"))
    print(format_pose(lift, "lift"))
    retreat = np.linalg.norm(grasp[:3, 3] - pre[:3, 3])
    print(f"retreat distance: {retreat:.4f} m (expect ~0.15)")
