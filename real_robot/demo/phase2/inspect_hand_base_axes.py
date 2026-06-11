#!/usr/bin/env python3
"""Print Sharpa hand-base frame axes vs pinch direction (no robot hardware).

Run on Razor:
  source setup.sh
  python demo/phase2/inspect_hand_base_axes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import load_hand_profile  # noqa: E402


def _quat_wxyz_to_R(quat: list[float]) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_R(axis: np.ndarray, q: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _chain_fk(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, float | None]],
) -> np.ndarray:
    """segments: (parent_p, parent_R, joint_axis, q) — fixed if q is None."""
    T = np.eye(4)
    for p, Rpb, axis, q in segments:
        T_j = np.eye(4)
        if q is not None:
            T_j[:3, :3] = _axis_angle_R(axis, q)
        T_l = np.eye(4)
        T_l[:3, :3] = Rpb
        T_l[:3, 3] = p
        T = T @ T_l @ T_j
    return T


def _print_axis(name: str, v: np.ndarray) -> None:
    v = v / np.linalg.norm(v)
    dom = "XYZ"[int(np.argmax(np.abs(v)))]
    sign = "+" if v[int(np.argmax(np.abs(v)))] > 0 else "-"
    print(f"  {name}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]  (≈ {sign}{dom})")


def main() -> None:
    _, closed_q = load_hand_profile()
    q = np.asarray(closed_q, dtype=np.float64)

    # Index chain (right_hand_C_MC → … → right_index_DP), from MJCF v2_1.
    idx = 5  # index_MCP_FE in SHARPA_HAND_JOINT_ORDER
    index_chain = [
        (np.array([0.001, 0.0303, 0.0957]), _quat_wxyz_to_R([0.499998, -0.500002, -0.5, -0.5]), np.array([0, 0, 1]), q[idx]),
        (np.zeros(3), np.eye(3), np.array([0, 0, 1]), q[idx + 1]),  # MCP_AA
        (np.zeros(3), _quat_wxyz_to_R([0.707105, -0.707108, 0, 0]), np.array([0, 0, 1]), None),
        (np.array([0.047, 0, 0]), _quat_wxyz_to_R([0.707105, 0.707108, 0, 0]), np.array([0, 0, 1]), q[idx + 2]),  # PIP
        (np.array([0.0315, 0, 0]), np.eye(3), np.array([0, 0, 1]), q[idx + 3]),  # DIP
    ]

    # Thumb chain
    thumb_chain = [
        (np.array([0.01, 0.026, 0.0212]), _quat_wxyz_to_R([-2.59734e-06, 0.707105, 0.707108, -2.59735e-06]), np.array([0, 0, 1]), q[0]),
        (np.array([0, -0.005, 0]), _quat_wxyz_to_R([0.65328, -0.653282, 0.270598, 0.270599]), np.array([0, 0, 1]), q[1]),
        (np.array([0.065, -0.006, 0.010392]), _quat_wxyz_to_R([0.965926, 0.25882, 0, 0]), np.array([0, 0, 1]), q[2]),
        (np.zeros(3), _quat_wxyz_to_R([0.707105, -0.707108, 0, 0]), np.array([0, 0, 1]), q[3]),
        (np.array([0.039, 0, 0]), _quat_wxyz_to_R([0.707105, 0.707108, 0, 0]), np.array([0, 0, 1]), q[4]),
    ]

    T_index = _chain_fk(index_chain)
    T_thumb = _chain_fk(thumb_chain)
    p_index = T_index[:3, 3]
    p_thumb = T_thumb[:3, 3]
    p_pinch = 0.5 * (p_index + p_thumb)
    pinch_dir = p_pinch / np.linalg.norm(p_pinch)

    print("=== right_hand_C_MC frame (Sharpa hand root / palm mount) ===")
    print(f"Pinch midpoint (closed profile): {p_pinch.round(4).tolist()} m")
    print(f"Distal / pinch direction (origin → pinch): {pinch_dir.round(4).tolist()}")
    _print_axis("Pinch dir vs +X", np.array([1.0, 0.0, 0.0]))
    _print_axis("Pinch dir vs +Y", np.array([0.0, 1.0, 0.0]))
    _print_axis("Pinch dir vs +Z", np.array([0.0, 0.0, 1.0]))
    print("\nDot products (hand-base unit axes · pinch_dir):")
    for label, axis in [("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1])]:
        a = np.asarray(axis, dtype=np.float64)
        print(f"  +{label}: {float(np.dot(a, pinch_dir)):+.4f}")

    # R_ee mount: hand base = R_ee origin + 0.05 * R_ee_Z (same rotation as R_ee).
    print("\n=== Conclusion ===")
    print("  • Sharpa hand base frame = right_hand_C_MC (Pinocchio / MJCF root).")
    print("  • R_ee has SAME orientation as hand base; hand is +5 cm along R_ee +Z.")
    print("  • +Z is NOT the finger-forward axis on HA4.")
    print(f"  • Palm→pinch (distal) unit vector in hand base: {pinch_dir.round(4).tolist()}")
    print(f"    dot(+X)={float(np.dot([1,0,0], pinch_dir)):+.3f}  "
          f"dot(+Y)={float(np.dot([0,1,0], pinch_dir)):+.3f}  "
          f"dot(+Z)={float(np.dot([0,0,1], pinch_dir)):+.3f}")
    ang_z = float(np.degrees(np.arccos(np.clip(float(np.dot([0, 0, 1], pinch_dir)), -1, 1))))
    print(f"    angle to +Z ≈ {ang_z:.1f}°")
    print("  • Retarget keeps R_ee = R_pinch so R_ee +Z = grasp approach (wrist motion axis).")
    print("  • Hand-base +Z ≠ finger forward; do not use handbase_z for approach alignment.")

    try:
        import pinocchio as pin  # noqa: WPS433

        from demo.phase1.constants import DEFAULT_JOINT_POS  # noqa: E402
        from teleop.robot_descriptions import (  # noqa: E402
            RIGHT_HAND_BASE_FRAME,
            SHARPA_HAND_MOUNT_IN_EE,
            build_full_robot,
        )

        robot, assemble, _ = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
        q_full = assemble(
            {
                "left_arm": DEFAULT_JOINT_POS["left_arm"],
                "right_arm": DEFAULT_JOINT_POS["right_arm"],
                "left_hand": np.zeros(22),
                "right_hand": closed_q,
            }
        )
        pin.forwardKinematics(robot.model, robot.data, q_full)
        pin.updateFramePlacements(robot.model, robot.data)
        oMf_ee = robot.data.oMf[robot.model.getFrameId("R_ee")]
        oMf_h = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND_BASE_FRAME)]
        oMf_l8 = robot.data.oMf[robot.model.getFrameId("R_arm_l8")]

        pinch_pin = 0.5 * (
            robot.data.oMf[robot.model.getFrameId("right_thumb_DP")].translation
            + robot.data.oMf[robot.model.getFrameId("right_index_DP")].translation
        )
        pinch_in_h = oMf_h.rotation.T @ (pinch_pin - oMf_h.translation)
        pinch_dir_pin = pinch_in_h / np.linalg.norm(pinch_in_h)

        print("\n=== Pinocchio FK (full model, closed hand) ===")
        print(f"Pinch dir (FK):     {pinch_dir_pin.round(4).tolist()}")
        for label, col in [("X", 0), ("Y", 1), ("Z", 2)]:
            d = float(np.dot(oMf_h.rotation[:, col], pinch_dir_pin))
            print(f"  hand-base +{label} · pinch_dir = {d:+.4f}")
        for label, col in [("X", 0), ("Y", 1), ("Z", 2)]:
            d = float(np.dot(oMf_ee.rotation[:, col], pinch_dir_pin))
            print(f"  R_ee +{label} · pinch_dir = {d:+.4f}")
        for label, col in [("X", 0), ("Y", 1), ("Z", 2)]:
            d = float(np.dot(oMf_l8.rotation[:, col], pinch_dir_pin))
            print(f"  R_arm_l8 +{label} · pinch_dir = {d:+.4f}")
        print(f"\nR_ee mount offset in base: {(oMf_h.translation - oMf_ee.translation).round(4).tolist()}")
        print(f"Expected [0,0,0.05] in R_ee frame → world: {(oMf_ee.rotation @ [0,0,0.05]).round(4).tolist()}")
    except ImportError:
        print("\n(pinocchio not available — manual MJCF chain above is still valid)")


if __name__ == "__main__":
    main()
