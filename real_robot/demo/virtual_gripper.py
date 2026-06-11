"""Virtual parallel-jaw gripper using Sharpa HA4 thumb + index.

Hand pose convention (right hand, 22-DOF, radians):
  - Middle / ring / pinky: straight (PIP/DIP = 0), MCP abduction splayed for clearance;
    pinky CMC at URDF upper limit. Same for open and closed.
  - Thumb + index: straight phalanges (IP/PIP/DIP = 0); open = spread at CMC/MCP roots;
    closed = roots retracted toward pinch (tune on hardware if contact is off).

Joint order matches teleop/robot_descriptions.SHARPA_HAND_JOINT_ORDER and Sharpa SDK.
"""

from __future__ import annotations

import numpy as np

from demo.constants import HAND_JOINT_COUNT

# fmt: off
HAND_JOINT_NAMES: tuple[str, ...] = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)

# URDF limits: manus/retargeting/urdf/right_sharpa_ha4/right_sharpa_ha4_v2_1.xml
_JOINT_LOW = np.array(
    [
        -0.1745, -0.3491, -0.5236, -0.3491, 0.0,
        -0.8727, -0.3491, 0.0, 0.0,
        -0.8727, -0.3491, 0.0, 0.0,
        -0.8727, -0.3491, 0.0, 0.0,
        0.0, -0.8727, -0.3491, 0.0, 0.0,
    ],
    dtype=np.float64,
)
_JOINT_HIGH = np.array(
    [
        1.9199, 0.3491, 1.3963, 0.3491, 1.3963,
        1.5708, 0.3491, 1.7453, 1.3963,
        1.5708, 0.3491, 1.7453, 1.3963,
        1.5708, 0.3491, 1.7453, 1.3963,
        0.2618, 1.5708, 0.3491, 1.7453, 1.3963,
    ],
    dtype=np.float64,
)

# Middle / ring / pinky: extended chains + max root splay (clearance).
_CLEARANCE_MIDDLE_RING_PINKY = np.array(
    [
        # middle
        0.0, 0.3491, 0.0, 0.0,
        # ring (AA opposite middle to spread)
        0.0, -0.3491, 0.0, 0.0,
        # pinky (CMC at upper limit)
        0.2618, 0.0, 0.3491, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Thumb + index roots only (indices 0-8); phalanges stay straight.
_THUMB_INDEX_OPEN = np.array(
    [
        1.5, 0.30, 0.0, 0.0, 0.0,   # thumb: CMC spread, straight
        0.0, -0.3491, 0.0, 0.0,    # index: MCP AA toward thumb side
    ],
    dtype=np.float64,
)

_THUMB_INDEX_CLOSED = np.array(
    [
        0.5, 0.05, 0.0, 0.0, 0.0,  # thumb: roots retracted
        0.35, 0.0, 0.0, 0.0,       # index: slight MCP flex, AA neutral
    ],
    dtype=np.float64,
)
# fmt: on

HAND_NEUTRAL = np.zeros(HAND_JOINT_COUNT, dtype=np.float64)


def _clip_to_limits(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64), _JOINT_LOW, _JOINT_HIGH)


def clip_hand_q(q: np.ndarray) -> np.ndarray:
    """Clip 22-DOF hand joint vector to URDF limits."""
    return _clip_to_limits(validate_hand_q(q))


def clearance_joint_pos() -> np.ndarray:
    """Middle, ring, pinky: straight + splayed (indices 9-21)."""
    return _clip_to_limits(_CLEARANCE_MIDDLE_RING_PINKY.copy())


def build_hand_open_joint_pos() -> np.ndarray:
    """Open virtual gripper: thumb/index roots spread; clearance fingers unchanged."""
    q = np.zeros(HAND_JOINT_COUNT, dtype=np.float64)
    q[0:9] = _THUMB_INDEX_OPEN
    q[9:22] = _CLEARANCE_MIDDLE_RING_PINKY
    return _clip_to_limits(q)


def build_hand_closed_joint_pos() -> np.ndarray:
    """Closed virtual gripper: thumb/index roots pinched; clearance fingers unchanged."""
    q = np.zeros(HAND_JOINT_COUNT, dtype=np.float64)
    q[0:9] = _THUMB_INDEX_CLOSED
    q[9:22] = _CLEARANCE_MIDDLE_RING_PINKY
    return _clip_to_limits(q)


HAND_PINCH_OPEN = build_hand_open_joint_pos()
HAND_PINCH_CLOSED = build_hand_closed_joint_pos()


def validate_hand_q(q: np.ndarray, name: str = "hand_q") -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (HAND_JOINT_COUNT,):
        raise ValueError(f"{name} must have shape ({HAND_JOINT_COUNT},), got {q.shape}")
    return q


def left_hand_neutral() -> np.ndarray:
    return HAND_NEUTRAL.copy()
