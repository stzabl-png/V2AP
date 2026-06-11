"""Load and save Phase 1 grasp YAML configs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from demo.constants import DEFAULT_HAND_PROFILE_PATH
from demo.phase1.constants import (
    DEFAULT_HOLD_TIME_S,
    DEFAULT_TABLE_HEIGHT_M,
    HEAD_DEFAULT_JOINT_POS,
    LEFT_ARM_DEFAULT_JOINT_POS,
    LIFT_HEIGHT_M,
    PHASE1_CONFIG_DIR,
    PRE_GRASP_OFFSET_M,
    RIGHT_ARM_DEFAULT_JOINT_POS,
    TORSO_DEFAULT_JOINT_POS,
)
from demo.phase1.grasp_geometry import compute_lift_pose, compute_pre_grasp_pose
from demo.virtual_gripper import (
    HAND_PINCH_CLOSED,
    HAND_PINCH_OPEN,
    left_hand_neutral,
    validate_hand_q,
)


def grasp_config_path(object_name: str, config_dir: Path | None = None) -> Path:
    """Resolve ``<config_dir>/<object_name>.yaml`` (strips optional .yaml suffix)."""
    stem = object_name.strip()
    if stem.endswith((".yaml", ".yml")):
        stem = Path(stem).stem
    return (config_dir or PHASE1_CONFIG_DIR) / f"{stem}.yaml"


def _as_vec(data: list | None, default: np.ndarray) -> np.ndarray:
    if data is None:
        return default.copy()
    return np.asarray(data, dtype=np.float64)


def _resolve_hand_profile_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("V2AP_HAND_PROFILE")
    if env:
        return Path(env)
    return DEFAULT_HAND_PROFILE_PATH


def load_hand_profile(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load shared right-hand open/closed joint vectors (same for all objects)."""
    profile_path = _resolve_hand_profile_path(path)
    if not profile_path.is_file():
        return HAND_PINCH_OPEN.copy(), HAND_PINCH_CLOSED.copy()

    with profile_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Hand profile must be a YAML mapping: {profile_path}")

    open_q = validate_hand_q(
        _as_vec(raw.get("hand_open_joint_pos"), HAND_PINCH_OPEN),
        "hand_open_joint_pos",
    )
    closed_q = validate_hand_q(
        _as_vec(raw.get("hand_closed_joint_pos"), HAND_PINCH_CLOSED),
        "hand_closed_joint_pos",
    )
    return open_q.copy(), closed_q.copy()


def save_hand_profile(
    hand_open: np.ndarray,
    hand_closed: np.ndarray,
    path: str | Path | None = None,
    notes: str = "",
) -> Path:
    profile_path = _resolve_hand_profile_path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_name": "right_hand_pinch",
        "hand_open_joint_pos": validate_hand_q(hand_open, "hand_open_joint_pos").tolist(),
        "hand_closed_joint_pos": validate_hand_q(hand_closed, "hand_closed_joint_pos").tolist(),
        "notes": notes,
    }
    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return profile_path


def _as_pose4(data: list | None) -> np.ndarray | None:
    if data is None:
        return None
    T = np.asarray(data, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"grasp_pose must be 4x4, got shape {T.shape}")
    return T


@dataclass
class GraspObjectConfig:
    object_name: str
    table_height: float = DEFAULT_TABLE_HEIGHT_M
    grasp_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    pre_grasp_pose: np.ndarray | None = None
    lift_pose: np.ndarray | None = None
    left_arm_joint_pos: np.ndarray = field(default_factory=lambda: LEFT_ARM_DEFAULT_JOINT_POS.copy())
    right_arm_joint_pos: np.ndarray = field(default_factory=lambda: RIGHT_ARM_DEFAULT_JOINT_POS.copy())
    torso_joint_pos: np.ndarray = field(default_factory=lambda: TORSO_DEFAULT_JOINT_POS.copy())
    head_joint_pos: np.ndarray = field(default_factory=lambda: HEAD_DEFAULT_JOINT_POS.copy())
    left_hand_joint_pos: np.ndarray = field(default_factory=left_hand_neutral)
    hand_open_joint_pos: np.ndarray = field(default_factory=lambda: HAND_PINCH_OPEN.copy())
    hand_closed_joint_pos: np.ndarray = field(default_factory=lambda: HAND_PINCH_CLOSED.copy())
    pre_grasp_offset_m: float = PRE_GRASP_OFFSET_M
    lift_height_m: float = LIFT_HEIGHT_M
    hold_time_s: float = DEFAULT_HOLD_TIME_S
    notes: str = ""
    # Phase 2 Titan: open-grip retarget IK (see open_grip_retarget_geometry.py).
    titan_T_base_pinch: np.ndarray | None = None
    open_pinch_forward_offset_m: float = 0.015
    open_grip_ik_palm_soft: bool = False

    def resolved_pre_grasp_pose(self) -> np.ndarray:
        if self.pre_grasp_pose is not None:
            return self.pre_grasp_pose.copy()
        return compute_pre_grasp_pose(self.grasp_pose, self.pre_grasp_offset_m)

    def resolved_lift_pose(self) -> np.ndarray:
        if self.lift_pose is not None:
            return self.lift_pose.copy()
        return compute_lift_pose(self.grasp_pose, self.lift_height_m)

    def to_dict(self, *, embed_hand: bool = False) -> dict[str, Any]:
        return {
            "object_name": self.object_name,
            "table_height": float(self.table_height),
            "grasp_pose": self.grasp_pose.tolist(),
            "pre_grasp_pose": None if self.pre_grasp_pose is None else self.pre_grasp_pose.tolist(),
            "lift_pose": None if self.lift_pose is None else self.lift_pose.tolist(),
            "left_arm_joint_pos": self.left_arm_joint_pos.tolist(),
            "right_arm_joint_pos": self.right_arm_joint_pos.tolist(),
            "torso_joint_pos": self.torso_joint_pos.tolist(),
            "head_joint_pos": self.head_joint_pos.tolist(),
            "left_hand_joint_pos": self.left_hand_joint_pos.tolist(),
            "hand_open_joint_pos": (
                None
                if not embed_hand
                else self.hand_open_joint_pos.tolist()
            ),
            "hand_closed_joint_pos": (
                None
                if not embed_hand
                else self.hand_closed_joint_pos.tolist()
            ),
            "pre_grasp_offset_m": float(self.pre_grasp_offset_m),
            "lift_height_m": float(self.lift_height_m),
            "hold_time_s": float(self.hold_time_s),
            "notes": self.notes,
        }


def load_grasp_config(path: str | Path) -> GraspObjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    grasp_pose = _as_pose4(raw.get("grasp_pose"))
    if grasp_pose is None:
        raise ValueError(f"grasp_pose is required in {path}")

    profile_open, profile_closed = load_hand_profile()
    hand_open_raw = raw.get("hand_open_joint_pos")
    hand_closed_raw = raw.get("hand_closed_joint_pos")

    return GraspObjectConfig(
        object_name=str(raw.get("object_name", path.stem)),
        table_height=float(raw.get("table_height", DEFAULT_TABLE_HEIGHT_M)),
        grasp_pose=grasp_pose,
        pre_grasp_pose=_as_pose4(raw.get("pre_grasp_pose")),
        lift_pose=_as_pose4(raw.get("lift_pose")),
        left_arm_joint_pos=_as_vec(raw.get("left_arm_joint_pos"), LEFT_ARM_DEFAULT_JOINT_POS),
        right_arm_joint_pos=_as_vec(raw.get("right_arm_joint_pos"), RIGHT_ARM_DEFAULT_JOINT_POS),
        torso_joint_pos=_as_vec(raw.get("torso_joint_pos"), TORSO_DEFAULT_JOINT_POS),
        head_joint_pos=_as_vec(raw.get("head_joint_pos"), HEAD_DEFAULT_JOINT_POS),
        left_hand_joint_pos=validate_hand_q(
            _as_vec(raw.get("left_hand_joint_pos"), left_hand_neutral()), "left_hand_joint_pos"
        ),
        hand_open_joint_pos=validate_hand_q(
            _as_vec(hand_open_raw, profile_open), "hand_open_joint_pos"
        ),
        hand_closed_joint_pos=validate_hand_q(
            _as_vec(hand_closed_raw, profile_closed), "hand_closed_joint_pos"
        ),
        pre_grasp_offset_m=float(raw.get("pre_grasp_offset_m", PRE_GRASP_OFFSET_M)),
        lift_height_m=float(raw.get("lift_height_m", LIFT_HEIGHT_M)),
        hold_time_s=float(float(raw.get("hold_time_s", DEFAULT_HOLD_TIME_S))),
        notes=str(raw.get("notes", "")),
    )


def save_grasp_config(config: GraspObjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, default_flow_style=False)
