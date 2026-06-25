"""Serializable evaluation contracts.

This module intentionally avoids IsaacSim imports. It can be used by policy
adapters, result aggregation scripts, and non-sim test code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


PolicyKind = Literal["open_loop_grasp", "joint_trajectory", "closed_loop_actions"]


def _to_builtin(value: Any) -> Any:
    """Convert numpy-heavy dataclasses to JSON-friendly Python containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


@dataclass
class SceneSpec:
    """Reproducible, serializable description of one evaluation episode."""

    episode_id: str
    obj_id: str
    dataset: str | None = None
    object_scale: float = 1.0
    usd_path: str = ""
    sam3d_mesh_path: str | None = None
    rotated_mesh_path: str | None = None
    object_position_world: list[float] = field(default_factory=list)
    object_orientation_euler_deg: list[float] = field(default_factory=list)
    object_quat_wxyz: list[float] | None = None
    sim_z_yaw_deg: float = 0.0
    obj_xy_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    seed: int = 0
    robot_position: list[float] = field(default_factory=list)
    robot_orientation_euler_deg: list[float] = field(default_factory=list)
    table_position: list[float] = field(default_factory=list)
    table_orientation_euler_deg: list[float] = field(default_factory=list)
    table_scale: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass
class OpenLoopGraspCommand:
    """A grasp pose command in an explicit frame."""

    position: np.ndarray
    rotation: np.ndarray
    gripper_width: float
    frame: str = "object_mesh"
    ee_frame_convention: str = "a2g_grasp_frame"
    name: str = "candidate"
    score: float = 0.0
    approach_type: str = ""
    is_manual: bool = False
    mesh_prerotation_euler: list[float] | None = None
    mesh_prerotation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass
class PolicyOutput:
    """Policy adapter output. First implementation supports open-loop grasps."""

    kind: PolicyKind
    command: OpenLoopGraspCommand | None = None
    actions: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass
class ExecutionResult:
    """Common execution result fields returned by sim executors."""

    success: bool
    failure_stage: str | None = None
    z_delta_m: float | None = None
    initial_object_position_world: list[float] | None = None
    final_object_position_world: list[float] | None = None
    gripper_tips_loc: Any = None
    finger_width_actual: float | None = None
    executed_at_close: dict[str, Any] | None = None
    executed_post_lift: dict[str, Any] | None = None
    planning: dict[str, Any] = field(default_factory=dict)
    video_path: str | None = None
    video_n_frames: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))

