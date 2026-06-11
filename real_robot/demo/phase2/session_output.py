"""Load Phase 2 session output/ + build Phase 1 grasp config from Titan candidates."""

from __future__ import annotations

import json
from pathlib import Path

from demo.phase1.config_io import GraspObjectConfig, grasp_config_path, load_grasp_config
from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M, PHASE1_CONFIG_DIR
from demo.phase2.retarget import TitanGraspPoses, TitanSessionOutput, load_titan_output
from demo.phase2.table_height import load_table_height_m  # noqa: F401 — re-export


def session_dir_for_id(session_id: str, sessions_root: Path) -> Path:
    return Path(sessions_root) / session_id


def load_capture_pose_source(session_dir: Path) -> str | None:
    session_path = session_dir / "input" / "session.json"
    if not session_path.is_file():
        return None
    with session_path.open(encoding="utf-8") as f:
        session = json.load(f)
    return session.get("capture", {}).get("pose_source")


def load_start_config_for_session(
    session_dir: Path,
    *,
    config_dir: Path = PHASE1_CONFIG_DIR,
) -> GraspObjectConfig | None:
    """Load Phase 1 start.yaml referenced by capture ``pose_source``."""
    pose_source = load_capture_pose_source(session_dir)
    if not pose_source:
        return None
    # e.g. phase1/configs/start.yaml → object name "start"
    stem = Path(pose_source).stem
    path = grasp_config_path(stem, config_dir)
    if not path.is_file():
        return None
    return load_grasp_config(path)


def grasp_config_from_titan(
    poses: TitanGraspPoses,
    *,
    start_config: GraspObjectConfig | None,
    object_slug: str,
    table_height_m: float = DEFAULT_TABLE_HEIGHT_M,
    open_pinch_forward_offset_m: float = 0.015,
    open_grip_ik_palm_soft: bool = False,
) -> GraspObjectConfig:
    """Merge Titan EE poses with capture-time arm/head/torso from start.yaml."""
    if start_config is None:
        raise ValueError(
            "start_config required (capture session must reference phase1/configs/start.yaml)"
        )
    return GraspObjectConfig(
        object_name=object_slug,
        table_height=table_height_m,
        grasp_pose=poses.grasp_pose.copy(),
        pre_grasp_pose=poses.pre_grasp_pose.copy(),
        lift_pose=poses.lift_pose.copy(),
        left_arm_joint_pos=start_config.left_arm_joint_pos.copy(),
        right_arm_joint_pos=start_config.right_arm_joint_pos.copy(),
        torso_joint_pos=start_config.torso_joint_pos.copy(),
        head_joint_pos=start_config.head_joint_pos.copy(),
        left_hand_joint_pos=start_config.left_hand_joint_pos.copy(),
        pre_grasp_offset_m=float(start_config.pre_grasp_offset_m),
        lift_height_m=float(start_config.lift_height_m),
        hold_time_s=float(start_config.hold_time_s),
        notes=(
            f"Titan auto-grasp rank={poses.rank} ({poses.name}), "
            f"score={poses.score:.1f}"
        ),
        titan_T_base_pinch=poses.T_base_pinch.copy(),
        open_pinch_forward_offset_m=float(open_pinch_forward_offset_m),
        open_grip_ik_palm_soft=bool(open_grip_ik_palm_soft),
    )

