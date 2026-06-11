#!/usr/bin/env python3
"""
Debug showcase: rotate hand so **R_ee +Z (= right_hand_C_MC +Z)** aligns with robot base axes.

No Titan retarget. Same ``R_ee`` position (from demo start FK); only orientation changes:

  1. hand +Z ∥ base +Z  (world up)
  2. hand +Z ∥ base +X
  3. hand +Z ∥ base +Y

Usage (Razor):
  source setup.sh
  python demo/phase2/debug_hand_z_axis_showcase.py
  python demo/phase2/debug_hand_z_axis_showcase.py --no-prompts --dwell 5

Pose math only:
  python demo/phase2/debug_hand_z_axis_showcase.py --pose-only

Legacy Titan pre-grasp debug (optional, separate script):
  python demo/phase2/debug_retarget_arm_distal_pregrasp.py --session-id ... --titan-pregrasp
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import GraspObjectConfig, load_grasp_config, grasp_config_path  # noqa: E402
from demo.phase1.constants import PHASE1_CONFIG_DIR  # noqa: E402
from demo.phase1.grasp_geometry import format_pose, normalize, se3_to_homogeneous  # noqa: E402
from demo.phase2.arm_distal_geometry import (  # noqa: E402
    BASE_AXIS_SHOWCASE,
    RIGHT_EE_FRAME,
    T_base_ee_hand_z_along,
)

RIGHT_HAND_BASE_FRAME = "right_hand_C_MC"


@dataclass
class HandZShowcasePose:
    label: str
    target_axis: np.ndarray
    T_base_ee: np.ndarray


def build_showcase_poses(position_in_base: np.ndarray) -> list[HandZShowcasePose]:
    pos = np.asarray(position_in_base, dtype=np.float64)
    x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    out: list[HandZShowcasePose] = []
    for label, axis in BASE_AXIS_SHOWCASE:
        T = T_base_ee_hand_z_along(axis, pos, x_hint_in_base=x_hint)
        out.append(HandZShowcasePose(label=label, target_axis=axis.copy(), T_base_ee=T))
    return out


def log_showcase_pose(pose: HandZShowcasePose) -> None:
    hand_z = normalize(pose.T_base_ee[:3, 2])
    dot = float(np.dot(hand_z, normalize(pose.target_axis)))
    logger.info(
        f"Showcase [{pose.label}]: align {RIGHT_HAND_BASE_FRAME} +Z (= {RIGHT_EE_FRAME} +Z) "
        f"with robot base {pose.label}"
    )
    logger.info(format_pose(pose.T_base_ee, f"T_base_{RIGHT_EE_FRAME}"))
    logger.info(f"  {RIGHT_EE_FRAME} +Z · target = {dot:.6f} (expect +1.0)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug: hand +Z aligned to base +Z, +X, +Y in sequence (no Titan)"
    )
    p.add_argument(
        "--start-config",
        type=str,
        default="start",
        help="Phase1 config name for start pose / joint seeds (default: start)",
    )
    p.add_argument("--config-dir", type=Path, default=PHASE1_CONFIG_DIR)
    p.add_argument("--dry-run", action="store_true", help="IK check only")
    p.add_argument("--pose-only", action="store_true", help="Print poses; no robot")
    p.add_argument("--no-prompts", action="store_true", help="Auto-advance (use --dwell)")
    p.add_argument(
        "--dwell",
        type=float,
        default=4.0,
        help="Seconds to hold each showcase pose when --no-prompts (default 4)",
    )
    p.add_argument("--skip-start", action="store_true", help="Skip move to demo start")
    return p.parse_args()


def _load_start_config(name: str, config_dir: Path) -> GraspObjectConfig:
    path = grasp_config_path(name, config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Start config not found: {path}")
    return load_grasp_config(path)


def _shell_config(start: GraspObjectConfig, T_ee: np.ndarray) -> GraspObjectConfig:
    return GraspObjectConfig(
        object_name="hand_z_showcase",
        table_height=float(start.table_height),
        grasp_pose=T_ee.copy(),
        pre_grasp_pose=T_ee.copy(),
        lift_pose=T_ee.copy(),
        left_arm_joint_pos=start.left_arm_joint_pos.copy(),
        right_arm_joint_pos=start.right_arm_joint_pos.copy(),
        torso_joint_pos=start.torso_joint_pos.copy(),
        head_joint_pos=start.head_joint_pos.copy(),
        left_hand_joint_pos=start.left_hand_joint_pos.copy(),
        pre_grasp_offset_m=float(start.pre_grasp_offset_m),
        lift_height_m=float(start.lift_height_m),
        hold_time_s=float(start.hold_time_s),
        notes="hand +Z axis showcase",
    )


def _ee_position_from_start(executor, start: GraspObjectConfig) -> np.ndarray:
    fk = executor.pink_ik.fk(
        frames=[RIGHT_EE_FRAME],
        joint_pos_by_component={
            "left_arm": start.left_arm_joint_pos,
            "right_arm": start.right_arm_joint_pos,
        },
    )
    return se3_to_homogeneous(fk[RIGHT_EE_FRAME])[:3, 3].copy()


def _ee_position_after_start_move(executor) -> np.ndarray:
    left_q, right_q = executor._read_arm_joint_pos()
    fk = executor.pink_ik.fk(
        frames=[RIGHT_EE_FRAME],
        joint_pos_by_component={"left_arm": left_q, "right_arm": right_q},
    )
    return se3_to_homogeneous(fk[RIGHT_EE_FRAME])[:3, 3].copy()


def _log_achieved_hand_z(executor, pose: HandZShowcasePose) -> None:
    if executor.dry_run or executor.pink_ik is None:
        return
    left_q, right_q = executor._read_arm_joint_pos()
    fk = executor.pink_ik.fk(
        frames=[RIGHT_EE_FRAME],
        joint_pos_by_component={"left_arm": left_q, "right_arm": right_q},
    )
    T = se3_to_homogeneous(fk[RIGHT_EE_FRAME])
    achieved_z = normalize(T[:3, 2])
    target = normalize(pose.target_axis)
    logger.info(
        f"  Achieved {RIGHT_EE_FRAME} +Z · {pose.label} = "
        f"{float(np.dot(achieved_z, target)):+.4f} "
        f"(1.0 = aligned; <<1 if IK did not track orientation)"
    )


def _ik_feasible(executor, T_ee: np.ndarray, seed: GraspObjectConfig, *, label: str) -> bool:
    try:
        executor._solve_ik_for_right_ee(
            T_ee,
            seed.left_arm_joint_pos,
            seed.right_arm_joint_pos,
        )
        return True
    except Exception as exc:
        logger.info(f"IK infeasible ({label}): {exc}")
        return False


def _move_to_ee(
    executor,
    config: GraspObjectConfig,
    T_ee: np.ndarray,
    *,
    start: GraspObjectConfig,
    label: str,
    no_prompts: bool,
    dwell: float,
    step_index: int,
    total_steps: int,
) -> None:
    from demo.phase1.planned_motion import BackgroundPlan, wait_enter_with_prefetch  # noqa: WPS433

    open_hand = config.hand_open_joint_pos
    left_q, right_q = executor._read_arm_joint_pos()
    ik_seed = _shell_config(
        start,
        T_ee,
    )
    ik_seed.left_arm_joint_pos = left_q
    ik_seed.right_arm_joint_pos = right_q

    prefetch = BackgroundPlan(
        label,
        lambda: executor._plan_move_to_right_ee(
            config,
            T_ee,
            open_hand,
            ik_seed_config=ik_seed,
            label=label,
            right_arm_collision_boxes=None,
        ),
    )
    prompt = (
        f"[{step_index}/{total_steps}] Press Enter to move: "
        f"hand +Z aligned with robot {label}..."
    )
    if no_prompts:
        logger.info(f"[{step_index}/{total_steps}] Moving to {label}...")
        prefetch.start()
        try:
            planned = prefetch.wait()
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Plan failed ({exc}); direct IK fallback")
            planned = None
    else:
        try:
            planned = wait_enter_with_prefetch(prompt, prefetch)
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Plan failed ({exc}); direct IK fallback")
            planned = None

    executor._move_to_right_ee_pose(
        config,
        T_ee,
        open_hand,
        planned=planned,
        ik_seed_config=ik_seed,
        right_arm_collision_boxes=None,
        label=label,
        allow_direct_fallback=True,
    )


def _run_showcase(args: argparse.Namespace) -> None:
    start = _load_start_config(args.start_config, args.config_dir)

    if args.pose_only:
        pos = np.asarray(start.grasp_pose, dtype=np.float64)[:3, 3].copy()
        logger.info(
            f"Pose-only: {RIGHT_EE_FRAME} origin from "
            f"phase1/configs/{args.start_config}.yaml grasp_pose "
            f"(xyz={pos.round(3).tolist()})"
        )
        for p in build_showcase_poses(pos):
            log_showcase_pose(p)
        logger.info(
            "On robot: open hand at start, same wrist position; only hand +Z rotation changes."
        )
        return

    from demo.phase1.executor import GraspExecutor  # noqa: WPS433

    executor = GraspExecutor(dry_run=args.dry_run)
    try:
        executor.setup()
        executor.apply_hand_profile(start)
        executor.apply_demo_head_pose(start)

        if args.skip_start:
            if args.dry_run:
                pos = _ee_position_from_start(executor, start)
            else:
                pos = _ee_position_after_start_move(executor)
            logger.info("Skipping move to start; using current/s seed R_ee position.")
        else:
            if not args.dry_run:
                config0 = _shell_config(start, np.eye(4))
                executor._move_to_start_pose_arms_then_open_hand(config0, start)
                pos = _ee_position_after_start_move(executor)
            else:
                pos = _ee_position_from_start(executor, start)
            logger.info(
                f"Fixed {RIGHT_EE_FRAME} position from demo start "
                f"(xyz={pos.round(3).tolist()}); rotating hand +Z only."
            )

        poses = build_showcase_poses(pos)
        for p in poses:
            log_showcase_pose(p)

        if args.dry_run:
            for p in poses:
                if not _ik_feasible(executor, p.T_base_ee, start, label=p.label):
                    raise SystemExit(f"IK failed for showcase pose {p.label}")
            logger.info("[dry_run] All showcase orientations IK ok.")
            return

        for p in poses:
            if not _ik_feasible(executor, p.T_base_ee, start, label=p.label):
                raise SystemExit(f"IK failed for showcase pose {p.label}")

        config = _shell_config(start, poses[0].T_base_ee)
        total = len(poses)
        for i, p in enumerate(poses, start=1):
            logger.info(
                f"=== Showcase {i}/{total}: {RIGHT_HAND_BASE_FRAME} +Z → {p.label} ==="
            )
            _move_to_ee(
                executor,
                config,
                p.T_base_ee,
                start=start,
                label=f"hand_z_{p.label}",
                no_prompts=args.no_prompts,
                dwell=args.dwell,
                step_index=i,
                total_steps=total,
            )
            _log_achieved_hand_z(executor, p)
            logger.info(
                f"Holding {p.label}. Look along hand palm normal (+Z): "
                f"should point with robot {p.label}."
            )
            if args.no_prompts:
                time.sleep(float(args.dwell))
            elif i < total:
                input(f"At {p.label}. Press Enter for next axis... ")
            else:
                input("Showcase complete. Press Enter to shutdown... ")
    finally:
        executor.shutdown()


def main() -> None:
    args = _parse_args()
    _run_showcase(args)


if __name__ == "__main__":
    main()
