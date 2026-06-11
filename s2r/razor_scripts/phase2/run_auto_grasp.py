#!/usr/bin/env python3
"""
Phase 2 automatic grasp execution from candidates.json.

Loads grasp candidates produced by the 5090 pipeline, retargets them
to robot-frame EE poses, and executes pick-and-lift on each candidate
until one succeeds or all fail.

This is the Phase 2 counterpart of Phase 1's run_grasp.py:
  - Phase 1: human tunes joints per object → run_grasp.py replays them
  - Phase 2: 5090 computes candidates.json → this script executes them

Usage:
    # Real execution
    python demo/phase2/run_auto_grasp.py --session 20260603_143022_chips

    # Dry-run (no hardware, check retarget + IK only)
    python demo/phase2/run_auto_grasp.py --session 20260603_143022_chips --dry-run

    # Specify session directory explicitly
    python demo/phase2/run_auto_grasp.py \
        --session-dir demo/phase2/sessions/20260603_143022_chips

Prerequisites:
    1. demo/phase2/calib/ee_retarget.yaml calibrated (one-time)
    2. demo/right_hand_profile.yaml tuned (one-time via hand_tuner.py)
    3. sessions/<id>/output/status.json exists with success=true
    4. sessions/<id>/output/inference/candidates.json exists
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import GraspObjectConfig, load_hand_profile
from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M
from demo.phase1.grasp_geometry import format_pose
from demo.phase2.retarget import RetargetedGrasp, retarget_session


# ── Session discovery ──────────────────────────────────────

_DEFAULT_SESSIONS_DIR = Path(__file__).parent / "sessions"


def find_session_dir(session_id: str, sessions_dir: Path | None = None) -> Path:
    """Resolve session directory from ID."""
    base = sessions_dir or _DEFAULT_SESSIONS_DIR
    d = base / session_id
    if d.is_dir():
        return d
    # Try glob
    matches = list(base.glob(f"*{session_id}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous session '{session_id}': {[m.name for m in matches]}"
        )
    raise FileNotFoundError(f"Session not found: {session_id} (in {base})")


def validate_session_output(session_dir: Path) -> dict:
    """Check that 5090 pipeline succeeded."""
    status_path = session_dir / "output" / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(
            f"No output/status.json in {session_dir}\n"
            "Did you run process_razor_session.py on 5090 and copy output/ back?"
        )
    with open(status_path) as f:
        status = json.load(f)
    if not status.get("success", False):
        errors = status.get("errors", [])
        raise RuntimeError(
            f"5090 pipeline failed for this session.\n"
            f"Errors: {errors}"
        )

    candidates_path = session_dir / "output" / "inference" / "candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"No candidates.json in {session_dir}/output/inference/")

    return status


# ── Grasp execution ────────────────────────────────────────

def build_grasp_config_from_retarget(
    rg: RetargetedGrasp,
    *,
    table_height: float = DEFAULT_TABLE_HEIGHT_M,
    object_name: str = "auto_grasp",
) -> GraspObjectConfig:
    """Build a GraspObjectConfig from a retargeted grasp for the executor.

    The executor expects grasp_pose as a 4×4 in base frame.
    We set pre_grasp_pose and lift_pose explicitly since they are computed
    from the approach direction (not from FK).
    """
    hand_open, hand_closed = load_hand_profile()

    return GraspObjectConfig(
        object_name=object_name,
        table_height=table_height,
        grasp_pose=rg.T_base_ee.copy(),
        # Arm joints: will be solved by IK, not preset
        right_arm_joint_pos=np.zeros(7),  # placeholder, IK will override
        left_arm_joint_pos=np.zeros(7),   # placeholder
        torso_joint_pos=np.zeros(3),
        head_joint_pos=np.zeros(3),
        hand_open_joint_pos=hand_open,
        hand_closed_joint_pos=hand_closed,
        pre_grasp_pose=rg.T_base_pre.copy(),
        lift_pose=rg.T_base_lift.copy(),
    )


def run_auto_grasp(
    session_dir: Path,
    *,
    dry_run: bool = False,
    calib_path: str | None = None,
    table_height: float = DEFAULT_TABLE_HEIGHT_M,
    max_attempts: int = 5,
    skip_home_at_end: bool = False,
) -> bool:
    """Execute automatic grasp from candidates.json.

    Args:
        session_dir: Path to session directory.
        dry_run: If True, only print retarget results + IK solutions, no hardware.
        calib_path: Override path to ee_retarget.yaml.
        table_height: Table height for collision planning (meters).
        max_attempts: Maximum number of candidates to try.
        skip_home_at_end: Stay at lift pose after success.

    Returns:
        True if any grasp succeeded.
    """
    # ── 1. Validate session ────────────────────────────
    status = validate_session_output(session_dir)
    logger.info(f"Session: {session_dir.name}")
    logger.info(f"Policy:  {status.get('policy', 'unknown')}")

    # ── 2. Retarget candidates ─────────────────────────
    candidates_path = session_dir / "output" / "inference" / "candidates.json"
    result = retarget_session(candidates_path, calib_path=calib_path)

    n = len(result.candidates)
    if n == 0:
        logger.error("No candidates to execute")
        return False

    logger.info(f"Retargeted {n} candidates (trying up to {max_attempts})")

    # Print all candidates
    for rg in result.candidates:
        logger.info(
            f"  [{rg.rank}] {rg.name:>16s}  score={rg.score:.3f}  "
            f"approach_z={rg.approach_base[2]:+.3f}  "
            f"width={rg.gripper_width_m*100:.1f}cm"
        )
        if dry_run:
            logger.info(format_pose(rg.T_base_ee, f"       EE"))
            logger.info(format_pose(rg.T_base_pre, f"       pre"))
            logger.info(format_pose(rg.T_base_lift, f"       lift"))

    if dry_run:
        logger.info("[dry_run] Retarget complete. No hardware execution.")
        return True

    # ── 3. Connect hardware ────────────────────────────
    from demo.phase1.executor import GraspExecutor

    executor = GraspExecutor(dry_run=False)
    executor.setup()

    try:
        # ── 4. Load start config ───────────────────────
        from demo.phase1.config_io import load_config

        start_config = None
        start_path = Path("demo/phase1/configs/start.yaml")
        if start_path.exists():
            start_config = load_config(start_path)
            logger.info(f"Loaded start config: {start_path}")

        # ── 5. Move to start ───────────────────────────
        if start_config is not None:
            executor.apply_hand_profile(start_config)
            executor.apply_demo_head_pose(start_config)
            logger.info("Moving to start pose...")
            # Use _move_to_start_pose_arms_then_open_hand from executor
            # We need a planning_config with table_height
            planning_config = build_grasp_config_from_retarget(
                result.candidates[0],
                table_height=table_height,
                object_name="planning",
            )
            executor._move_to_start_pose_arms_then_open_hand(
                planning_config, start_config
            )

        input("Place object on table, then press Enter to start auto-grasp...")

        # ── 6. Try candidates ──────────────────────────
        success = False
        attempts = min(max_attempts, n)

        for i, rg in enumerate(result.candidates[:attempts]):
            logger.info(f"\n{'='*50}")
            logger.info(f"Attempting candidate [{rg.rank}] {rg.name} "
                        f"(score={rg.score:.3f})")
            logger.info(f"{'='*50}")

            try:
                # Build config for this candidate
                config = build_grasp_config_from_retarget(
                    rg,
                    table_height=table_height,
                    object_name=f"auto_{rg.name}",
                )
                executor.apply_hand_profile(config)
                executor.apply_demo_head_pose(config)

                # Pre-grasp (collision-planned move)
                logger.info("Moving to pre-grasp...")
                executor._move_to_right_ee_pose(
                    config,
                    rg.T_base_pre,
                    config.hand_open_joint_pos,
                    label="pre_grasp",
                    allow_direct_fallback=True,
                )

                # Grasp (move to contact)
                logger.info("Moving to grasp pose...")
                executor._move_to_right_ee_pose(
                    config,
                    rg.T_base_ee,
                    config.hand_open_joint_pos,
                    label="grasp",
                    allow_direct_fallback=True,
                )

                # Close hand
                logger.info("Closing hand (stall-detect)...")
                q_closed = executor._close_right_hand_until_stall(config)

                # Lift
                logger.info("Lifting...")
                executor._move_to_right_ee_pose(
                    config,
                    rg.T_base_lift,
                    q_closed,
                    label="lift",
                    allow_direct_fallback=True,
                )

                # Hold and check
                logger.info(f"Holding at lift for 2s...")
                time.sleep(2.0)

                # Simple success check: ask operator
                user = input("Grasp successful? [y/n]: ").strip().lower()
                if user == "y":
                    logger.info(f"✅ Grasp succeeded with candidate [{rg.rank}] {rg.name}")
                    success = True
                    break
                else:
                    logger.info(f"❌ Grasp failed, preparing next candidate...")
                    # Open hand, move back
                    executor._open_right_grip_stepped(config)
                    time.sleep(1.0)

            except (RuntimeError, AssertionError) as e:
                logger.warning(f"Candidate [{rg.rank}] failed: {e}")
                continue

        if not success:
            logger.warning(f"All {attempts} candidates failed")

        # ── 7. Return to start ─────────────────────────
        if not skip_home_at_end and start_config is not None:
            input("Press Enter to return to start pose...")
            planning_config = build_grasp_config_from_retarget(
                result.candidates[0],
                table_height=table_height,
            )
            executor._open_right_grip_stepped(planning_config)
            start_targets = executor.joint_targets_with_open_grip(
                start_config, planning_config
            )
            executor._move_to_joint_targets(
                planning_config,
                start_targets,
                pose_config=start_config,
                label="return_start",
                allow_direct_fallback=True,
            )

        return success

    finally:
        executor.shutdown()


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: automatic grasp from candidates.json"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=str,
                       help="Session ID (auto-discover in sessions/)")
    group.add_argument("--session-dir", type=str,
                       help="Explicit session directory path")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print retarget results only, no hardware")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to ee_retarget.yaml")
    parser.add_argument("--table-height", type=float,
                        default=DEFAULT_TABLE_HEIGHT_M,
                        help=f"Table height in meters (default: {DEFAULT_TABLE_HEIGHT_M})")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max candidates to try")
    parser.add_argument("--skip-home-at-end", action="store_true",
                        help="Stay at lift after success")
    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = find_session_dir(args.session)

    ok = run_auto_grasp(
        session_dir=session_dir,
        dry_run=args.dry_run,
        calib_path=args.calib,
        table_height=args.table_height,
        max_attempts=args.max_attempts,
        skip_home_at_end=args.skip_home_at_end,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
