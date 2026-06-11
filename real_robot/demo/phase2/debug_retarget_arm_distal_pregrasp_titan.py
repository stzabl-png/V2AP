#!/usr/bin/env python3
"""
Debug retarget: place **R_arm_l8 distal (hand-mount face)** at Titan pre-grasp.

``R_ee`` (Pink IK target) is **not** the arm front — URDF fixes it at ``R_arm_l8`` origin
with a π/2 Z twist vs ``R_arm_l8``. The physical last-link front is **+5 cm along R_ee +Z**
(hand flange / Sharpa mount).

This script:

  1. Builds constructed ``T_base_arm_distal``: origin at pinch pre-grasp, **+Y = approach**.
  2. Converts to IK: ``T_base_ee = T_base_arm_distal @ inv(T_arm_distal_in_ee)``.
  3. Open hand → move to that ``R_ee`` pose only (no close / lift).

Mount offset stays URDF +5 cm along **R_ee +Z** (not along +Y).

Usage (Razor):
  source setup.sh
  python demo/phase2/debug_retarget_arm_distal_pregrasp.py --session-id 20260602_192346_chips --dry-run
  python demo/phase2/debug_retarget_arm_distal_pregrasp.py --session-id 20260602_192346_chips --no-prompts

Pose math only (no Pinocchio IK):
  python demo/phase2/debug_retarget_arm_distal_pregrasp.py --session-id ... --pose-only

Legacy name ``debug_retarget_handbase_pregrasp.py`` forwards here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import GraspObjectConfig  # noqa: E402
from demo.phase1.constants import PHASE1_CONFIG_DIR  # noqa: E402
from demo.phase1.grasp_geometry import approach_dir_from_pose, format_pose, normalize  # noqa: E402
from demo.phase2.arm_distal_geometry import (  # noqa: E402
    ARM_DISTAL_IN_EE_TRANSLATION_M,
    RIGHT_ARM_L8_FRAME,
    RIGHT_EE_FRAME,
    T_arm_distal_in_l8_homogeneous,
    arm_distal_alignment_checks,
    base_ee_from_arm_distal,
    T_base_arm_distal_pregrasp_from_pinch,
)
from demo.phase2.constants import SESSIONS_DIR  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pinch_pose_in_mesh_frame(candidate: dict[str, Any]) -> np.ndarray:
    R = np.asarray(candidate["rotation"], dtype=np.float64)
    t = np.asarray(candidate["grasp_point"], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pinch_pose_in_base(T_base_mesh: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    return np.asarray(T_base_mesh, dtype=np.float64) @ pinch_pose_in_mesh_frame(candidate)


def load_titan_candidate(
    session_dir: Path, rank: int
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    session_dir = Path(session_dir)
    status = _load_json(session_dir / "output" / "status.json")
    if not status.get("success"):
        raise RuntimeError(f"Titan output failed: {status.get('errors', [])}")
    candidates = _load_json(session_dir / "output" / "inference" / "candidates.json")
    T_base_mesh = np.asarray(candidates["T_base_mesh"], dtype=np.float64)
    cands = sorted(
        candidates.get("candidates", []),
        key=lambda c: int(c.get("rank", 9999)),
    )
    cand = next((c for c in cands if int(c.get("rank", -1)) == rank), None)
    if cand is None:
        raise SystemExit(f"No candidate rank={rank} in {session_dir}")
    return T_base_mesh, cand, candidates


@dataclass
class ArmDistalPregraspRetarget:
    rank: int
    name: str
    score: float
    T_base_pinch: np.ndarray
    T_base_arm_distal: np.ndarray
    T_base_ee: np.ndarray
    pre_grasp_offset_m: float


def compute_arm_distal_pregrasp_retarget(
    T_base_pinch: np.ndarray,
    candidate: dict,
    *,
    pre_grasp_offset_m: float = 0.15,
) -> ArmDistalPregraspRetarget:
    T_base_arm_distal = T_base_arm_distal_pregrasp_from_pinch(
        T_base_pinch, pre_grasp_offset_m=pre_grasp_offset_m
    )
    T_base_ee = base_ee_from_arm_distal(T_base_arm_distal)
    return ArmDistalPregraspRetarget(
        rank=int(candidate.get("rank", -1)),
        name=str(candidate.get("name", f"rank_{candidate.get('rank', '?')}")),
        score=float(candidate.get("score", 0.0)),
        T_base_pinch=np.asarray(T_base_pinch, dtype=np.float64),
        T_base_arm_distal=T_base_arm_distal,
        T_base_ee=T_base_ee,
        pre_grasp_offset_m=float(pre_grasp_offset_m),
    )


def log_arm_distal_pregrasp_sanity(retarget: ArmDistalPregraspRetarget) -> None:
    approach = approach_dir_from_pose(retarget.T_base_pinch)
    checks = arm_distal_alignment_checks(
        retarget.T_base_pinch,
        retarget.T_base_arm_distal,
        retarget.T_base_ee,
        pre_grasp_offset_m=retarget.pre_grasp_offset_m,
    )
    logger.info(
        f"Debug arm-distal pre-grasp rank={retarget.rank} ({retarget.name}), "
        f"score={retarget.score:.1f}, offset={retarget.pre_grasp_offset_m:.3f} m"
    )
    logger.info(
        f"URDF: {RIGHT_EE_FRAME} origin = {RIGHT_ARM_L8_FRAME} origin; "
        f"arm-distal face = +{ARM_DISTAL_IN_EE_TRANSLATION_M[2]:.2f} m along {RIGHT_EE_FRAME} +Z"
    )
    logger.info(format_pose(retarget.T_base_pinch, "T_base_pinch (Titan grasp)"))
    logger.info(
        format_pose(
            retarget.T_base_arm_distal,
            "T_base_arm_distal (constructed: mount face, +Y=approach)",
        )
    )
    logger.info(format_pose(retarget.T_base_ee, f"T_base_{RIGHT_EE_FRAME} (Pink IK target)"))
    logger.info(
        f"Arm-distal +Y · Titan approach = {checks['arm_distal_y_dot_approach']:.6f} "
        f"(expect +1.0)"
    )
    logger.info(
        f"Arm-distal +X · Titan finger_open = {checks['arm_distal_x_dot_finger_open']:.6f}"
    )
    logger.info(
        f"R_ee +Y · Titan approach = {checks['R_ee_y_dot_approach']:.6f} "
        f"(same rotation as arm-distal)"
    )
    logger.info(
        f"R_ee +Z · Titan approach = {checks['R_ee_z_dot_approach']:.6f} "
        f"(URDF mount axis; expect ~0 when +Y=approach)"
    )
    logger.info(
        f"R_ee origin is {checks['ee_behind_arm_distal_m']:.4f} m from arm-distal "
        f"({checks['mount_offset_along_ee_z_m']:.4f} m along R_ee +Z; "
        f"expect {ARM_DISTAL_IN_EE_TRANSLATION_M[2]:.3f} m)"
    )
    logger.info(
        f"T_arm_distal_in_l8 t = {T_arm_distal_in_l8_homogeneous()[:3, 3].round(4).tolist()} "
        f"(URDF chain; native l8 axes)"
    )
    if checks["arm_distal_origin_err_m"] > 1e-5:
        logger.warning("Arm-distal origin mismatch vs pinch - offset*approach.")
    if checks["arm_distal_y_dot_approach"] < 0.999:
        logger.warning("Arm-distal +Y is not aligned with Titan approach.")
    if abs(abs(checks["mount_offset_along_ee_z_m"]) - ARM_DISTAL_IN_EE_TRANSLATION_M[2]) > 0.002:
        logger.warning("R_ee ↔ arm-distal spacing along R_ee +Z differs from URDF 5 cm.")

    R_ee_y = normalize(retarget.T_base_ee[:3, 1])
    logger.info(
        f"IK target {RIGHT_EE_FRAME} +Y · approach = {float(np.dot(R_ee_y, approach)):+.4f} "
        f"(arm approach alignment)"
    )
    _log_native_l8_axes_if_available(retarget)


def _log_native_l8_axes_if_available(retarget: ArmDistalPregraspRetarget) -> None:
    try:
        import pinocchio as pin  # noqa: WPS433

        from demo.phase1.constants import DEFAULT_JOINT_POS  # noqa: WPS433
        from demo.phase1.config_io import load_hand_profile  # noqa: WPS433
        from teleop.robot_descriptions import build_full_robot  # noqa: WPS433

        _, open_q = load_hand_profile()
        robot, assemble, _ = build_full_robot(default_joint_by_component=DEFAULT_JOINT_POS)
        q = assemble(
            {
                "left_arm": DEFAULT_JOINT_POS["left_arm"],
                "right_arm": DEFAULT_JOINT_POS["right_arm"],
                "left_hand": np.zeros(22),
                "right_hand": open_q,
            }
        )
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        oMf_l8 = robot.data.oMf[robot.model.getFrameId(RIGHT_ARM_L8_FRAME)]
        oMf_ee = robot.data.oMf[robot.model.getFrameId(RIGHT_EE_FRAME)]
        approach = approach_dir_from_pose(retarget.T_base_pinch)
        logger.info("Pinocchio FK at start-like arm q (open hand, default right arm):")
        for label, oMf in [(RIGHT_ARM_L8_FRAME, oMf_l8), (RIGHT_EE_FRAME, oMf_ee)]:
            for ax, col in [("X", 0), ("Y", 1), ("Z", 2)]:
                d = float(np.dot(normalize(oMf.rotation[:, col]), approach))
                logger.info(f"  {label} +{ax} · approach = {d:+.4f}")
        mount_world = oMf_ee.translation + oMf_ee.rotation @ ARM_DISTAL_IN_EE_TRANSLATION_M
        logger.info(
            f"  FK arm-distal mount − {RIGHT_EE_FRAME} = "
            f"{np.linalg.norm(mount_world - oMf_ee.translation):.4f} m "
            f"(URDF {ARM_DISTAL_IN_EE_TRANSLATION_M[2]:.3f} m)"
        )
    except ImportError:
        logger.debug("Pinocchio unavailable — skipping native l8/R_ee axis FK log")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug: arm-distal at Titan pre-grasp, constructed +Y = approach"
    )
    p.add_argument("--session-id", type=str, required=True)
    p.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    p.add_argument("--config-dir", type=Path, default=PHASE1_CONFIG_DIR)
    p.add_argument("--rank", type=int, default=0, help="Titan candidate rank (default 0)")
    p.add_argument(
        "--pre-grasp-offset",
        type=float,
        default=None,
        help="Retreat along -approach from pinch to arm-distal origin (m)",
    )
    p.add_argument("--dry-run", action="store_true", help="IK check only")
    p.add_argument("--pose-only", action="store_true", help="Pose math only")
    p.add_argument("--no-prompts", action="store_true", help="Skip Enter prompts")
    p.add_argument("--skip-start", action="store_true", help="Skip move to demo start")
    return p.parse_args()


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


def _minimal_motion_config(
    retarget: ArmDistalPregraspRetarget,
    *,
    start_config: GraspObjectConfig,
    object_slug: str,
    table_height_m: float,
) -> GraspObjectConfig:
    return GraspObjectConfig(
        object_name=object_slug,
        table_height=table_height_m,
        grasp_pose=retarget.T_base_ee.copy(),
        pre_grasp_pose=retarget.T_base_ee.copy(),
        lift_pose=retarget.T_base_ee.copy(),
        left_arm_joint_pos=start_config.left_arm_joint_pos.copy(),
        right_arm_joint_pos=start_config.right_arm_joint_pos.copy(),
        torso_joint_pos=start_config.torso_joint_pos.copy(),
        head_joint_pos=start_config.head_joint_pos.copy(),
        left_hand_joint_pos=start_config.left_hand_joint_pos.copy(),
        pre_grasp_offset_m=retarget.pre_grasp_offset_m,
        lift_height_m=float(start_config.lift_height_m),
        hold_time_s=float(start_config.hold_time_s),
        notes=(
            f"debug arm-distal pre-grasp rank={retarget.rank} ({retarget.name}), +Y=approach"
        ),
    )


def _execute_pregrasp(
    executor,
    config: GraspObjectConfig,
    retarget: ArmDistalPregraspRetarget,
    *,
    start_config: GraspObjectConfig | None,
    no_prompts: bool,
    skip_start: bool,
) -> None:
    from demo.phase1.planned_motion import BackgroundPlan, wait_enter_with_prefetch  # noqa: WPS433

    open_hand = config.hand_open_joint_pos
    ik_seed = start_config if start_config is not None else config

    executor.apply_hand_profile(config)
    if start_config is not None:
        executor.apply_hand_profile(start_config)
    executor.apply_demo_head_pose(config, start_config)

    if skip_start:
        logger.info("Skipping move to start (--skip-start).")
    elif start_config is not None:
        executor._move_to_start_pose_arms_then_open_hand(config, start_config)
    else:
        executor._move_to_start_pose_arms_then_open_hand(config, config, label="home_pose")

    prefetch = BackgroundPlan(
        "debug_arm_distal_pregrasp",
        lambda: executor._plan_move_to_right_ee(
            config,
            retarget.T_base_ee,
            open_hand,
            ik_seed_config=ik_seed,
            label="debug_arm_distal_pregrasp",
            right_arm_collision_boxes=None,
        ),
    )
    prompt = (
        "Open hand ready. Press Enter to plan+execute arm-distal pre-grasp debug pose "
        "(mount face +Y=approach)..."
    )
    if no_prompts:
        logger.info("Planning move to arm-distal pre-grasp (open hand)...")
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
        retarget.T_base_ee,
        open_hand,
        planned=planned,
        right_arm_collision_boxes=None,
        label="debug_arm_distal_pregrasp",
        allow_direct_fallback=True,
    )
    logger.info(
        "At debug pre-grasp. Inspect arm-distal / R_ee +Y vs approach; "
        "R_ee +Z is URDF mount axis (~5 cm from flange to mount face)."
    )
    if not no_prompts:
        input("Press Enter to finish (robot stays at pose until shutdown)... ")


def main() -> None:
    args = _parse_args()
    session_dir = Path(args.sessions_dir) / args.session_id
    if not (session_dir / "output" / "status.json").is_file():
        raise FileNotFoundError(f"Missing Titan output: {session_dir / 'output'}")

    T_base_mesh, cand, candidates_json = load_titan_candidate(session_dir, args.rank)
    conv = candidates_json.get("conventions", {})
    pre_off = float(
        args.pre_grasp_offset
        if args.pre_grasp_offset is not None
        else conv.get("pre_grasp_offset_m", 0.15)
    )
    T_base_pinch = pinch_pose_in_base(T_base_mesh, cand)
    retarget = compute_arm_distal_pregrasp_retarget(
        T_base_pinch, cand, pre_grasp_offset_m=pre_off
    )
    log_arm_distal_pregrasp_sanity(retarget)

    if args.pose_only:
        logger.info("Pose-only mode; exiting.")
        return

    from demo.phase2.session_output import load_start_config_for_session  # noqa: WPS433
    from demo.phase2.table_height import estimate_table_height_m_from_session  # noqa: WPS433

    start_config = load_start_config_for_session(session_dir, config_dir=args.config_dir)
    if start_config is None:
        raise SystemExit("Need phase1/configs/start.yaml for joint seeds.")

    table_h, table_src = estimate_table_height_m_from_session(session_dir)
    logger.info(f"Planning table_height: {table_h:.3f} m ({table_src})")
    status = _load_json(session_dir / "output" / "status.json")
    object_slug = str(
        status.get("titan", {}).get("object_slug", args.session_id.split("_")[-1])
    )
    config = _minimal_motion_config(
        retarget,
        start_config=start_config,
        object_slug=object_slug,
        table_height_m=table_h,
    )

    from demo.phase1.executor import GraspExecutor  # noqa: WPS433

    executor = GraspExecutor(dry_run=args.dry_run)
    try:
        executor.setup()
        if not _ik_feasible(
            executor, retarget.T_base_ee, start_config, label="debug_arm_distal_pregrasp"
        ):
            raise SystemExit(f"IK failed for rank {args.rank}.")
        if args.dry_run:
            logger.info("[dry_run] arm-distal pre-grasp IK ok.")
            return
        _execute_pregrasp(
            executor,
            config,
            retarget,
            start_config=start_config,
            no_prompts=args.no_prompts,
            skip_start=args.skip_start,
        )
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
