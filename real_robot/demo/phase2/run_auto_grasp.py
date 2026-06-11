#!/usr/bin/env python3
"""
Phase 2: execute Titan affordance grasp on Razor (retarget + Phase 1 executor).

Pre-grasp OMPL planning optionally includes a Titan mesh collision box (--object-obstacle).
Final approach (grasp EE move) never uses that box.

Usage (Razor):
  source setup.sh
  python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips
  python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips --debug

Off-robot check:
  python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import GraspObjectConfig  # noqa: E402
from demo.phase1.constants import PHASE1_CONFIG_DIR  # noqa: E402
from demo.phase1.grasp_geometry import format_pose  # noqa: E402
from demo.phase2.constants import (  # noqa: E402
    SESSIONS_DIR,
)
from demo.phase2.open_grip_config_io import DEFAULT_OPEN_GRIP_YAML, load_open_grip_config  # noqa: E402
from demo.phase2.object_obstacle import (  # noqa: E402
    format_object_collision_box,
    object_collision_box,
)
from demo.phase2.retarget import (  # noqa: E402
    TitanGraspPoses,
    candidate_poses,
    iter_ranked_candidates,
    load_titan_output,
    log_live_fingertips_vs_titan,
    log_retarget_sanity,
    log_titan_pinch_in_base,
)
from demo.phase2.open_grip_retarget_geometry import right_fingertip_positions_in_base  # noqa: E402
from demo.phase2.session_output import (  # noqa: E402
    grasp_config_from_titan,
    load_start_config_for_session,
    session_dir_for_id,
)
from demo.phase2.candidate_table_collision import (  # noqa: E402
    candidate_table_z_clearance_ok,
    planning_table_height_m_from_session,
    table_obstacle_top_z_m,
)
from demo.phase2.visualize_grasp import show_candidate_grasp_preview  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 auto grasp from Titan output")
    p.add_argument("--session-id", type=str, required=True)
    p.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    p.add_argument("--config-dir", type=Path, default=PHASE1_CONFIG_DIR)
    p.add_argument(
        "--open-grip-config",
        type=Path,
        default=DEFAULT_OPEN_GRIP_YAML,
        help="Open-grip IK parameters YAML (open_pinch_forward_offset_m)",
    )
    p.add_argument(
        "--attempt-index",
        type=int,
        default=None,
        help="Multi-grasp test: use candidate at this sorted index (0=best rank). "
        "Forces --no-random-candidate.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Pool size: consider up to N Titan candidates for IK (default: all exported "
        "in this session). See --random-candidate.",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Force a single candidate rank (disables random shuffle)",
    )
    p.add_argument(
        "--random-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle candidate try order each run (default: on). "
        "Use --no-random-candidate for rank-0-first search.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for --random-candidate (reproducible shuffle)",
    )
    p.add_argument(
        "--object-obstacle",
        action="store_true",
        help="Add Titan mesh AABB during pre-grasp OMPL (default: off)",
    )
    p.add_argument(
        "--object-padding",
        type=float,
        default=1.08,
        help="Scale mesh AABB when --object-obstacle is set",
    )
    p.add_argument(
        "--table-height",
        type=float,
        default=None,
        help="Override table top z in base frame (m). Default: depth ROI median from session input/",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="IK feasibility + pose logs only; no robot connection",
    )
    p.add_argument(
        "--skip-home-at-end",
        action="store_true",
        help="Stay at lift after hold (same as phase1 run_grasp)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Unattended auto demo: skip all Enter prompts (same as --no-prompts)",
    )
    p.add_argument(
        "--no-prompts",
        action="store_true",
        help="Skip Enter prompts between phases (lab automation; enabled by --debug)",
    )
    p.add_argument(
        "--open-pinch-forward-offset",
        type=float,
        default=None,
        help="Open-grip tip-line align point ahead of Titan pinch along approach (m). "
        "Default: open_grip.yaml or 0.015 (1.5 cm)",
    )
    p.add_argument(
        "--ik-palm-soft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include palm lateral (y·y_titan) in open-grip IK soft cost (default: off). "
        "Palm is always hard-checked after IK regardless.",
    )
    p.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip grasp preview entirely (no PNG, no popup)",
    )
    p.add_argument(
        "--grasp-preview-label",
        type=str,
        default=None,
        help="Optional suffix for preview PNG/window title (e.g. attempt-2 from pipeline)",
    )
    p.add_argument(
        "--skip-candidate-collision",
        action="store_true",
        help="Disable table collision check during candidate IK filtering",
    )
    p.add_argument(
        "--no-candidate-fallback",
        action="store_true",
        help="Try only one candidate (first in try order / shuffled pool). "
        "If IK or table collision fails, exit immediately (no next rank).",
    )
    return p.parse_args()


def _create_executor(dry_run: bool):
    """Lazy import so ``--dry-run`` pose preview works without full teleop deps."""
    try:
        from demo.phase1.executor import GraspExecutor  # noqa: WPS433

        executor = GraspExecutor(dry_run=dry_run)
        executor.setup()
        return executor
    except ImportError as exc:
        if dry_run:
            logger.warning(
                f"IK / executor unavailable ({exc}); dry-run will print poses only "
                "(use Razor ``source setup.sh`` for full IK check)."
            )
            return None
        raise


def _sync_grasp_poses_from_open_ik(
    cfg: GraspObjectConfig,
    left_q: np.ndarray,
    right_q: np.ndarray,
    executor,
) -> None:
    from demo.phase1.grasp_geometry import (  # noqa: WPS433
        approach_dir_from_pose,
        compute_lift_pose,
        compute_pre_grasp_pose,
    )
    from demo.phase2.open_grip_retarget_geometry import base_ee_from_open_grip_fk  # noqa: WPS433

    cfg.grasp_pose = base_ee_from_open_grip_fk(
        left_q,
        right_q,
        cfg.hand_open_joint_pos,
        pin_robot=executor.pin_full_robot,
        assemble=executor.assemble_qpos,
    )
    cfg.left_arm_joint_pos = np.asarray(left_q, dtype=np.float64).copy()
    cfg.right_arm_joint_pos = np.asarray(right_q, dtype=np.float64).copy()
    if cfg.titan_T_base_pinch is not None:
        approach = approach_dir_from_pose(cfg.titan_T_base_pinch)
        cfg.pre_grasp_pose = compute_pre_grasp_pose(
            cfg.grasp_pose,
            cfg.pre_grasp_offset_m,
            approach_in_base=approach,
        )
        cfg.lift_pose = compute_lift_pose(cfg.grasp_pose, cfg.lift_height_m)


def _open_grip_joint_targets(
    cfg: GraspObjectConfig,
    start_config: GraspObjectConfig,
    left_q: np.ndarray,
    right_q: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "left_arm": np.asarray(left_q, dtype=np.float64),
        "right_arm": np.asarray(right_q, dtype=np.float64),
        "left_hand": start_config.left_hand_joint_pos.copy(),
        "right_hand": cfg.hand_open_joint_pos.copy(),
    }


def _ik_feasible(
    executor,
    cfg: GraspObjectConfig,
    left_seed: np.ndarray,
    right_seed: np.ndarray,
    *,
    label: str,
    start_config: GraspObjectConfig,
    check_table_collision: bool = True,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Run open-grip IK; optionally verify table collision (Pinocchio, open-hand FK)."""
    from demo.phase1.executor import GraspExecutor  # noqa: WPS433

    GraspExecutor.apply_hand_profile(cfg)
    motion_label = "grasp_approach" if label == "grasp" else label
    T_ee = cfg.grasp_pose if label == "grasp" else cfg.resolved_pre_grasp_pose()
    try:
        left_q, right_q = executor._solve_ik_for_right_ee(
            T_ee,
            left_seed,
            right_seed,
            planning_config=cfg,
            motion_label=motion_label,
        )
        if label == "grasp":
            _sync_grasp_poses_from_open_ik(cfg, left_q, right_q, executor)
        if check_table_collision:
            targets = _open_grip_joint_targets(cfg, start_config, left_q, right_q)
            table_h = float(cfg.table_height)
            if not executor.joint_targets_table_collision_free(
                targets, table_height=table_h
            ):
                box_top = table_obstacle_top_z_m(cfg.table_height)
                logger.info(
                    f"Candidate IK: {label} Pinocchio table collision "
                    f"(box top z≈{box_top:.3f} m, table_h={table_h:.3f} m)"
                )
                return False, None, None
            pin_robot = executor.pin_full_robot
            assemble = executor.assemble_qpos
            if pin_robot is None or assemble is None:
                from demo.phase2.pinch_ik import _full_robot_assemble  # noqa: WPS433

                pin_robot, assemble, _, _, _, _ = _full_robot_assemble()
            z_ok, z_msg = candidate_table_z_clearance_ok(
                cfg,
                start_config,
                left_q,
                right_q,
                label=label,
                pin_robot=pin_robot,
                assemble=assemble,
            )
            if not z_ok:
                logger.info(f"Candidate IK: {z_msg}")
                return False, None, None
        return True, left_q, right_q
    except Exception as exc:
        logger.info(f"IK infeasible ({label}): {exc}")
        return False, None, None


def _candidate_ik_feasible(
    executor,
    cfg: GraspObjectConfig,
    start_config: GraspObjectConfig,
    *,
    check_table_collision: bool = True,
) -> bool:
    """
    Candidate must pass pre_grasp + grasp open-grip IK and table collision (both poses).

    Pre-grasp is seeded from ``start_config`` (motion begins at home/start).
    Grasp is seeded from the pre-grasp solution (matches execution order).
    """
    ok_pre, lq_pre, rq_pre = _ik_feasible(
        executor,
        cfg,
        start_config.left_arm_joint_pos,
        start_config.right_arm_joint_pos,
        label="pre_grasp",
        start_config=start_config,
        check_table_collision=check_table_collision,
    )
    if not ok_pre:
        logger.info("Candidate IK: pre_grasp failed")
        return False
    ok_grasp, lq_grasp, rq_grasp = _ik_feasible(
        executor,
        cfg,
        lq_pre,
        rq_pre,
        label="grasp",
        start_config=start_config,
        check_table_collision=check_table_collision,
    )
    if not ok_grasp:
        logger.info("Candidate IK: grasp failed (seeded from pre_grasp solution)")
        return False
    cfg.left_arm_joint_pos = np.asarray(lq_grasp, dtype=np.float64)
    cfg.right_arm_joint_pos = np.asarray(rq_grasp, dtype=np.float64)
    return True


def _candidate_try_order(
    cands: list[dict],
    *,
    max_candidates: int | None,
    random_candidate: bool,
    seed: int | None,
) -> list[dict]:
    """Build the IK try pool; optionally shuffle so each run picks a different rank first."""
    pool = list(cands if max_candidates is None else cands[:max_candidates])
    if random_candidate and len(pool) > 1:
        rng = random.Random(seed)
        rng.shuffle(pool)
    return pool


def run_titan_grasp_sequence(
    executor,
    config: GraspObjectConfig,
    *,
    start_config: GraspObjectConfig | None,
    object_boxes: list[dict],
    titan_poses: TitanGraspPoses | None = None,
    no_prompts: bool = False,
    skip_home_at_end: bool = False,
) -> None:
    """
    Like Phase 1 ``run_sequence`` but grasp targets come from Titan EE poses.

    Pre-grasp: collision plan with ``object_boxes`` on right-arm stage only.
    Grasp EE: plan/execute without object box (final approach).
    """
    executor.apply_hand_profile(config)
    if start_config is not None:
        executor.apply_hand_profile(start_config)
    executor.apply_demo_head_pose(config, start_config)

    pre = config.resolved_pre_grasp_pose()
    grasp = config.grasp_pose.copy()
    lift = config.resolved_lift_pose()
    open_hand = config.hand_open_joint_pos

    logger.info(f"=== Phase 2 auto grasp: {config.object_name} ===")
    logger.info(f"Planning table_height: {config.table_height:.3f} m")
    logger.info(format_pose(grasp, "grasp (R_ee)"))
    logger.info(format_pose(pre, "pre_grasp"))
    logger.info(format_pose(lift, "lift"))
    if config.titan_T_base_pinch is not None:
        logger.info(
            f"Arm IK: open-grip retarget "
            f"(forward_offset={config.open_pinch_forward_offset_m:.3f} m, "
            f"ik_palm_soft={config.open_grip_ik_palm_soft})"
        )
    if object_boxes:
        logger.info(
            f"Object obstacle (pre-grasp only): {format_object_collision_box(object_boxes[0])}"
        )

    from demo.phase1.planned_motion import BackgroundPlan, wait_enter_with_prefetch  # noqa: WPS433

    if executor.dry_run:
        for label, T in [("pre_grasp", pre), ("grasp_approach", grasp), ("lift", lift)]:
            _, rq = executor._solve_ik_for_right_ee(
                T,
                config.left_arm_joint_pos,
                config.right_arm_joint_pos,
                planning_config=config,
                motion_label=label,
            )
            logger.info(f"[dry_run] {label}: right_arm IK ok, q={np.round(rq, 3).tolist()}")
        logger.info("[dry_run] sequence complete (no hardware)")
        return

    ik_seed = start_config if start_config is not None else config

    if start_config is not None:
        executor._move_to_start_pose_arms_then_open_hand(config, start_config)
    else:
        logger.warning("No start.yaml — arms-first home move with hand unchanged")
        executor._move_to_start_pose_arms_then_open_hand(config, config, label="home_pose")

    prefetch_pre = BackgroundPlan(
        "pre_grasp",
        lambda: executor._plan_move_to_right_ee(
            config,
            pre,
            open_hand,
            ik_seed_config=ik_seed,
            label="pre_grasp",
            right_arm_collision_boxes=object_boxes,
        ),
    )
    if no_prompts:
        logger.info(
            f"Planning pre-grasp with object obstacle for '{config.object_name}'..."
        )
        prefetch_pre.start()
        try:
            planned_pre = prefetch_pre.wait()
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Pre-grasp background plan failed ({exc}); will try direct IK fallback")
            planned_pre = None
    else:
        try:
            planned_pre = wait_enter_with_prefetch(
                f"Place '{config.object_name}' on the table, then press Enter to "
                f"plan+execute pre-grasp (object obstacle ON)...",
                prefetch_pre,
            )
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Pre-grasp background plan failed ({exc}); will try direct IK fallback")
            planned_pre = None
    executor._move_to_right_ee_pose(
        config,
        pre,
        open_hand,
        planned=planned_pre,
        right_arm_collision_boxes=object_boxes,
        label="pre_grasp",
        allow_direct_fallback=True,
    )

    # Final approach: grasp EE without object collision box
    prefetch_grasp = BackgroundPlan(
        "grasp_approach",
        lambda: executor._plan_move_to_right_ee(
            config,
            grasp,
            open_hand,
            ik_seed_config=ik_seed,
            label="grasp_approach",
            right_arm_collision_boxes=None,
        ),
    )
    if no_prompts:
        logger.info("Planning final grasp approach (object obstacle OFF)...")
        prefetch_grasp.start()
        try:
            planned_grasp = prefetch_grasp.wait()
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Grasp approach plan failed ({exc}); will try direct IK fallback")
            planned_grasp = None
    else:
        try:
            planned_grasp = wait_enter_with_prefetch(
                "Press Enter to plan+execute final grasp approach (object obstacle OFF)...",
                prefetch_grasp,
            )
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"Grasp approach plan failed ({exc}); will try direct IK fallback")
            planned_grasp = None
    executor._move_to_right_ee_pose(
        config,
        grasp,
        open_hand,
        planned=planned_grasp,
        right_arm_collision_boxes=None,
        label="grasp_approach",
        allow_direct_fallback=True,
    )

    left_at_grasp, right_at_grasp = executor._read_arm_joint_pos()

    logger.info("Closing virtual gripper (stall-detect)...")
    q_grasp = executor._close_right_hand_until_stall(config)

    if titan_poses is not None and not executor.dry_run:
        left_q, right_q = executor._read_arm_joint_pos()
        tips = right_fingertip_positions_in_base(
            left_q,
            right_q,
            q_grasp,
            pin_robot=executor.pin_full_robot,
            assemble=executor.assemble_qpos,
        )
        log_live_fingertips_vs_titan(tips, titan_poses)

    executor._move_to_right_ee_pose(
        config, lift, q_grasp, right_arm_collision_boxes=None, label="lift",
        allow_direct_fallback=True,
    )

    grasp_closed_targets = {
        "left_arm": left_at_grasp,
        "right_arm": right_at_grasp,
        "left_hand": config.left_hand_joint_pos,
        "right_hand": q_grasp,
    }

    prefetch_grasp_return = BackgroundPlan(
        "lift_to_grasp",
        lambda: executor._plan_joint_targets(
            config,
            grasp_closed_targets,
            label="lift_to_grasp",
        ),
    )
    prefetch_grasp_return.start()
    logger.info(
        f"Holding at lift for {config.hold_time_s:.1f}s "
        f"(hand closed; planning lower to grasp pose)..."
    )
    time.sleep(config.hold_time_s)

    if skip_home_at_end:
        logger.info("Staying at lift (--skip-home-at-end).")
        return

    if no_prompts:
        logger.info("Lowering to grasp pose (hand still closed)...")
        try:
            planned_grasp_return = prefetch_grasp_return.wait()
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"lift_to_grasp plan failed ({exc}); direct move")
            planned_grasp_return = None
    else:
        try:
            planned_grasp_return = wait_enter_with_prefetch(
                "Lift complete. Press Enter to lower to grasp pose (hand still closed)...",
                prefetch_grasp_return,
                prefetch_already_started=True,
            )
        except (AssertionError, RuntimeError) as exc:
            logger.warning(f"lift_to_grasp plan failed ({exc}); direct move")
            planned_grasp_return = None
    executor._move_to_joint_targets(
        config, grasp_closed_targets, planned=planned_grasp_return, label="lift_to_grasp",
        allow_direct_fallback=True,
    )
    left_q, right_q = executor._read_arm_joint_pos()
    executor._publish_action_buffer(
        {
            "left_arm": left_q,
            "right_arm": right_q,
            "left_hand": config.left_hand_joint_pos,
            "right_hand": q_grasp,
        }
    )

    logger.info("Opening gripper at grasp pose (object on table)...")
    executor._open_right_grip_stepped(config)

    prefetch_start = None
    if start_config is not None:
        prefetch_start = BackgroundPlan(
            "return_start",
            lambda: executor._plan_joint_targets(
                config,
                executor.joint_targets_with_open_grip(start_config, config),
                pose_config=start_config,
                label="return_start",
            ),
        )
    if no_prompts:
        if prefetch_start is not None:
            prefetch_start.start()
            try:
                planned_start = prefetch_start.wait()
            except (AssertionError, RuntimeError) as exc:
                logger.warning(f"return_start plan failed ({exc})")
                planned_start = None
        else:
            planned_start = None
        if planned_start is not None:
            logger.info("Returning to demo start pose...")
            executor._move_to_joint_targets(
                config,
                executor.joint_targets_with_open_grip(start_config, config),
                pose_config=start_config,
                planned=planned_start,
                label="return_start",
                allow_direct_fallback=True,
            )
    else:
        planned_start = wait_enter_with_prefetch(
            f"Remove '{config.object_name}' from the gripper, then press Enter to return to start...",
            prefetch_start,
        )
        if planned_start is not None:
            logger.info("Returning to demo start pose...")
            executor._move_to_joint_targets(
                config,
                executor.joint_targets_with_open_grip(start_config, config),
                pose_config=start_config,
                planned=planned_start,
                label="return_start",
                allow_direct_fallback=True,
            )
    logger.info("Phase 2 auto grasp sequence complete.")


def main() -> None:
    args = _parse_args()
    if args.debug:
        args.no_prompts = True
        logger.info("Debug mode: skipping all Enter prompts")
    session_dir = session_dir_for_id(args.session_id, args.sessions_dir)
    if not (session_dir / "output" / "status.json").is_file():
        raise FileNotFoundError(f"Missing Titan output: {session_dir / 'output'}")

    output = load_titan_output(session_dir)
    open_grip_cfg = load_open_grip_config(args.open_grip_config)
    start_config = load_start_config_for_session(session_dir, config_dir=args.config_dir)

    if args.attempt_index is not None:
        ranked = iter_ranked_candidates(output)
        idx = int(args.attempt_index)
        if idx < 0 or idx >= len(ranked):
            raise SystemExit(
                f"--attempt-index {idx} out of range "
                f"(Titan exported {len(ranked)} candidates)"
            )
        args.rank = int(ranked[idx]["rank"])
        args.random_candidate = False
        logger.info(
            f"Grasp attempt-index {idx} → Titan rank {args.rank} "
            f"({ranked[idx].get('name', '?')})"
        )

    from demo.phase1.constants import DEFAULT_TABLE_HEIGHT_M  # noqa: WPS433

    table_h, table_src = planning_table_height_m_from_session(
        session_dir, default=DEFAULT_TABLE_HEIGHT_M
    )
    logger.info(f"Planning table_height: {table_h:.3f} m ({table_src})")
    if args.table_height is not None:
        table_h = float(args.table_height)
        logger.info(f"Using --table-height={table_h:.3f} m (CLI override)")
    object_slug = str(output.status.get("titan", {}).get("object_slug", args.session_id.split("_")[-1]))
    if args.object_obstacle:
        obj_box = object_collision_box(output, padding=args.object_padding)
        object_boxes: list[dict] = [obj_box]
        logger.info(
            f"Object obstacle (pre-grasp right-arm plan): {format_object_collision_box(obj_box)}"
        )
    else:
        object_boxes = []
        logger.info("Object obstacle: disabled (pre-grasp OMPL without mesh collision box)")

    executor = _create_executor(args.dry_run)

    cands = iter_ranked_candidates(output)
    if args.rank is not None:
        cands = [c for c in cands if int(c["rank"]) == args.rank]
        if not cands:
            raise SystemExit(f"No candidate rank={args.rank}")

    try_pool = _candidate_try_order(
        cands,
        max_candidates=args.max_candidates,
        random_candidate=args.random_candidate and args.rank is None,
        seed=args.seed,
    )
    if args.no_candidate_fallback:
        try_pool = try_pool[:1]
        logger.info(
            "--no-candidate-fallback: single candidate only "
            f"(rank={int(try_pool[0].get('rank', -1)) if try_pool else 'none'}); "
            "IK/collision fail → exit, no retry"
        )
    n_try = len(try_pool)
    if n_try == 0:
        raise SystemExit("No candidates to try (empty Titan export or rank filter)")
    if args.rank is not None:
        logger.info(
            f"Candidate search: forced rank={args.rank} "
            f"(Titan exported {len(cands)} total; IK + table collision at this stage)"
        )
    elif args.random_candidate:
        order = ", ".join(str(int(c.get("rank", -1))) for c in try_pool)
        seed_note = f", seed={args.seed}" if args.seed is not None else ""
        logger.info(
            f"Candidate search: random try order [{order}] from pool of {n_try} "
            f"(Titan exported {len(cands)} total{seed_note}; IK + table collision "
            f"at this stage)"
        )
    else:
        logger.info(
            f"Candidate search: pre_grasp + grasp IK + table collision "
            f"on ranks 0..{n_try - 1} "
            f"(Titan exported {len(cands)} total)"
        )

    chosen_config: GraspObjectConfig | None = None
    chosen_poses = None
    try:
        for cand in try_pool:
            rank = int(cand.get("rank", -1))
            poses = candidate_poses(output, cand)
            if start_config is None:
                logger.error("Need capture pose_source → phase1/configs/start.yaml on disk")
                raise SystemExit(1)
            cfg = grasp_config_from_titan(
                poses,
                start_config=start_config,
                object_slug=object_slug,
                table_height_m=table_h,
                open_pinch_forward_offset_m=(
                    args.open_pinch_forward_offset
                    if args.open_pinch_forward_offset is not None
                    else open_grip_cfg.open_pinch_forward_offset_m
                ),
                open_grip_ik_palm_soft=args.ik_palm_soft,
            )
            if executor is not None:
                logger.info(f"Trying rank {rank} ({poses.name})...")
                if not _candidate_ik_feasible(
                    executor,
                    cfg,
                    start_config,
                    check_table_collision=not args.skip_candidate_collision,
                ):
                    fail_msg = (
                        f"Rank {rank} ({poses.name}): "
                        "pre_grasp/grasp IK (incl. thumb-axis·approach hard gate), or table collision failed"
                    )
                    if args.no_candidate_fallback:
                        raise SystemExit(
                            f"{fail_msg} (--no-candidate-fallback)"
                        )
                    logger.warning(f"Skip {fail_msg}")
                    continue
                logger.info(
                    f"Rank {rank} ({poses.name}): pre_grasp + grasp IK "
                    f"{'+ table ok ' if not args.skip_candidate_collision else ''}ok"
                )
            logger.info(
                f"Selected candidate rank={poses.rank} ({poses.name}), score={poses.score:.1f}"
            )
            log_titan_pinch_in_base(poses)
            chosen_config = cfg
            chosen_poses = poses
            poses.grasp_pose = cfg.grasp_pose.copy()
            log_retarget_sanity(
                poses,
                forward_offset_m=chosen_config.open_pinch_forward_offset_m,
                right_arm_q=chosen_config.right_arm_joint_pos,
                left_arm_q=chosen_config.left_arm_joint_pos,
                right_hand_open=chosen_config.hand_open_joint_pos,
            )
            break

        if chosen_config is None:
            if args.rank is not None:
                raise SystemExit(f"Rank {args.rank} failed pinch-first IK (grasp or pre_grasp)")
            raise SystemExit(
                f"No feasible candidate in try pool of {n_try} "
                f"(try --max-candidates or tune start pose / --open-pinch-forward-offset)"
            )

        if not args.no_visualize and chosen_poses is not None:
            try:
                show_candidate_grasp_preview(
                    session_dir,
                    output,
                    chosen_poses,
                    block=True,
                    preview_label=args.grasp_preview_label,
                )
            except Exception as exc:
                logger.warning(f"Grasp preview failed ({exc}); continuing to motion.")

        if executor is None:
            log_titan_pinch_in_base(chosen_poses)
            logger.info(format_pose(chosen_config.grasp_pose, "grasp (retarget)"))
            logger.info(format_pose(chosen_config.resolved_pre_grasp_pose(), "pre_grasp"))
            logger.info(format_pose(chosen_config.resolved_lift_pose(), "lift"))
            logger.info(
                "Dry-run pose preview only (no IK). Re-run on Razor with setup.sh for full dry-run."
            )
        else:
            run_titan_grasp_sequence(
                executor,
                chosen_config,
                start_config=start_config,
                object_boxes=object_boxes,
                titan_poses=chosen_poses,
                no_prompts=args.no_prompts,
                skip_home_at_end=args.skip_home_at_end,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        if executor is not None:
            executor.shutdown()


if __name__ == "__main__":
    main()
