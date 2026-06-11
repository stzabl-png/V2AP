"""Stall-detecting pinch close for Sharpa right hand (contact-safe)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from demo.constants import (
    HAND_INTERPOLATE,
    HAND_TRACK_TOL_RAD,
    PREVIEW_CLOSE_HOLD_S,
    STALL_CLOSE_GRACE_S,
    STALL_CLOSE_HOLD_S,
    STALL_CLOSE_MAX_S,
    STALL_CLOSE_MIN_PROGRESS_RAD,
    STALL_CLOSE_MIN_STEPS,
    STALL_CLOSE_MONITOR_JOINTS,
    STALL_CLOSE_POLL_S,
    STALL_CLOSE_REACH_TOL_RAD,
    STALL_CLOSE_STEP_RAD,
    STALL_CLOSE_VEL_RAD,
)
from demo.hardware import set_hand_joint_positions
from demo.virtual_gripper import clip_hand_q, validate_hand_q

StallCloseReason = Literal["stall", "target", "timeout"]


@dataclass
class StallCloseResult:
    q_final: np.ndarray
    reason: StallCloseReason
    steps: int


def read_hand_joint_pos(hand) -> np.ndarray:
    return clip_hand_q(np.array(hand.get_states().angles, dtype=np.float64))


def _pinch_max_delta(prev: np.ndarray, curr: np.ndarray, joint_indices: tuple[int, ...]) -> float:
    idx = list(joint_indices)
    return float(np.max(np.abs(curr[idx] - prev[idx])))


def _pinch_tracking_error(
    q_feedback: np.ndarray, q_target: np.ndarray, joint_indices: tuple[int, ...]
) -> float:
    idx = list(joint_indices)
    return float(np.max(np.abs(q_target[idx] - q_feedback[idx])))


def close_hand_until_stall(
    hand,
    q_closed: np.ndarray,
    *,
    q_start: np.ndarray | None = None,
    monitor_joint_indices: tuple[int, ...] = STALL_CLOSE_MONITOR_JOINTS,
    step_rad: float = STALL_CLOSE_STEP_RAD,
    poll_s: float = STALL_CLOSE_POLL_S,
    stall_vel_rad: float = STALL_CLOSE_VEL_RAD,
    stall_hold_s: float = STALL_CLOSE_HOLD_S,
    grace_s: float = STALL_CLOSE_GRACE_S,
    min_steps: int = STALL_CLOSE_MIN_STEPS,
    min_progress_rad: float = STALL_CLOSE_MIN_PROGRESS_RAD,
    max_duration_s: float = STALL_CLOSE_MAX_S,
    reach_tol_rad: float = STALL_CLOSE_REACH_TOL_RAD,
    dry_run: bool = False,
) -> StallCloseResult:
    """
    Incrementally close toward q_closed; stop when thumb/index joints stall (contact)
    or the target is reached.

    Uses a command integrator (not lagging feedback) so steps keep advancing even if
    the hand is slow to report motion. Stall is only checked after a grace period and
    minimum progress, to avoid exiting before the hand starts moving.
    """
    q_closed = clip_hand_q(validate_hand_q(q_closed, "q_closed"))
    q_feedback = clip_hand_q(q_start) if q_start is not None else read_hand_joint_pos(hand)
    q_cmd = q_feedback.copy()

    if dry_run:
        logger.info(
            f"[dry_run] close_until_stall: start thumb/index={np.round(q_feedback[:9], 3).tolist()} "
            f"target={np.round(q_closed[:9], 3).tolist()}"
        )
        return StallCloseResult(q_final=q_closed.copy(), reason="target", steps=0)

    pinch_err_initial = _pinch_tracking_error(q_feedback, q_closed, monitor_joint_indices)
    stall_samples_needed = max(1, int(round(stall_hold_s / poll_s)))
    stall_run = 0
    steps = 0
    t0 = time.monotonic()
    prev_feedback = q_feedback.copy()

    logger.debug(
        f"close_until_stall: initial pinch err {pinch_err_initial:.4f} rad, "
        f"grace {grace_s:.2f}s, min_steps {min_steps}"
    )

    while time.monotonic() - t0 < max_duration_s:
        pinch_err = _pinch_tracking_error(q_feedback, q_closed, monitor_joint_indices)
        if pinch_err <= reach_tol_rad:
            set_hand_joint_positions(hand, q_feedback, interpolate=False)
            logger.info(
                f"close_until_stall: reached target (pinch err {pinch_err:.4f} rad) "
                f"after {steps} steps"
            )
            return StallCloseResult(q_final=q_feedback.copy(), reason="target", steps=steps)

        delta = q_closed - q_cmd
        step = np.clip(delta, -step_rad, step_rad)
        q_cmd = clip_hand_q(q_cmd + step)
        set_hand_joint_positions(hand, q_cmd, interpolate=HAND_INTERPOLATE)
        steps += 1
        time.sleep(poll_s)

        q_feedback = read_hand_joint_pos(hand)
        elapsed = time.monotonic() - t0
        progress = pinch_err_initial - pinch_err
        max_dq = _pinch_max_delta(prev_feedback, q_feedback, monitor_joint_indices)

        may_check_stall = (
            elapsed >= grace_s
            and steps >= min_steps
            and progress >= min_progress_rad
            and pinch_err > reach_tol_rad
        )
        if may_check_stall:
            if max_dq < stall_vel_rad:
                stall_run += 1
            else:
                stall_run = 0
            if stall_run >= stall_samples_needed:
                set_hand_joint_positions(hand, q_feedback, interpolate=False)
                logger.info(
                    f"close_until_stall: stall after {steps} steps, "
                    f"progress {progress:.4f} rad, pinch err {pinch_err:.4f}; "
                    f"thumb/index={np.round(q_feedback[:9], 3).tolist()}"
                )
                return StallCloseResult(q_final=q_feedback.copy(), reason="stall", steps=steps)
        else:
            stall_run = 0

        prev_feedback = q_feedback.copy()

    set_hand_joint_positions(hand, q_feedback, interpolate=False)
    logger.warning(
        f"close_until_stall: timeout after {max_duration_s:.1f}s, {steps} steps, "
        f"pinch err {_pinch_tracking_error(q_feedback, q_closed, monitor_joint_indices):.4f}"
    )
    return StallCloseResult(q_final=q_feedback.copy(), reason="timeout", steps=steps)


def move_hand_to_target(
    hand,
    q_target: np.ndarray,
    *,
    q_start: np.ndarray | None = None,
    step_rad: float = STALL_CLOSE_STEP_RAD,
    poll_s: float = STALL_CLOSE_POLL_S,
    reach_tol_rad: float = HAND_TRACK_TOL_RAD,
    max_duration_s: float = STALL_CLOSE_MAX_S,
    dry_run: bool = False,
) -> np.ndarray:
    """
    Step toward q_target with interpolate=False on every command.

    Avoids Sharpa SDK interpolate, which can pass through a full palm-spread pose
    before reaching the virtual-gripper open configuration.
    """
    q_target = clip_hand_q(validate_hand_q(q_target, "q_target"))
    q_cmd = clip_hand_q(q_start) if q_start is not None else read_hand_joint_pos(hand)

    if dry_run:
        logger.info(f"[dry_run] move_hand_to_target: {np.round(q_target[:9], 3).tolist()}")
        return q_target.copy()

    t0 = time.monotonic()
    steps = 0
    while time.monotonic() - t0 < max_duration_s:
        err = float(np.max(np.abs(q_target - q_cmd)))
        if err <= reach_tol_rad:
            set_hand_joint_positions(hand, q_target, interpolate=False)
            logger.debug(f"move_hand_to_target: reached in {steps} steps")
            return read_hand_joint_pos(hand)

        delta = q_target - q_cmd
        q_cmd = clip_hand_q(q_cmd + np.clip(delta, -step_rad, step_rad))
        set_hand_joint_positions(hand, q_cmd, interpolate=False)
        steps += 1
        time.sleep(poll_s)

    set_hand_joint_positions(hand, q_cmd, interpolate=False)
    logger.warning(f"move_hand_to_target: timeout after {max_duration_s:.1f}s ({steps} steps)")
    return read_hand_joint_pos(hand)


def preview_stall_close_then_open(
    hand,
    q_closed: np.ndarray,
    q_open: np.ndarray,
    *,
    hold_s: float = PREVIEW_CLOSE_HOLD_S,
    dry_run: bool = False,
) -> StallCloseResult:
    """Close until stall/target, hold, then reopen (pose_tuner ']' preview)."""
    result = close_hand_until_stall(hand, q_closed, dry_run=dry_run)
    if dry_run:
        return result
    logger.info(f"Preview close done ({result.reason}); holding {hold_s:.1f}s before open...")
    time.sleep(hold_s)
    move_hand_to_target(hand, q_open)
    logger.info("Hand reopened after preview close")
    return result
