"""Interactive right-hand joint tuning (run on Razor).

Edits the shared hand profile used by all Phase 1 object configs.
Does not move the robot arm.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.constants import (
    DEFAULT_HAND_PROFILE_PATH,
    HAND_APPLY_WAIT_S,
    HAND_TRACK_TOL_RAD,
)
from demo.phase1.config_io import load_hand_profile, save_hand_profile
from demo.phase1.constants import TUNER_HAND_JOINT_STEP_RAD
from demo.hand_close import close_hand_until_stall
from demo.hardware import (
    connect_right_hand_only,
    set_hand_joint_positions,
    setup_cpp_logging,
    shutdown_right_hand,
)
from demo.virtual_gripper import HAND_JOINT_NAMES, clip_hand_q

_ESC = "ESC"
_ESCAPE_FOLLOWUP_TIMEOUT_S = 0.02

_FINGER_RANGES: tuple[tuple[int, int, str], ...] = (
    (0, 5, "thumb"),
    (5, 9, "index"),
    (9, 13, "middle"),
    (13, 17, "ring"),
    (17, 22, "pinky"),
)


@dataclass
class HandTunerState:
    q_live: np.ndarray
    q_open: np.ndarray
    q_closed: np.ndarray
    finger_idx: int = 0
    joint_offset: int = 0

    @property
    def joint_index(self) -> int:
        start, _, _ = _FINGER_RANGES[self.finger_idx]
        return start + self.joint_offset

    @property
    def joint_name(self) -> str:
        return HAND_JOINT_NAMES[self.joint_index]

    def select_finger(self, finger_num: int) -> None:
        self.finger_idx = max(0, min(len(_FINGER_RANGES) - 1, finger_num - 1))
        self.joint_offset = 0

    def cycle_joint(self, delta: int) -> None:
        start, end, _ = _FINGER_RANGES[self.finger_idx]
        count = end - start
        self.joint_offset = (self.joint_offset + delta) % count


def _consume_escape_sequence() -> None:
    if not select.select([sys.stdin], [], [], 0)[0]:
        return
    lead = sys.stdin.read(1)
    if lead == "[":
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch.isalpha() or ch in "~$":
                break
    elif lead == "O" and select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)


def _read_key() -> str | None:
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    if not select.select([sys.stdin], [], [], _ESCAPE_FOLLOWUP_TIMEOUT_S)[0]:
        return _ESC
    _consume_escape_sequence()
    return None


def _read_right_hand_q(right_hand) -> np.ndarray:
    return clip_hand_q(np.array(right_hand.get_states().angles, dtype=np.float64))


def _apply_hand(
    right_hand,
    q: np.ndarray,
    *,
    wait: bool = False,
    wait_s: float = HAND_APPLY_WAIT_S,
) -> np.ndarray:
    q = clip_hand_q(q)
    set_hand_joint_positions(right_hand, q)
    if wait:
        _wait_until_near(right_hand, q, wait_s=wait_s)
    return q


def _wait_until_near(
    right_hand,
    target: np.ndarray,
    *,
    wait_s: float = HAND_APPLY_WAIT_S,
    tol: float = HAND_TRACK_TOL_RAD,
) -> None:
    deadline = time.monotonic() + wait_s
    target = clip_hand_q(target)
    while time.monotonic() < deadline:
        time.sleep(0.05)
        current = clip_hand_q(np.array(right_hand.get_states().angles, dtype=np.float64))
        if float(np.max(np.abs(current - target))) <= tol:
            return
    err = float(np.max(np.abs(current - target)))
    logger.warning(f"hand still {err:.3f} rad from target after {wait_s:.1f}s")


def _log_hand_vec(label: str, q: np.ndarray) -> None:
    logger.info(f"{label} thumb/index: {np.round(q[:9], 4).tolist()}")


def _live_differs_from_closed(state: HandTunerState, tol: float = 0.02) -> bool:
    return bool(np.max(np.abs(state.q_live - state.q_closed)) > tol)


def _close_with_stall(right_hand, q_closed: np.ndarray) -> np.ndarray:
    result = close_hand_until_stall(right_hand, q_closed)
    logger.info(f"Stall close: reason={result.reason}, steps={result.steps}")
    return result.q_final


def _print_state(state: HandTunerState, profile_path: Path, step: float) -> None:
    start, end, finger = _FINGER_RANGES[state.finger_idx]
    print("\n" + "=" * 72)
    print(f"profile: {profile_path}  (shared by all objects)")
    print(f"finger: {finger} ({start}-{end - 1})  step: {step:.3f} rad")
    print(
        f"selected: [{state.joint_index:2d}] {state.joint_name} = "
        f"{state.q_live[state.joint_index]:+.4f} rad"
    )
    print(f"live  thumb/index: {np.round(state.q_live[:9], 3).tolist()}")
    print(f"open  thumb/index: {np.round(state.q_open[:9], 3).tolist()}")
    print(f"closed thumb/index: {np.round(state.q_closed[:9], 3).tolist()}")
    print("=" * 72 + "\n")


def _print_help() -> None:
    print(
        """
Keys (no Enter):
  Finger select:           1=thumb  2=index  3=middle  4=ring  5=pinky
  Joint within finger:     J=next   K=prev
  Nudge selected joint:    W=+      S=-
  Profile presets:         [ = open slot   ] = closed slot (from YAML)
  Mark open/closed slots:  O = copy live -> open   C = copy live -> closed
  Read from hardware:      R
  Apply slot to hand:      G = apply open   N = apply closed
  Print state:             P
  Save / reload profile:   > = save slots to disk
                           (C/O also auto-save; required for --apply-closed)
  Quit:                    Esc (arrow keys ignored)

N uses the closed SLOT in memory; --apply-closed reads the YAML on disk.
After tuning with W/S, press C before N or > so disk matches what you see.

Safety: keep the hand clear of objects before tuning.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune shared right-hand profile (all Phase 1 objects)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=str(DEFAULT_HAND_PROFILE_PATH),
        help="Hand profile YAML (default: demo/right_hand_profile.yaml)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=TUNER_HAND_JOINT_STEP_RAD,
        help="Joint nudge step size in radians",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=None,
        help="Start with this joint index selected (0-21)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify imports only; does not connect to hardware",
    )
    parser.add_argument(
        "--apply-open",
        action="store_true",
        help="Apply profile open pose once and exit (requires hardware)",
    )
    parser.add_argument(
        "--apply-closed",
        action="store_true",
        help="Apply profile closed pose once and exit (requires hardware)",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("hand_tuner dry-run: imports OK (no hardware connection)")
        return

    profile_path = Path(args.profile)
    q_open, q_closed = load_hand_profile(profile_path)
    state = HandTunerState(q_live=q_open.copy(), q_open=q_open, q_closed=q_closed)

    if args.joint is not None:
        j = args.joint
        if not 0 <= j < len(HAND_JOINT_NAMES):
            raise ValueError(f"--joint must be 0-{len(HAND_JOINT_NAMES) - 1}, got {j}")
        for i, (start, end, _) in enumerate(_FINGER_RANGES):
            if start <= j < end:
                state.finger_idx = i
                state.joint_offset = j - start
                break

    setup_cpp_logging(console_log=False)
    right_hand, manager = connect_right_hand_only()

    one_shot = args.apply_open or args.apply_closed
    try:
        if args.apply_open:
            _log_hand_vec(f"Loading hand_open from {profile_path}", state.q_open)
            _apply_hand(right_hand, state.q_open, wait=True)
            logger.info(f"Applied hand_open from {profile_path}")
            return
        if args.apply_closed:
            _log_hand_vec(f"Loading hand_closed from {profile_path}", state.q_closed)
            _apply_hand(right_hand, state.q_open, wait=True)
            _close_with_stall(right_hand, state.q_closed)
            logger.info(f"Applied hand_closed (stall-detect) from {profile_path}")
            return

        state.q_live = _apply_hand(right_hand, state.q_live, wait=True)
        _print_help()
        _print_state(state, profile_path, args.step)

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while True:
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                key = _read_key()
                if key is None:
                    continue
                if key == _ESC:
                    break

                ch = key
                if ch in "12345":
                    state.select_finger(int(ch))
                    _print_state(state, profile_path, args.step)
                elif ch in ("j", "J"):
                    state.cycle_joint(1)
                    _print_state(state, profile_path, args.step)
                elif ch in ("k", "K"):
                    state.cycle_joint(-1)
                    _print_state(state, profile_path, args.step)
                elif ch in ("w", "W"):
                    idx = state.joint_index
                    state.q_live[idx] += args.step
                    state.q_live = _apply_hand(right_hand, state.q_live)
                    _print_state(state, profile_path, args.step)
                elif ch in ("s", "S"):
                    idx = state.joint_index
                    state.q_live[idx] -= args.step
                    state.q_live = _apply_hand(right_hand, state.q_live)
                    _print_state(state, profile_path, args.step)
                elif ch == "[":
                    state.q_live = _apply_hand(right_hand, state.q_open.copy())
                    _print_state(state, profile_path, args.step)
                elif ch == "]":
                    state.q_live = _close_with_stall(right_hand, state.q_closed.copy())
                    _print_state(state, profile_path, args.step)
                elif ch in ("o", "O"):
                    state.q_open = state.q_live.copy()
                    written = save_hand_profile(state.q_open, state.q_closed, profile_path)
                    logger.info(f"Copied live -> open slot; saved {written}")
                    _log_hand_vec("open", state.q_open)
                    _print_state(state, profile_path, args.step)
                elif ch in ("c", "C"):
                    state.q_closed = state.q_live.copy()
                    written = save_hand_profile(state.q_open, state.q_closed, profile_path)
                    logger.info(f"Copied live -> closed slot; saved {written}")
                    _log_hand_vec("closed", state.q_closed)
                    _print_state(state, profile_path, args.step)
                elif ch in ("r", "R"):
                    state.q_live = _read_right_hand_q(right_hand)
                    logger.info("Read current hand joints from hardware")
                    _print_state(state, profile_path, args.step)
                elif ch in ("g", "G"):
                    state.q_live = _apply_hand(right_hand, state.q_open, wait=True)
                    logger.info("Applied hand_open slot to robot")
                    _print_state(state, profile_path, args.step)
                elif ch in ("n", "N"):
                    state.q_live = _close_with_stall(right_hand, state.q_closed)
                    logger.info("Applied hand_closed slot (stall-detect)")
                    _print_state(state, profile_path, args.step)
                elif ch in ("p", "P"):
                    _print_state(state, profile_path, args.step)
                elif ch == ">":
                    if _live_differs_from_closed(state):
                        logger.warning(
                            "live != closed slot: --apply-closed uses the closed SLOT, not live. "
                            "Press C to copy live -> closed (auto-saves), then > if needed."
                        )
                    written = save_hand_profile(state.q_open, state.q_closed, profile_path)
                    logger.info(f"Saved {written}")
                    _log_hand_vec("open", state.q_open)
                    _log_hand_vec("closed", state.q_closed)
                    _print_state(state, profile_path, args.step)
                elif ch == "<":
                    state.q_open, state.q_closed = load_hand_profile(profile_path)
                    logger.info(f"Reloaded {profile_path}")
                    _print_state(state, profile_path, args.step)
        except KeyboardInterrupt:
            logger.info("Interrupted (Ctrl+C)")
        finally:
            if not one_shot:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    finally:
        shutdown_right_hand(right_hand, manager)


if __name__ == "__main__":
    main()
