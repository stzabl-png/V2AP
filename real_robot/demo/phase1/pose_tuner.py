"""Interactive joint tuning for Phase 1 grasp configs (run on Razor at the robot).

If demo/phase1/configs/<object>.yaml exists: load it and move to the saved grasp pose.
Otherwise: capture the robot state at startup (no motion) and create a new config on save.
Tunes right_arm_joint_pos in real time (no IK). grasp_pose in YAML is updated via FK on save.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.hand_close import move_hand_to_target, preview_stall_close_then_open
from demo.phase1.config_io import (
    GraspObjectConfig,
    grasp_config_path,
    load_grasp_config,
    load_hand_profile,
    save_grasp_config,
)
from demo.phase1.constants import (
    DEFAULT_START_OBJECT_NAME,
    DEFAULT_TABLE_HEIGHT_M,
    PHASE1_CONFIG_DIR,
    TUNER_ARM_JOINT_STEP_RAD,
)
from demo.phase1.executor import GraspExecutor
from demo.phase1.grasp_geometry import format_pose
from demo.hand_close import move_hand_to_target, preview_stall_close_then_open

_ESC = "ESC"
_ESCAPE_FOLLOWUP_TIMEOUT_S = 0.02

RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
)


@dataclass
class PoseTunerState:
    joint_index: int = 0


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


def _print_state(config: GraspObjectConfig, config_path: Path, tuner: PoseTunerState) -> None:
    j = tuner.joint_index
    print("\n" + "=" * 72)
    print(f"object: {config.object_name}  ->  {config_path}")
    print(
        f"selected: [{j}] {RIGHT_ARM_JOINT_NAMES[j]} = "
        f"{config.right_arm_joint_pos[j]:+.4f} rad"
    )
    print(f"right_arm: {np.round(config.right_arm_joint_pos, 3).tolist()}")
    print(format_pose(config.grasp_pose, "grasp (FK from joints)"))
    print(format_pose(config.resolved_pre_grasp_pose(), "pre_grasp (derived)"))
    print(format_pose(config.resolved_lift_pose(), "lift (derived)"))
    print("=" * 72 + "\n")


def _open_grip_for_init(executor: GraspExecutor, config: GraspObjectConfig) -> None:
    assert executor.handles is not None
    assert executor.hardware_lock is not None
    with executor.hardware_lock:
        move_hand_to_target(executor.handles.right_hand, config.hand_open_joint_pos)
    logger.info("Right hand at open grip (stepped, no SDK palm-spread path)")


def _read_hand_joint_pos(executor: GraspExecutor) -> tuple[np.ndarray, np.ndarray]:
    assert executor.handles is not None
    assert executor.hardware_lock is not None
    with executor.hardware_lock:
        left_q = np.array(executor.handles.left_hand.get_states().angles, dtype=np.float64)
        right_q = np.array(executor.handles.right_hand.get_states().angles, dtype=np.float64)
    return left_q, right_q


def _move_to_grasp_joints(
    executor: GraspExecutor,
    config: GraspObjectConfig,
    *,
    keep_current_hands: bool = False,
) -> None:
    if keep_current_hands:
        left_hand, right_hand = _read_hand_joint_pos(executor)
    else:
        left_hand = config.left_hand_joint_pos
        right_hand = config.hand_open_joint_pos
    executor._move_to_joint_targets(
        config,
        {
            "left_arm": config.left_arm_joint_pos,
            "right_arm": config.right_arm_joint_pos,
            "left_hand": left_hand,
            "right_hand": right_hand,
        },
    )


def _print_help(*, existing_config: bool) -> None:
    startup = (
        "Loads saved YAML, steps hand to open grip, then moves arms to grasp pose."
        if existing_config
        else "New object: position robot at grasp pose first; startup captures without moving."
    )
    print(
        f"""
Keys (no Enter) — right arm only, applied live:

  Joint select:            1-7 = R_arm_j1 .. R_arm_j7
  Nudge selected joint:    W = +   S = -
  Hand (shared profile):   [ = open   ] = close (stall/target), hold 1.5s, reopen
  Read arm from hardware:  R
  Print:                   P
  Save / reload YAML:      > = save   < = reload file (does not move arm)
  Pre-grasp (slow IK):     N   (preview only; does not change saved grasp joints)
  Back to grasp (slow):    G
  Full pick sequence:      X
  Quit:                    Esc

Tune hand joints: python demo/hand_tuner.py -> demo/right_hand_profile.yaml
{startup}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune right_arm joints for Phase 1 grasp. "
            "Loads existing YAML if present; otherwise captures current robot pose."
        )
    )
    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help="Object name; saves to demo/phase1/configs/<object-name>.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(PHASE1_CONFIG_DIR),
        help="Directory for grasp YAML configs",
    )
    parser.add_argument(
        "--table-height",
        type=float,
        default=DEFAULT_TABLE_HEIGHT_M,
        help="Table height for collision planning (meters)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=TUNER_ARM_JOINT_STEP_RAD,
        help="Per-keypress joint step in radians",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=None,
        help="Start with this right-arm joint index selected (0-6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify imports only; does not connect to robot",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("pose_tuner dry-run: imports OK (no hardware connection)")
        return

    config_dir = Path(args.config_dir)
    config_path = grasp_config_path(args.object_name, config_dir)
    existing_config = config_path.is_file()

    executor = GraspExecutor(dry_run=False)
    executor.setup()

    if existing_config:
        config = load_grasp_config(config_path)
        config.hand_open_joint_pos, config.hand_closed_joint_pos = load_hand_profile()
        logger.info(f"Loaded {config_path}; opening grip then moving arms...")
        _open_grip_for_init(executor, config)
        _move_to_grasp_joints(executor, config, keep_current_hands=True)
        logger.info(
            "At saved grasp pose. right_arm="
            f"{np.round(config.right_arm_joint_pos, 3).tolist()}"
        )
    else:
        config = executor.capture_config_from_robot(
            object_name=args.object_name,
            table_height=args.table_height,
        )
        config.hand_open_joint_pos, config.hand_closed_joint_pos = load_hand_profile()
        executor.sync_grasp_pose_from_right_arm(config)
        logger.info(
            f"No config at {config_path}; captured current pose (no motion). right_arm="
            f"{np.round(config.right_arm_joint_pos, 3).tolist()}"
        )

    tuner = PoseTunerState()
    if args.joint is not None:
        if not 0 <= args.joint < len(RIGHT_ARM_JOINT_NAMES):
            raise ValueError(f"--joint must be 0-6, got {args.joint}")
        tuner.joint_index = args.joint

    _print_help(existing_config=existing_config)
    _print_state(config, config_path, tuner)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def apply_arm_live() -> None:
        executor.sync_grasp_pose_from_right_arm(config)
        executor.apply_live_targets(config)

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
            if ch in "1234567":
                tuner.joint_index = int(ch) - 1
                _print_state(config, config_path, tuner)
            elif ch in ("w", "W"):
                config.right_arm_joint_pos[tuner.joint_index] += args.step
                apply_arm_live()
                _print_state(config, config_path, tuner)
            elif ch in ("s", "S"):
                config.right_arm_joint_pos[tuner.joint_index] -= args.step
                apply_arm_live()
                _print_state(config, config_path, tuner)
            elif ch in ("r", "R"):
                _, right_q = executor._read_arm_joint_pos()
                config.right_arm_joint_pos = right_q.copy()
                executor.sync_grasp_pose_from_right_arm(config)
                logger.info("Read right_arm from hardware")
                _print_state(config, config_path, tuner)
            elif ch == "[":
                config.hand_open_joint_pos = load_hand_profile()[0]
                assert executor.handles is not None
                assert executor.hardware_lock is not None
                with executor.hardware_lock:
                    move_hand_to_target(
                        executor.handles.right_hand, config.hand_open_joint_pos
                    )
                apply_arm_live()
                _print_state(config, config_path, tuner)
            elif ch == "]":
                config.hand_closed_joint_pos = load_hand_profile()[1]
                config.hand_open_joint_pos = load_hand_profile()[0]
                assert executor.handles is not None
                assert executor.hardware_lock is not None
                with executor.hardware_lock:
                    result = preview_stall_close_then_open(
                        executor.handles.right_hand,
                        config.hand_closed_joint_pos,
                        config.hand_open_joint_pos,
                    )
                logger.info(f"Preview close: reason={result.reason}, steps={result.steps}")
                apply_arm_live()
                _print_state(config, config_path, tuner)
            elif ch in ("p", "P"):
                _print_state(config, config_path, tuner)
            elif ch == ">":
                executor.sync_grasp_pose_from_right_arm(config)
                save_grasp_config(config, config_path)
                logger.info(f"Saved {config_path}")
                _print_state(config, config_path, tuner)
            elif ch == "<":
                if config_path.exists():
                    config = load_grasp_config(config_path)
                    config.hand_open_joint_pos, config.hand_closed_joint_pos = (
                        load_hand_profile()
                    )
                    executor.sync_grasp_pose_from_right_arm(config)
                    logger.info(f"Reloaded {config_path} (arm not moved; use W/S or R)")
                    _print_state(config, config_path, tuner)
                else:
                    logger.warning(f"No saved file yet: {config_path}")
            elif ch in ("n", "N"):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                try:
                    logger.info("Moving to pre_grasp via IK (slow)...")
                    executor._move_to_right_ee_pose(
                        config,
                        config.resolved_pre_grasp_pose(),
                        config.hand_open_joint_pos,
                    )
                    logger.info("At pre_grasp. Press G to return to grasp joints.")
                finally:
                    tty.setcbreak(fd)
                _print_state(config, config_path, tuner)
            elif ch in ("g", "G"):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                try:
                    logger.info("Returning to grasp joints (slow)...")
                    _move_to_grasp_joints(executor, config)
                finally:
                    tty.setcbreak(fd)
                _print_state(config, config_path, tuner)
            elif ch in ("x", "X"):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                try:
                    executor.sync_grasp_pose_from_right_arm(config)
                    start_path = grasp_config_path(DEFAULT_START_OBJECT_NAME, config_dir)
                    start_cfg = (
                        load_grasp_config(start_path) if start_path.is_file() else None
                    )
                    executor.run_sequence(config, start_config=start_cfg)
                finally:
                    tty.setcbreak(fd)
    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C)")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        executor.shutdown()


if __name__ == "__main__":
    main()
