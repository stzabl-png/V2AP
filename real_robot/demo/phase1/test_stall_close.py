#!/usr/bin/env python3
"""Test stall-detecting pinch close on the right hand.

Opens from profile, waits for you to place an object, then closes with stall detection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import load_hand_profile
from demo.constants import DEFAULT_HAND_PROFILE_PATH, HAND_APPLY_WAIT_S
from demo.hand_close import close_hand_until_stall, read_hand_joint_pos
from demo.hardware import (
    connect_right_hand_only,
    set_hand_joint_positions,
    setup_cpp_logging,
    shutdown_right_hand,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test stall-detecting hand close")
    parser.add_argument(
        "--profile",
        type=str,
        default=str(DEFAULT_HAND_PROFILE_PATH),
        help="Hand profile YAML (open/closed targets)",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=3.0,
        help="Seconds to place object after open before closing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log only; no hardware",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile)
    q_open, q_closed = load_hand_profile(profile_path)
    logger.info(f"Profile: {profile_path}")
    logger.info(f"open  thumb/index: {q_open[:9].round(3).tolist()}")
    logger.info(f"closed thumb/index: {q_closed[:9].round(3).tolist()}")

    if args.dry_run:
        result = close_hand_until_stall(None, q_closed, q_start=q_open, dry_run=True)
        logger.info(f"dry-run result: reason={result.reason}")
        return

    setup_cpp_logging(console_log=False)
    right_hand, manager = connect_right_hand_only()
    try:
        logger.info("Opening hand...")
        set_hand_joint_positions(right_hand, q_open)
        time.sleep(HAND_APPLY_WAIT_S)

        logger.info(
            f"Place object between thumb and index. Closing in {args.delay_s:.0f}s..."
        )
        time.sleep(args.delay_s)

        q_before = read_hand_joint_pos(right_hand)
        logger.info(f"Close start thumb/index: {q_before[:9].round(3).tolist()}")

        result = close_hand_until_stall(right_hand, q_closed, q_start=q_before)
        q_final = result.q_final

        pinch_gap = float(
            np.max(np.abs(q_final[:9] - q_closed[:9]))
        )
        logger.info(f"Done: reason={result.reason}, steps={result.steps}")
        logger.info(f"Final  thumb/index: {q_final[:9].round(3).tolist()}")
        logger.info(f"Target thumb/index: {q_closed[:9].round(3).tolist()}")
        logger.info(f"Pinch joint gap to profile closed: {pinch_gap:.4f} rad")

        input("Press Enter to open hand and exit...")
        set_hand_joint_positions(right_hand, q_open)
        time.sleep(HAND_APPLY_WAIT_S)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        shutdown_right_hand(right_hand, manager)


if __name__ == "__main__":
    main()
