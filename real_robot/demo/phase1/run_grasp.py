#!/usr/bin/env python3
"""Phase 1: scripted pick-and-lift demo (hard-coded / YAML grasp configs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase1.config_io import grasp_config_path, load_grasp_config
from demo.phase1.constants import DEFAULT_START_OBJECT_NAME, PHASE1_CONFIG_DIR
from demo.phase1.executor import GraspExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="V2AP Phase 1 grasp demo")
    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help="Object name (loads demo/phase1/configs/<name>.yaml)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(PHASE1_CONFIG_DIR),
        help="Directory containing object YAML configs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print IK/plan only; no robot connection (works off-robot)",
    )
    parser.add_argument(
        "--start-object-name",
        type=str,
        default=DEFAULT_START_OBJECT_NAME,
        help=f"Start pose object name (default: {DEFAULT_START_OBJECT_NAME})",
    )
    parser.add_argument(
        "--skip-home-at-end",
        action="store_true",
        help="Stay at lift after hold; skip grasp/open/remove/start teardown",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    config_path = grasp_config_path(args.object_name, config_dir)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No grasp config for object '{args.object_name}' at {config_path}. "
            f"Tune with: python demo/phase1/pose_tuner.py --object-name {args.object_name}"
        )

    config = load_grasp_config(config_path)
    logger.info(f"Object config: {config_path}")

    start_path = grasp_config_path(args.start_object_name, config_dir)
    start_config = None
    if start_path.is_file():
        start_config = load_grasp_config(start_path)
        logger.info(f"Start pose: {start_path}")
    else:
        logger.warning(
            f"No start config for '{args.start_object_name}' at {start_path}; "
            f"sequence begins from object grasp joints. "
            f"Tune with: python demo/phase1/pose_tuner.py --object-name {args.start_object_name}"
        )

    executor = GraspExecutor(dry_run=args.dry_run)
    try:
        executor.setup()
        executor.run_sequence(
            config,
            start_config=start_config,
            skip_home_at_end=args.skip_home_at_end,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
