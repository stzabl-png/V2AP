#!/usr/bin/env python3
"""
Quick head-camera check: does the robot answer sensors/head_camera/info?

Same Zenoh query that logs:
  Failed to query dm/<robot>/sensors/head_camera/info after 2 attempts

Usage:
  source setup.sh
  python camera/diagnose_head_camera.py
  python camera/diagnose_head_camera.py --timeout 5 --retries 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query robot head_camera/info over Zenoh (dexsensor health check)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Seconds to wait per query attempt (default: 2)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Extra retries after first attempt (default: 1 → 2 attempts total)",
    )
    args = parser.parse_args()

    from dexcontrol.utils.constants import COMM_CFG_PATH_ENV_VAR, ROBOT_NAME_ENV_VAR
    from dexcontrol.utils.comm_helper import query_json_service
    from dexcontrol.utils.os_utils import resolve_key_name

    robot_name = os.getenv(ROBOT_NAME_ENV_VAR)
    zenoh_cfg = os.getenv(COMM_CFG_PATH_ENV_VAR)

    print("Head camera info query")
    print(f"  {ROBOT_NAME_ENV_VAR}={robot_name or '(unset)'}")
    print(f"  {COMM_CFG_PATH_ENV_VAR}={zenoh_cfg or '(unset)'}")

    if not robot_name:
        print(f"\nFAIL: set {ROBOT_NAME_ENV_VAR} (source setup.sh)")
        return 1
    if not zenoh_cfg or not Path(zenoh_cfg).expanduser().is_file():
        print(f"\nFAIL: {COMM_CFG_PATH_ENV_VAR} must point to an existing zenoh json5")
        return 1

    topic = "sensors/head_camera/info"
    resolved = resolve_key_name(topic)
    print(f"\nQuerying: {resolved}")

    info = query_json_service(
        topic,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    if info is None:
        print(
            f"\nFAIL: no response from {resolved}\n"
            "\nMeaning: robot dexsensor head_camera service is not answering.\n"
            "Arms can still work; head ZED stack is down on the robot.\n"
            "\nAsk lab to:\n"
            "  • reboot robot / restart dexsensor\n"
            "  • check head ZED cable + power\n"
            "  • confirm this query works on a known-good laptop\n"
        )
        return 1

    print("\nOK: head_camera/info responded")
    status = info.get("status", "?")
    model = info.get("model") or info.get("camera_id") or "?"
    streams = info.get("streams")
    print(f"  status: {status}")
    print(f"  model/id: {model}")
    if streams is not None:
        print(f"  streams: {streams}")

    actual = info.get("actual")
    if isinstance(actual, dict):
        w, h = actual.get("width"), actual.get("height")
        if w and h:
            print(f"  resolution: {w}x{h}")

    print("\nFull JSON:")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("\nIf info is OK but view_head_camera still times out, streams may not be publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
