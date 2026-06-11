#!/usr/bin/env python3
"""Estimate table top height (m) in robot ``base`` frame from a captured session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase2.constants import SESSIONS_DIR  # noqa: E402
from demo.phase2.table_height import estimate_table_height_m_from_session  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backproject table ROI depth → median z in base frame"
    )
    p.add_argument("--session-id", type=str, required=True)
    p.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    args = p.parse_args()

    session_dir = args.sessions_dir / args.session_id
    table_h, source = estimate_table_height_m_from_session(session_dir)
    print(f"table_height_m={table_h:.3f}  source={source}")
    table_json = session_dir / "input" / "scene" / "table.json"
    if table_json.is_file():
        import json

        recorded = float(json.loads(table_json.read_text()).get("table_height_m", float("nan")))
        if recorded == recorded:
            print(f"scene/table.json (not used by run_auto_grasp): {recorded:.3f} m")


if __name__ == "__main__":
    main()
