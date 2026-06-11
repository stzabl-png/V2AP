#!/usr/bin/env python3
"""Default: hand +Z axis showcase. Legacy Titan pre-grasp: pass ``--titan-pregrasp --session-id ...``."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run_titan_pregrasp() -> None:
    from demo.phase2.debug_retarget_arm_distal_pregrasp_titan import main as titan_main

    titan_main()


def main() -> None:
    if "--titan-pregrasp" in sys.argv:
        _run_titan_pregrasp()
        return
    from demo.phase2.debug_hand_z_axis_showcase import main as showcase_main

    showcase_main()


if __name__ == "__main__":
    main()
