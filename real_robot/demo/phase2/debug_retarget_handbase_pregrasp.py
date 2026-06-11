#!/usr/bin/env python3
"""Deprecated alias → hand +Z axis showcase (``debug_hand_z_axis_showcase.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase2.debug_retarget_arm_distal_pregrasp import main

if __name__ == "__main__":
    main()
