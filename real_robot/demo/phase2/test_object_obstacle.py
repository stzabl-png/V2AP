"""Tests for Titan object collision box → base frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from demo.phase2.object_obstacle import object_collision_box
from demo.phase2.retarget import load_titan_output


def test_chips_session_box_uses_full_height_and_registration_z() -> None:
    session = Path(__file__).resolve().parent / "sessions" / "20260602_192346_chips"
    if not (session / "output" / "inference" / "candidates.json").is_file():
        return

    output = load_titan_output(session)
    box = object_collision_box(output, padding=1.08)
    pos = np.asarray(box["position"], dtype=np.float64)
    ext = np.asarray(box["full_extents"], dtype=np.float64)
    mesh_z = float(output.T_base_mesh[2, 3])
    span_z = float(output.mesh_span_m[2]) * 1.08

    np.testing.assert_allclose(ext[2], span_z, rtol=1e-5)
    np.testing.assert_allclose(pos[2], mesh_z, rtol=0.02)
    np.testing.assert_allclose(pos[0], output.T_base_mesh[0, 3], rtol=0.02)
    np.testing.assert_allclose(pos[1], output.T_base_mesh[1, 3], rtol=0.02)

    lo_z = pos[2] - 0.5 * ext[2]
    hi_z = pos[2] + 0.5 * ext[2]
    assert lo_z < mesh_z < hi_z
    assert ext[2] > 0.25  # no 12 cm clip
