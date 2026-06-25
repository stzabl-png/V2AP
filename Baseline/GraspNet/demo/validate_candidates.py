#!/usr/bin/env python3
"""
graspnet_demo/validate_candidates.py
=====================================
Validate that a candidates.json is compatible with V2AP run_auto_grasp.py.

Usage:
    python -m graspnet_demo.validate_candidates \
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from graspnet_demo.session_io import session_dir_for_id, load_candidates, load_status


def validate(session_dir: Path, verbose: bool = True) -> bool:
    """Returns True if candidates.json is V2AP-compatible."""
    ok = True
    errs = []
    warns = []

    def _check(cond, msg, fatal=True):
        nonlocal ok
        if not cond:
            if fatal:
                errs.append(f"❌  {msg}")
                ok = False
            else:
                warns.append(f"⚠️   {msg}")

    print(f"\n{'='*55}")
    print(f"  Validating session: {session_dir.name}")
    print(f"{'='*55}")

    # ── status.json ──────────────────────────────────────
    try:
        status = load_status(session_dir)
        _check(status.get("success") is True, "status.json: success must be True")
        _check("titan" in status, "status.json: missing 'titan' block (V2AP reads titan.object_slug)", fatal=False)
        if "titan" in status:
            _check("object_slug" in status["titan"], "status.json: titan.object_slug missing", fatal=False)
        print(f"  status.json: success={status.get('success')}")
    except FileNotFoundError as e:
        errs.append(f"❌  {e}")
        ok = False

    # ── candidates.json ───────────────────────────────────
    try:
        cands = load_candidates(session_dir)
    except FileNotFoundError as e:
        errs.append(f"❌  {e}")
        print("\n".join(errs))
        return False

    # Schema version
    _check(cands.get("schema_version") == "1.1", f"schema_version must be '1.1', got {cands.get('schema_version')!r}")

    # T_base_mesh (critical for V2AP retarget.py)
    _check("T_base_mesh" in cands, "Missing top-level 'T_base_mesh'")
    if "T_base_mesh" in cands:
        Tb = np.array(cands["T_base_mesh"])
        _check(Tb.shape == (4, 4), f"T_base_mesh must be 4x4, got {Tb.shape}")

    # conventions
    conv = cands.get("conventions", {})
    _check("rotation_columns" in conv,
           "conventions.rotation_columns missing (V2AP uses col index 2 as approach)", fatal=False)
    _check(conv.get("approach_column_index") == 2,
           f"conventions.approach_column_index should be 2, got {conv.get('approach_column_index')}", fatal=False)
    _check("pre_grasp_offset_m" in conv, "conventions.pre_grasp_offset_m missing", fatal=False)

    # candidates list
    cand_list = cands.get("candidates", [])
    _check(len(cand_list) > 0, "candidates list is empty")

    for i, c in enumerate(cand_list):
        prefix = f"candidates[{i}]"
        _check("rank" in c, f"{prefix}: missing 'rank'")
        _check("grasp_point" in c, f"{prefix}: missing 'grasp_point'")
        _check("rotation" in c, f"{prefix}: missing 'rotation'")
        _check("score" in c, f"{prefix}: missing 'score'", fatal=False)

        if "grasp_point" in c:
            gp = np.array(c["grasp_point"])
            _check(gp.shape == (3,), f"{prefix}: grasp_point must be (3,), got {gp.shape}")

        if "rotation" in c:
            R = np.array(c["rotation"])
            _check(R.shape == (3, 3), f"{prefix}: rotation must be 3x3, got {R.shape}")
            if R.shape == (3, 3):
                det = np.linalg.det(R)
                _check(abs(det - 1.0) < 0.01, f"{prefix}: rotation det={det:.4f} (expected ~1.0)")

    # Print top candidates summary
    print(f"\n  n_candidates: {len(cand_list)}")
    T_base_mesh = np.array(cands.get("T_base_mesh", np.eye(4)))
    is_identity = np.allclose(T_base_mesh, np.eye(4))
    print(f"  T_base_mesh:  {'eye(4) — grasp_point already in base frame ✓' if is_identity else T_base_mesh.tolist()}")

    print(f"\n  Top candidates:")
    for c in cand_list[:5]:
        gp = c.get("grasp_point", [0, 0, 0])
        R = np.array(c.get("rotation", np.eye(3)))
        approach = R[:, 2] if R.shape == (3, 3) else [0, 0, 0]
        print(f"    rank={c.get('rank','-')} score={c.get('score',0):.3f} "
              f"pt=[{gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}] "
              f"approach=[{approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f}]")

    # Pre-grasp sanity (retreat 15cm along -approach)
    print(f"\n  Pre-grasp positions (retreat 0.15m along -approach):")
    for c in cand_list[:3]:
        gp = np.array(c.get("grasp_point", [0, 0, 0]))
        R = np.array(c.get("rotation", np.eye(3)))
        approach = R[:, 2] if R.shape == (3, 3) else np.array([0, 0, 1])
        pre = gp - 0.15 * approach
        print(f"    rank={c.get('rank','-')} pre_grasp=[{pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f}]")

    # Results
    print()
    for w in warns:
        print(f"  {w}")
    for e in errs:
        print(f"  {e}")

    if ok:
        print(f"\n  ✅ V2AP-compatible! Ready for run_auto_grasp.py --session-id {session_dir.name}")
    else:
        print(f"\n  ❌ Validation FAILED. Fix errors above before running on Razor.")

    print(f"{'='*55}\n")
    return ok


def main():
    p = argparse.ArgumentParser(description="Validate candidates.json for V2AP compatibility")
    p.add_argument("--session-dir", type=str, required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        sd = session_dir_for_id(args.session_dir)
    except FileNotFoundError:
        sd = Path(args.session_dir)

    ok = validate(sd, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
