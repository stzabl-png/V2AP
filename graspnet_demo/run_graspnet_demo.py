#!/usr/bin/env python3
"""
graspnet_demo/run_graspnet_demo.py
====================================
Main entry point for the GraspNet-Demo pipeline.

Workflow:
  1. Run GraspNet scene-level inference on RGB-D session
  2. Write V2AP-compatible candidates.json + status.json
  3. (dry-run) Print all grasp poses without connecting to robot
  4. (validate) Check format compatibility

Usage:
    # Full inference on USB session
    python -m graspnet_demo.run_graspnet_demo \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips

    # Dry-run: print poses only (no GPU/inference)
    python -m graspnet_demo.run_graspnet_demo \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips \\
        --dry-run

    # Validate existing candidates.json
    python -m graspnet_demo.run_graspnet_demo \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips \\
        --validate-only

    # Scene-level inference (recommended — no mask needed)
    python -m graspnet_demo.run_graspnet_demo \\
        --session-dir /media/lyh/KINGSTON/20260603_165343_chips \\
        --scene-level

Razor deployment:
    After running, rsync output/ to Razor and run:
        python demo/phase2/run_auto_grasp.py --session-id 20260603_165343_chips
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
from graspnet_demo.validate_candidates import validate


# ─────────────────────────────────────────────────────────────
# Dry-run: print pose summary without running inference
# ─────────────────────────────────────────────────────────────

def dry_run_poses(session_dir: Path) -> None:
    """Load existing candidates.json and print grasp/pre-grasp/lift poses."""
    print(f"\n{'='*60}")
    print(f"  GraspNet-Demo  DRY RUN")
    print(f"  Session: {session_dir.name}")
    print(f"{'='*60}")

    status = load_status(session_dir)
    if not status.get("success"):
        print(f"  ❌ status.json: success=False — run inference first")
        print(f"     errors: {status.get('errors', [])}")
        return

    cands = load_candidates(session_dir)
    T_base_mesh = np.array(cands["T_base_mesh"])  # eye(4) for GraspNet
    conv = cands.get("conventions", {})
    pre_off = conv.get("pre_grasp_offset_m", 0.15)
    lift_h  = conv.get("lift_height_m", 0.15)
    table_h = cands.get("titan", {}).get("table_height_m", 0.85)

    print(f"\n  table_height:     {table_h:.3f} m")
    print(f"  pre_grasp_offset: {pre_off:.3f} m  (retreat along -approach)")
    print(f"  lift_height:      {lift_h:.3f} m  (+Z from grasp)")
    print(f"  T_base_mesh:      {'eye(4)' if np.allclose(T_base_mesh, np.eye(4)) else 'custom'}")
    print(f"  n_candidates:     {len(cands['candidates'])}")

    print(f"\n  {'Rank':<5} {'Score':<7} {'Grasp xyz (base)':<30} {'Approach':<25} {'Pre-grasp z':<12} {'Width'}")
    print(f"  {'-'*100}")

    for c in cands["candidates"]:
        rank  = c.get("rank", -1)
        score = c.get("score", 0.0)
        gp    = np.array(c["grasp_point"])
        R     = np.array(c["rotation"])

        # V2AP retarget: T_base_pinch = T_base_mesh @ T_mesh_pinch
        # With T_base_mesh=I: pinch IS already gp in base frame
        T_mesh_pinch = np.eye(4)
        T_mesh_pinch[:3, :3] = R
        T_mesh_pinch[:3, 3]  = gp
        T_base_pinch = T_base_mesh @ T_mesh_pinch

        pinch   = T_base_pinch[:3, 3]
        approach = T_base_pinch[:3, 2]   # col 2 = approach

        pre_grasp = pinch - pre_off * approach
        lift_pos  = pinch.copy(); lift_pos[2] += lift_h
        width = c.get("gripper_width_m", None)

        print(f"  [{rank}]   {score:<7.3f} "
              f"[{pinch[0]:.3f},{pinch[1]:.3f},{pinch[2]:.3f}]          "
              f"[{approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f}]       "
              f"{pre_grasp[2]:.3f}m      "
              f"{f'{width:.3f}m' if width else '-'}")

    print(f"\n  ✅ Dry-run complete. To execute on Razor:")
    print(f"     python demo/phase2/run_auto_grasp.py \\")
    print(f"         --session-id {session_dir.name} --dry-run")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# Full inference pipeline
# ─────────────────────────────────────────────────────────────

def run_inference(session_dir: Path, args: argparse.Namespace) -> Path:
    """Run GraspNet inference and write V2AP-compatible output."""
    from s2r.graspnet_from_rgbd import process_rgbd_session

    roi = None
    if args.roi:
        roi = tuple(int(x) for x in args.roi.split(","))
        assert len(roi) == 4, "--roi must be u,v,w,h"

    if args.scene_level:
        # Scene-level: use the dedicated scene inference script
        from s2r.graspnet_scene_infer import process_scene_session
        return process_scene_session(
            session_dir=session_dir,
            device=args.device,
            n_top=args.n_top,
            max_candidates_json=args.max_candidates,
            table_height_m=args.table_height,
        )
    else:
        return process_rgbd_session(
            session_dir=session_dir,
            device=args.device,
            seg_mode=args.seg,
            roi=roi,
            mask_path=args.mask_path,
            n_top=args.n_top,
            max_candidates_json=args.max_candidates,
            table_height_m=args.table_height,
        )


# ─────────────────────────────────────────────────────────────
# Rsync helper
# ─────────────────────────────────────────────────────────────

def print_rsync_instructions(session_dir: Path, razor_host: str = "razor") -> None:
    sid = session_dir.name
    print(f"\n  📡 To deploy on Razor:")
    print(f"     rsync -avz {session_dir}/output/ \\")
    print(f"         {razor_host}:~/V2AP-demo/demo/phase2/sessions/{sid}/output/")
    print(f"\n     # Then on Razor:")
    print(f"     python demo/phase2/run_auto_grasp.py \\")
    print(f"         --session-id {sid} --dry-run    # verify first")
    print(f"     python demo/phase2/run_auto_grasp.py \\")
    print(f"         --session-id {sid}              # execute")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GraspNet-Demo: RGB-D inference → V2AP-compatible output"
    )
    p.add_argument("--session-dir", type=str, required=True,
                   help="Session directory (absolute path or session ID)")
    p.add_argument("--device", type=str, default="cuda")

    # Inference mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print poses from existing candidates.json (no inference)")
    mode.add_argument("--validate-only", action="store_true",
                      help="Validate candidates.json format only")
    mode.add_argument("--scene-level", action="store_true",
                      help="Use scene-level GraspNet (full workspace point cloud, recommended)")

    # Segmentation (for object-level mode)
    p.add_argument("--seg", choices=["color", "roi", "mask", "center"], default="color")
    p.add_argument("--roi", type=str, default=None, help="u,v,w,h pixels")
    p.add_argument("--mask-path", type=str, default=None)

    # Inference params
    p.add_argument("--n-top", type=int, default=50)
    p.add_argument("--max-candidates", type=int, default=10)
    p.add_argument("--table-height", type=float, default=None)

    # Output
    p.add_argument("--razor-host", type=str, default="razor",
                   help="Razor hostname for rsync instructions")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip post-inference validation")

    args = p.parse_args()

    # Resolve session directory
    try:
        session_dir = session_dir_for_id(args.session_dir)
    except FileNotFoundError:
        session_dir = Path(args.session_dir)

    print(f"  Session directory: {session_dir}")

    # ── Dry-run ────────────────────────────────────────
    if args.dry_run:
        dry_run_poses(session_dir)
        return

    # ── Validate only ──────────────────────────────────
    if args.validate_only:
        ok = validate(session_dir)
        sys.exit(0 if ok else 1)

    # ── Full inference ─────────────────────────────────
    out_path = run_inference(session_dir, args)
    print(f"\n  Output: {out_path}")

    # Post-inference validation
    if not args.no_validate:
        print()
        validate(session_dir)

    # Rsync instructions
    print_rsync_instructions(session_dir, args.razor_host)


if __name__ == "__main__":
    main()
