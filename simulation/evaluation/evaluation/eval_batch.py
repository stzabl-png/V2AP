#!/usr/bin/env python3
"""Batch evaluation wrapper: one subprocess per episode (restarts Isaac Sim each time).

Example (pool HDF5, 5 random trials per object, no video):

    export ISAAC_SIM_PATH=/home/vision/isaacsim
    python evaluation/eval_batch.py \\
        --candidate-dir output/grasp_collect_no_rot/candidates/pool \\
        --trials-per-object 5 \\
        --selection sample \\
        --headless \\
        --result-dir output/evaluation/batch_pool

Example (record 2 random trials per object out of 5):

    python evaluation/eval_batch.py \\
        --candidate-dir output/grasp_collect_no_rot/candidates/pool \\
        --trials-per-object 5 \\
        --record-video \\
        --record-count-per-object 2 \\
        --selection sample
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import add_affordance_checkpoint_args, resolve_affordance_checkpoint
from evaluation.episode import (
    build_episode_id,
    default_candidate_hdf5,
    discover_obj_ids,
    pick_record_trials,
)
from evaluation.placement import add_random_obj_xy_args
from evaluation.randomness import add_eval_seed_args, shuffle_objects_rng, record_trials_rng
from evaluation.yaw import parse_yaw_pool, resolve_z_yaw_deg


def isaac_python() -> str:
    root = os.environ.get("ISAAC_SIM_PATH", "").strip()
    if not root:
        raise RuntimeError("Set ISAAC_SIM_PATH to your Isaac Sim install (contains python.sh)")
    path = os.path.join(root, "python.sh")
    if not os.path.isfile(path):
        raise RuntimeError(f"Isaac python not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch evaluation via eval_single subprocess wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--usd-root", default=None, help="Directory or .usd file to discover obj ids")
    p.add_argument(
        "--candidate-dir",
        default=None,
        help="Directory of *_grasp.hdf5 (used for obj discovery and default candidate paths)",
    )
    p.add_argument("--obj", action="append", default=None, help="Explicit object id (repeatable)")
    p.add_argument("--obj-list", default=None, help="Text file with one obj id per line")
    p.add_argument("--obj-limit", type=int, default=None, help="Cap number of objects (after discovery)")
    p.add_argument("--trials-per-object", type=int, default=1)
    p.add_argument("--policy", choices=("a2g_pdm",), default="a2g_pdm")
    p.add_argument("--selection", choices=("top", "index", "sample"), default="sample")
    p.add_argument("--candidate-index", type=int, default=0)
    add_eval_seed_args(p)
    p.add_argument("--z-yaw-deg", type=float, default=None, help="Fixed sim/PDM z-yaw for all episodes")
    p.add_argument(
        "--z-yaw-grid",
        default=None,
        help="Comma-separated yaws; cycles with trial index, e.g. 0,90,180,270",
    )
    p.add_argument(
        "--z-yaw-random",
        action="store_true",
        help="Sample z-yaw per trial from --z-yaw-pool",
    )
    p.add_argument("--z-yaw-pool", default="0,90,180,270")
    p.add_argument(
        "--generate-candidate-each-trial",
        action="store_true",
        help="Run glb_to_pdm_grasp before each trial (needs --mesh-root)",
    )
    p.add_argument(
        "--mesh-root",
        default=None,
        help="Directory of meshes for --generate-candidate-each-trial (obj_id.glb or stem match)",
    )
    p.add_argument("--mesh-glob", default="*.glb")
    p.add_argument("--dataset", default=None)
    p.add_argument("--object-scale", type=float, default=1.0)
    add_random_obj_xy_args(p)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--result-dir",
        default=str(PROJ / "output" / "evaluation" / "batch"),
    )
    p.add_argument("--save-hdf5", action="store_true")
    p.add_argument("--record-video", action="store_true", help="Enable recording on selected trials")
    p.add_argument(
        "--record-count-per-object",
        type=int,
        default=None,
        help="Record this many random trials per object; default=all trials when --record-video",
    )
    p.add_argument("--record-fps", type=int, default=30)
    p.add_argument("--record-every", type=int, default=3)
    p.add_argument("--record-keep-frames", action="store_true")
    p.add_argument(
        "--log-only",
        action="store_true",
        help="Do not print episode progress; write full child logs under <result-dir>/logs.",
    )
    p.add_argument(
        "--loud",
        action="store_true",
        help="Print full child stdout/stderr and also write full logs under <result-dir>/logs.",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory for child logs (default: <result-dir>/logs).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument(
        "--shuffle-objects",
        action="store_true",
        help="Randomize object order (fresh RNG, no fixed seed)",
    )
    add_affordance_checkpoint_args(p)
    return p


def find_mesh_for_obj(mesh_root: Path, obj_id: str, glob_pat: str) -> Path | None:
    direct = mesh_root / f"{obj_id}.glb"
    if direct.is_file():
        return direct
    nested = mesh_root / obj_id / "mesh.ply"
    if nested.is_file():
        return nested
    for ds in ("unseen", "oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3"):
        nested = mesh_root / ds / obj_id / "mesh.ply"
        if nested.is_file():
            return nested
    for p in mesh_root.glob(glob_pat):
        if p.stem == obj_id:
            return p
    for p in mesh_root.rglob("mesh.ply"):
        if p.parent.name == obj_id:
            return p
    return None


def build_eval_single_cmd(
    args: argparse.Namespace,
    *,
    obj_id: str,
    trial: int,
    z_yaw_deg: float,
    episode_id: str,
    candidate_hdf5: str | None,
    record_video: bool,
    isaac_py: str,
) -> list[str]:
    cmd = [
        isaac_py,
        str(PROJ / "evaluation" / "eval_single.py"),
        "--obj-id",
        obj_id,
        "--policy",
        args.policy,
        "--trial",
        str(trial),
        "--z-yaw-deg",
        str(float(z_yaw_deg)),
        "--episode-id",
        episode_id,
        "--selection",
        args.selection,
        "--candidate-index",
        str(args.candidate_index),
        "--object-scale",
        str(args.object_scale),
        "--result-dir",
        str(Path(args.result_dir).expanduser().resolve()),
        "--seed",
        str(int(args.seed)),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.headless and not record_video:
        cmd.append("--headless")
    if args.save_hdf5:
        cmd.append("--save-hdf5")
    if record_video:
        cmd.append("--record-video")
        cmd.extend(["--record-fps", str(args.record_fps)])
        cmd.extend(["--record-every", str(args.record_every)])
        if args.record_keep_frames:
            cmd.append("--record-keep-frames")
    if args.policy_seed is not None:
        cmd.extend(["--policy-seed", str(args.policy_seed)])
    if args.log_only:
        cmd.append("--log-only")
    if args.loud:
        cmd.append("--loud")
    if args.random_obj_xy:
        cmd.append("--random-obj-xy")
        cmd.extend(["--obj-xy-jitter-m", str(float(args.obj_xy_jitter_m))])
    if args.log_dir:
        cmd.extend(["--log-dir", str(Path(args.log_dir).expanduser().resolve())])

    if args.generate_candidate_each_trial:
        if not args.mesh_root:
            raise ValueError("--generate-candidate-each-trial requires --mesh-root")
        mesh = find_mesh_for_obj(Path(args.mesh_root).expanduser(), obj_id, args.mesh_glob)
        if mesh is None:
            raise FileNotFoundError(f"no mesh for {obj_id} under {args.mesh_root}")
        cand_dir = Path(args.result_dir) / "candidates" / obj_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand_out = cand_dir / f"trial_{trial:03d}_grasp.hdf5"
        cmd.extend(
            [
                "--generate-candidate",
                "--mesh",
                str(mesh.resolve()),
                "--candidate-dir",
                str(cand_dir.resolve()),
                "--candidate-output",
                str(cand_out.resolve()),
            ]
        )
        if mesh.name == "mesh.ply" and mesh.parent.name == obj_id:
            cmd.append("--sam3d-rotated-mesh")
        if getattr(args, "affordance_checkpoint", None):
            cmd.extend(["--affordance-checkpoint", str(args.affordance_checkpoint)])
        if getattr(args, "hp_affordance", False):
            cmd.append("--hp-affordance")
    elif candidate_hdf5:
        cmd.extend(["--candidate-hdf5", candidate_hdf5])
    else:
        raise ValueError(f"no candidate HDF5 for {obj_id}; set --candidate-dir or --generate-candidate-each-trial")

    return cmd


def main() -> None:
    args = build_parser().parse_args()
    if args.log_only and args.loud:
        raise SystemExit("Use only one of --log-only or --loud")

    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(args.hp_affordance),
        affordance_checkpoint=args.affordance_checkpoint,
    )
    args.affordance_checkpoint = str(aff_ckpt)

    def important(message: str = "") -> None:
        if not args.log_only:
            print(message)

    yaw_grid = None
    if args.z_yaw_grid:
        yaw_grid = parse_yaw_pool(args.z_yaw_grid)
    yaw_pool = parse_yaw_pool(args.z_yaw_pool) if args.z_yaw_random else None

    obj_ids = discover_obj_ids(
        usd_root=args.usd_root,
        candidate_dir=args.candidate_dir,
        obj_ids=args.obj,
        obj_list_file=args.obj_list,
    )
    if not obj_ids:
        raise SystemExit("No objects discovered; pass --usd-root, --candidate-dir, or --obj")
    if args.obj_limit is not None:
        obj_ids = obj_ids[: max(0, int(args.obj_limit))]

    rng = shuffle_objects_rng(eval_seed=int(args.seed))
    if args.shuffle_objects:
        order = rng.permutation(len(obj_ids))
        obj_ids = [obj_ids[i] for i in order]

    candidate_dir = Path(args.candidate_dir).expanduser() if args.candidate_dir else None
    isaac_py = isaac_python()
    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    summary_rows: list[dict] = []

    for obj_id in obj_ids:
        record_cap = None if args.record_video else 0
        if args.record_video:
            record_cap = args.record_count_per_object
        record_trials = pick_record_trials(
            args.trials_per_object,
            record_cap,
            record_trials_rng(eval_seed=int(args.seed), obj_id=obj_id),
        )
        for trial in range(args.trials_per_object):
            z_yaw = resolve_z_yaw_deg(
                trial=trial,
                obj_id=obj_id,
                z_yaw_deg=args.z_yaw_deg,
                z_yaw_grid=yaw_grid,
                z_yaw_random_pool=yaw_pool,
                z_yaw_random=bool(args.z_yaw_random),
                eval_seed=int(args.seed),
            )
            episode_id = build_episode_id(obj_id, args.policy, trial, z_yaw)
            cand_path = None
            if candidate_dir and not args.generate_candidate_each_trial:
                p = default_candidate_hdf5(candidate_dir, obj_id, z_yaw)
                if not p.is_file():
                    print(f"[batch] skip {obj_id} trial {trial}: missing {p}")
                    continue
                cand_path = str(p.resolve())
            record = args.record_video and trial in record_trials
            jobs.append(
                {
                    "obj_id": obj_id,
                    "trial": trial,
                    "z_yaw_deg": z_yaw,
                    "episode_id": episode_id,
                    "candidate_hdf5": cand_path,
                    "record_video": record,
                }
            )

    important(f"[batch] {len(jobs)} episode(s) across {len(obj_ids)} object(s)")
    important(f"[batch] result_dir={result_dir}")

    for i, job in enumerate(jobs, 1):
        cmd = build_eval_single_cmd(
            args,
            obj_id=job["obj_id"],
            trial=job["trial"],
            z_yaw_deg=job["z_yaw_deg"],
            episode_id=job["episode_id"],
            candidate_hdf5=job["candidate_hdf5"],
            record_video=job["record_video"],
            isaac_py=isaac_py,
        )
        important(f"\n[batch] [{i}/{len(jobs)}] {job['episode_id']}")
        if args.dry_run or args.loud:
            important("[batch] " + " ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(PROJ))
        row = {
            "episode_id": job["episode_id"],
            "obj_id": job["obj_id"],
            "trial": job["trial"],
            "z_yaw_deg": job["z_yaw_deg"],
            "returncode": proc.returncode,
            "record_video": job["record_video"],
        }
        json_path = result_dir / f"{job['episode_id']}.json"
        if json_path.is_file():
            with json_path.open(encoding="utf-8") as f:
                ep = json.load(f)
            row["success"] = ep.get("success")
            row["failure_stage"] = ep.get("failure_stage")
            row["video_path"] = ep.get("video_path")
            important(
                "[batch] result "
                f"{'SUCCESS' if row.get('success') else 'FAILED'} "
                f"failure_stage={row.get('failure_stage')} "
                f"video={row.get('video_path') or ''}"
            )
        else:
            important(f"[batch] result missing json returncode={proc.returncode}")
        summary_rows.append(row)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "n_objects": len(obj_ids),
        "trials_per_object": args.trials_per_object,
        "dry_run": args.dry_run,
        "rows": summary_rows,
    }
    summary_path = result_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    important(f"\n[batch] wrote {summary_path}")


if __name__ == "__main__":
    main()
