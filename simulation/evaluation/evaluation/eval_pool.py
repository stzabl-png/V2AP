#!/usr/bin/env python3
"""Pool-style evaluation: pre-generate solutions, then run long-lived Isaac workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import add_affordance_checkpoint_args, resolve_affordance_checkpoint
from evaluation.candidate_batch import build_candidate_tasks, run_candidate_batch_generation
from evaluation.episode import discover_obj_ids
from evaluation.placement import add_random_obj_xy_args
from evaluation.randomness import add_eval_seed_args
from evaluation.eval_single import resolve_generate_mesh
from evaluation.solution_gen import generate_solutions, resolve_yaw_values
from evaluation.task_queue import build_task_queue, load_json, write_chunks, write_json
from evaluation.yaw import parse_yaw_pool


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
        description="Evaluation pool orchestrator with long-lived IsaacSim workers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--usd-root", default=None)
    p.add_argument("--candidate-dir", default=None)
    p.add_argument("--obj", action="append", default=None)
    p.add_argument("--obj-list", default=None)
    p.add_argument("--obj-limit", type=int, default=None)
    p.add_argument(
        "--trials-per-obj-yaw",
        type=int,
        default=None,
        help="Trials per object per yaw. Total per object = n_yaws * this value.",
    )
    p.add_argument(
        "--trials-per-object",
        type=int,
        default=None,
        help="Deprecated alias for --trials-per-obj-yaw.",
    )
    p.add_argument("--policy", choices=("a2g_pdm", "graspnet_baseline"), default="a2g_pdm")
    p.add_argument("--selection", choices=("top", "index", "sample"), default="sample")
    p.add_argument("--candidate-index", type=int, default=0)
    add_eval_seed_args(p)
    p.add_argument("--z-yaw-deg", type=float, default=None)
    p.add_argument("--z-yaw-grid", default=None)
    p.add_argument("--z-yaw-random", action="store_true")
    p.add_argument("--z-yaw-pool", default="0,90,180,270")
    p.add_argument("--generate-candidate-each-trial", action="store_true")
    p.add_argument(
        "--mesh-root",
        default=str(PROJ / "data_hub" / "meshes" / "SAM3DMesh" / "rotated_mesh"),
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--object-scale", type=float, default=1.0)
    add_random_obj_xy_args(p)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--result-dir", default=str(PROJ / "output" / "evaluation" / "pool"))
    p.add_argument("--save-hdf5", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--record-count-per-object", type=int, default=None)
    p.add_argument(
        "--record-all-workers",
        action="store_true",
        help="Allow all parallel workers to record video. Default records only worker 0 when n_workers > 1.",
    )
    p.add_argument("--record-fps", type=int, default=30)
    p.add_argument("--record-every", type=int, default=3)
    p.add_argument("--record-keep-frames", action="store_true")
    p.add_argument("--no-auto-xvfb", action="store_true")
    p.add_argument("--xvfb-screen", default="-screen 0 1280x720x24")
    p.add_argument("--sim-gpu-ids", default="0")
    p.add_argument("--sim-per-gpu", type=int, default=1)
    p.add_argument(
        "--sim-startup-stagger-s",
        type=float,
        default=15.0,
        help="Delay between Isaac worker launches on the same GPU.",
    )
    p.add_argument(
        "--startup-barrier-timeout-s",
        type=float,
        default=600.0,
        help="Seconds to wait for all workers to create SimulationApp before releasing tasks.",
    )
    p.add_argument(
        "--no-startup-barrier",
        action="store_true",
        help="Disable the worker startup barrier.",
    )
    p.add_argument("--candidate-python", default=None)
    p.add_argument(
        "--candidate-gpu-ids",
        default=None,
        help="GPUs for batch candidate generation; defaults to --sim-gpu-ids.",
    )
    p.add_argument(
        "--candidate-workers",
        type=int,
        default=None,
        help="Parallel batch_pdm_candidates processes (tasks split evenly). "
        "Default: one per GPU (--candidate-per-gpu 1).",
    )
    p.add_argument(
        "--candidate-per-gpu",
        type=int,
        default=None,
        help="If set and --candidate-workers omitted: workers = len(gpu_ids) * this "
        "(e.g. 2 GPUs × 3 = 6 workers).",
    )
    p.add_argument("--candidate-batch-multiplier", type=int, default=2)
    p.add_argument("--candidate-max-batches", type=int, default=10)
    p.add_argument(
        "--pdm-checkpoint",
        default=str(PROJ / "output" / "pdm" / "checkpoints_yaw_v6cond" / "best_model.pth"),
        help="PDM weights for batch_pdm_candidates (eval online generation).",
    )
    p.add_argument(
        "--pose-stats",
        default=str(PROJ / "output" / "pdm" / "checkpoints_yaw_v6cond" / "pose_stats.pt"),
        help="Pose normalization stats if not embedded in --pdm-checkpoint.",
    )
    add_affordance_checkpoint_args(p)
    p.add_argument(
        "--no-hard-gate",
        action="store_true",
        help="Skip grasp hard gates in batch_pdm_candidates (accept raw PDM samples).",
    )
    p.add_argument(
        "--no-filtering",
        action="store_true",
        help="Disable candidate filtering (hard gates + scoring) during batch PDM generation.",
    )
    p.add_argument(
        "--stop-after-solutions",
        action="store_true",
        help="Only generate candidates + solutions; do not launch Isaac workers.",
    )
    p.add_argument(
        "--candidates-only",
        action="store_true",
        help="Only run batch PDM candidate pools (requires --generate-candidate-each-trial). "
        "Skips solutions/*.json and Isaac. Use --trials-per-obj-yaw as pool size (e.g. 500).",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-only", action="store_true")
    p.add_argument("--loud", action="store_true")
    return p


def _gpu_ids(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _important(args, msg: str) -> None:
    if not args.log_only:
        print(msg, flush=True)


def _success_rate(n_success: int, n_total: int) -> float:
    return (float(n_success) / float(n_total)) if n_total else 0.0


def _aggregate_counts(rows: list[dict], key_fn) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        key = str(key_fn(row))
        item = out.setdefault(key, {"total": 0, "success": 0, "success_rate": 0.0})
        item["total"] += 1
        item["success"] += int(bool(row.get("success")))
    for item in out.values():
        item["success_rate"] = _success_rate(item["success"], item["total"])
    return dict(sorted(out.items()))


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        key = str(row.get("failure_stage") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _build_eval_summary(
    *,
    args: argparse.Namespace,
    result_dir: Path,
    obj_ids: list[str],
    queue: dict,
    rows: list[dict],
    n_success: int,
    total_rate: float,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir),
        "success": {
            "n_success": n_success,
            "n_total": int(queue["n_tasks"]),
            "success_rate": total_rate,
        },
        "inputs": {
            "objects": obj_ids,
            "obj_list": args.obj_list,
            "trials_per_obj_yaw": _trials_per_obj_yaw(args),
            "eval_seed": int(args.seed),
            "z_yaw_deg": args.z_yaw_deg,
            "z_yaw_grid": args.z_yaw_grid,
            "z_yaw_random": bool(args.z_yaw_random),
            "generate_candidate_each_trial": bool(args.generate_candidate_each_trial),
            "hp_affordance": bool(args.hp_affordance),
            "affordance_checkpoint": args.affordance_checkpoint,
            "candidate_gpus": args.candidate_gpu_ids or args.sim_gpu_ids,
            "sim_gpus": args.sim_gpu_ids,
            "sim_per_gpu": args.sim_per_gpu,
            "record_video": bool(args.record_video),
        },
        "counts": {
            "n_objects": len(obj_ids),
            "n_tasks": int(queue["n_tasks"]),
            "n_results": len(rows),
            "n_recorded": sum(1 for r in rows if r.get("video_path")),
        },
        "by_object": _aggregate_counts(rows, lambda r: r.get("obj_id", "unknown")),
        "by_yaw": _aggregate_counts(rows, lambda r: int(round(float(r.get("z_yaw_deg", 0.0)))) % 360),
        "by_object_yaw": _aggregate_counts(
            rows,
            lambda r: (
                f"{r.get('obj_id', 'unknown')}_"
                f"yaw{int(round(float(r.get('z_yaw_deg', 0.0)))) % 360:03d}"
            ),
        ),
        "failure_stages": _failure_counts(rows),
    }


def _trials_per_obj_yaw(args: argparse.Namespace) -> int:
    if args.trials_per_obj_yaw is not None:
        return max(1, int(args.trials_per_obj_yaw))
    if args.trials_per_object is not None:
        return max(1, int(args.trials_per_object))
    return 1


def _filter_objects_with_generate_mesh(args: argparse.Namespace, obj_ids: list[str]) -> list[str]:
    if not args.generate_candidate_each_trial:
        return obj_ids

    kept: list[str] = []
    skipped: list[tuple[str, str]] = []
    for obj_id in obj_ids:
        probe = argparse.Namespace(
            obj_id=obj_id,
            dataset=args.dataset,
            mesh=None,
            mesh_root=args.mesh_root,
            sam3d_rotated_mesh=False,
        )
        try:
            resolve_generate_mesh(probe)
            kept.append(obj_id)
        except Exception as exc:
            skipped.append((obj_id, str(exc)))

    for obj_id, reason in skipped:
        _important(args, f"[pool] skip {obj_id}: no generation mesh ({reason})")
    if not kept:
        raise SystemExit("No objects with generation mesh found; check --obj/--mesh-root/--dataset")
    return kept


def _make_isaac_worker_env(
    base_env: dict[str, str],
    *,
    gpu_id: str,
    result_dir: Path,
    chunk_path: Path,
) -> dict[str, str]:
    """Isolate Isaac/Kit cache, tmp, and GPU routing per worker process."""
    env = base_env.copy()
    chunk_id = chunk_path.stem
    worker_root = result_dir / "worker_isolation" / f"gpu{gpu_id}_{chunk_id}"
    hub_dir = worker_root / "hub"
    omni_cache = worker_root / "omni_cache"
    worker_tmp = worker_root / "tmp"
    worker_home = worker_root / "home"

    for path in (
        hub_dir,
        omni_cache,
        worker_tmp,
        worker_home / ".local" / "share" / "ov" / "data",
        worker_home / ".nvidia-omniverse" / "logs",
        worker_home / ".nvidia-omniverse" / "config",
        worker_home / ".nv" / "ComputeCache",
        worker_home / ".cache" / "ov",
        worker_home / ".cache" / "pip",
        worker_home / ".cache" / "nvidia" / "GLCache",
    ):
        path.mkdir(parents=True, exist_ok=True)

    # Share installed OV packages but keep mutable per-worker state isolated.
    global_pkg = Path.home() / ".local" / "share" / "ov" / "pkg"
    worker_pkg = worker_home / ".local" / "share" / "ov" / "pkg"
    if global_pkg.is_dir() and not worker_pkg.exists() and not worker_pkg.is_symlink():
        worker_pkg.parent.mkdir(parents=True, exist_ok=True)
        try:
            worker_pkg.symlink_to(global_pkg, target_is_directory=True)
        except FileExistsError:
            pass

    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["ISAAC_PHYSICAL_GPU_ID"] = str(gpu_id)
    # With CUDA_VISIBLE_DEVICES masking, PhysX/cudaDevice should use logical 0.
    env["ISAAC_SIM_GPU_ID"] = "0"
    # Kit/Vulkan active_gpu uses the physical GPU table index.
    env["ISAAC_KIT_ACTIVE_GPU"] = str(gpu_id)
    env["STRICT_GPU_MASK"] = "1"
    env["OMNICLIENT_HUB_CACHE_DIR"] = str(hub_dir)
    env["OMNI_CACHE_DIR"] = str(omni_cache)
    # Use real XDG_CACHE_HOME so curobo .so files are reused (not recompiled per worker).
    # env["XDG_CACHE_HOME"] = str(worker_root / "xdg_cache")
    env["TMPDIR"] = str(worker_tmp)
    env["TEMP"] = str(worker_tmp)
    env["TMP"] = str(worker_tmp)
    env["OMNI_USER"] = f"eval_g{gpu_id}_{chunk_id}"
    # Use real HOME so ~/.cache/torch_extensions is shared across workers.
    # env["HOME"] = str(worker_home)
    return env


def _prepare_startup_barrier(result_dir: Path) -> tuple[Path, Path]:
    barrier_dir = result_dir / "worker_start_barrier"
    ready_dir = barrier_dir / "ready"
    start_file = barrier_dir / "start.flag"
    if start_file.exists():
        start_file.unlink()
    if ready_dir.exists():
        shutil.rmtree(ready_dir)
    ready_dir.mkdir(parents=True, exist_ok=True)
    return ready_dir, start_file


def _wait_for_startup_barrier(
    args: argparse.Namespace,
    *,
    ready_files: list[Path],
    start_file: Path,
) -> None:
    if not ready_files:
        return
    timeout_s = max(1.0, float(args.startup_barrier_timeout_s))
    t0 = time.time()
    last_ready = -1
    while True:
        ready = [p for p in ready_files if p.is_file()]
        n_ready = len(ready)
        elapsed = time.time() - t0
        if n_ready != last_ready or int(elapsed) % 15 == 0:
            _important(
                args,
                f"[pool] startup barrier: {n_ready}/{len(ready_files)} ready "
                f"({elapsed:.1f}s)",
            )
            last_ready = n_ready
        if n_ready >= len(ready_files):
            break
        if elapsed >= timeout_s:
            missing = [p.name for p in ready_files if not p.is_file()]
            _important(
                args,
                "[pool] startup barrier timeout; releasing anyway. "
                f"missing={','.join(missing[:8])}",
            )
            break
        time.sleep(1.0)

    start_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        start_file,
        {
            "released_at": datetime.now(timezone.utc).isoformat(),
            "ready": len([p for p in ready_files if p.is_file()]),
            "expected": len(ready_files),
        },
    )
    _important(args, f"[pool] startup barrier released: {start_file}")


def _load_worker_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return load_json(path).get("results", [])
    except Exception:
        # Worker may be in the middle of rewriting the sidecar.
        return []


def _print_progress_row(
    args: argparse.Namespace,
    *,
    chunk_idx: int,
    gpu: str,
    row: dict,
    worker_done: int,
    worker_total: int,
    total_done: int,
    total_tasks: int,
) -> None:
    ok = "SUCCESS" if row.get("success") else "FAILED"
    parts = [
        "[pool] sim done",
        f"chunk={chunk_idx}",
        f"gpu={gpu}",
        f"obj={row.get('obj_id', '')}",
        f"yaw={float(row.get('z_yaw_deg', 0.0)):.0f}",
        ok,
    ]
    if row.get("video_path"):
        parts.append("recorded")
    if not row.get("success") and row.get("failure_stage"):
        parts.append(f"stage={row.get('failure_stage')}")
    parts.extend(
        [
            f"worker={worker_done}/{worker_total}",
            f"total={total_done}/{total_tasks}",
        ]
    )
    _important(args, " ".join(parts))


def _monitor_worker_progress(
    args: argparse.Namespace,
    launches: list[tuple[int, Path, str, subprocess.Popen | None, object, Path]],
    *,
    total_tasks: int,
) -> None:
    if args.dry_run:
        return

    seen_by_chunk: dict[int, set[str]] = {i: set() for i, *_ in launches}
    done_by_chunk: dict[int, int] = {i: 0 for i, *_ in launches}
    total_done = 0

    while True:
        all_finished = True
        for i, chunk_path, gpu, proc, _log_f, _log_path in launches:
            if proc is not None and proc.poll() is None:
                all_finished = False

            chunk = load_json(chunk_path)
            worker_total = len(chunk.get("tasks", []))
            rows = _load_worker_rows(chunk_path.with_name(f"{chunk_path.stem}_results.json"))
            for row in rows:
                task_id = str(row.get("task_id") or row.get("episode_id") or len(seen_by_chunk[i]))
                if task_id in seen_by_chunk[i]:
                    continue
                seen_by_chunk[i].add(task_id)
                done_by_chunk[i] += 1
                total_done += 1
                _print_progress_row(
                    args,
                    chunk_idx=i,
                    gpu=gpu,
                    row=row,
                    worker_done=done_by_chunk[i],
                    worker_total=worker_total,
                    total_done=total_done,
                    total_tasks=total_tasks,
                )

        if all_finished:
            break
        time.sleep(2.0)


def maybe_reexec_with_xvfb(args: argparse.Namespace) -> None:
    if not args.record_video or args.no_auto_xvfb or os.environ.get("DISPLAY"):
        return
    if os.environ.get("A2G_EVAL_XVFB_ACTIVE") == "1":
        return
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        raise RuntimeError(
            "--record-video needs a DISPLAY for pool workers. Install xvfb or run "
            "manually with `xvfb-run -s \"-screen 0 1280x720x24\" ...`."
        )
    cmd = [xvfb_run, "-a", "-s", args.xvfb_screen, sys.executable, *sys.argv]
    env = os.environ.copy()
    env["A2G_EVAL_XVFB_ACTIVE"] = "1"
    _important(args, "[pool] --record-video: DISPLAY missing; restarting under xvfb-run")
    if args.loud:
        _important(args, "[pool] " + " ".join(cmd))
    os.execvpe(xvfb_run, cmd, env)


def main() -> None:
    args = build_parser().parse_args()
    if args.log_only and args.loud:
        raise SystemExit("Use only one of --log-only or --loud")
    maybe_reexec_with_xvfb(args)

    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(args.hp_affordance),
        affordance_checkpoint=args.affordance_checkpoint,
    )
    args.affordance_checkpoint = str(aff_ckpt)

    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    obj_ids = discover_obj_ids(
        usd_root=args.usd_root,
        candidate_dir=args.candidate_dir,
        obj_ids=args.obj,
        obj_list_file=args.obj_list,
    )
    if args.obj_limit is not None:
        obj_ids = obj_ids[: max(0, int(args.obj_limit))]
    if not obj_ids:
        raise SystemExit("No objects discovered; pass --obj, --obj-list, --usd-root, or --candidate-dir")
    obj_ids = _filter_objects_with_generate_mesh(args, obj_ids)

    if args.candidates_only and not args.generate_candidate_each_trial:
        raise SystemExit("--candidates-only requires --generate-candidate-each-trial")

    yaw_grid = parse_yaw_pool(args.z_yaw_grid) if args.z_yaw_grid else None
    yaw_pool = parse_yaw_pool(args.z_yaw_pool) if args.z_yaw_random else None
    candidate_dir = Path(args.candidate_dir).expanduser() if args.candidate_dir else None

    if args.candidates_only:
        if args.dry_run:
            raise SystemExit("--candidates-only does not support --dry-run yet")
        _important(
            args,
            f"[pool] affordance checkpoint: {aff_ckpt}"
            + (" (hp-affordance)" if args.hp_affordance else ""),
        )
        pool_size = _trials_per_obj_yaw(args)
        yaw_values_by_obj = {
            oid: resolve_yaw_values(
                obj_id=oid,
                z_yaw_deg=args.z_yaw_deg,
                z_yaw_grid=yaw_grid,
                z_yaw_random_pool=yaw_pool,
                z_yaw_random=bool(args.z_yaw_random),
            )
            for oid in obj_ids
        }
        n_tasks = sum(len(yaws) for yaws in yaw_values_by_obj.values())
        _important(
            args,
            f"[pool] candidates-only: {len(obj_ids)} object(s), "
            f"{pool_size} pose(s)/obj×yaw, {n_tasks} pool file(s)",
        )
        if not args.no_hard_gate:
            _important(args, "[pool] hard gate ON (pass --no-hard-gate to disable)")
        if not args.no_filtering:
            _important(args, "[pool] filtering ON (pass --no-filtering to disable)")
        tasks = build_candidate_tasks(
            obj_ids=obj_ids,
            yaw_values_by_obj=yaw_values_by_obj,
            trials_per_obj_yaw=pool_size,
            result_dir=result_dir,
            mesh_root=args.mesh_root,
            dataset=args.dataset,
        )
        run_candidate_batch_generation(
            tasks=tasks,
            result_dir=result_dir,
            mesh_root=args.mesh_root,
            dataset=args.dataset,
            candidate_python=args.candidate_python,
            candidate_gpu_ids=args.candidate_gpu_ids or args.sim_gpu_ids,
            batch_multiplier=int(args.candidate_batch_multiplier),
            max_batches=int(args.candidate_max_batches),
            object_scale=float(args.object_scale),
            no_hard_gate=bool(args.no_hard_gate),
            no_filtering=bool(args.no_filtering),
            pdm_checkpoint=args.pdm_checkpoint,
            pose_stats=args.pose_stats,
            affordance_checkpoint=args.affordance_checkpoint,
            candidate_workers=args.candidate_workers,
            candidate_per_gpu=args.candidate_per_gpu,
            eval_seed=int(args.seed),
        )
        _important(args, f"[pool] done -> {result_dir}/candidates/{{obj_id}}/*_pool_grasp.hdf5")
        return

    _important(args, f"[pool] generating/loading solutions for {len(obj_ids)} object(s)")
    _important(
        args,
        f"[pool] affordance checkpoint: {aff_ckpt}"
        + (" (hp-affordance)" if args.hp_affordance else ""),
    )
    manifest = generate_solutions(
        obj_ids=obj_ids,
        result_dir=result_dir,
        trials_per_obj_yaw=_trials_per_obj_yaw(args),
        policy=args.policy,
        selection=args.selection,
        candidate_index=args.candidate_index,
        policy_seed=args.policy_seed,
        eval_seed=int(args.seed),
        z_yaw_deg=args.z_yaw_deg,
        z_yaw_grid=yaw_grid,
        z_yaw_random_pool=yaw_pool,
        z_yaw_random=bool(args.z_yaw_random),
        candidate_dir=candidate_dir,
        generate_candidate_each_trial=bool(args.generate_candidate_each_trial),
        mesh_root=args.mesh_root,
        dataset=args.dataset,
        record_video=bool(args.record_video),
        record_count_per_object=args.record_count_per_object,
        candidate_python=args.candidate_python,
        candidate_gpu_ids=args.candidate_gpu_ids or args.sim_gpu_ids,
        candidate_batch_multiplier=args.candidate_batch_multiplier,
        candidate_max_batches=args.candidate_max_batches,
        object_scale=args.object_scale,
        no_hard_gate=bool(args.no_hard_gate),
        no_filtering=bool(args.no_filtering),
        pdm_checkpoint=args.pdm_checkpoint,
        pose_stats=args.pose_stats,
        affordance_checkpoint=args.affordance_checkpoint,
        candidate_workers=args.candidate_workers,
        candidate_per_gpu=args.candidate_per_gpu,
        reuse_existing=bool(args.resume),
        dry_run=bool(args.dry_run),
        random_obj_xy=bool(args.random_obj_xy),
        obj_xy_jitter_m=float(args.obj_xy_jitter_m),
    )
    if args.random_obj_xy:
        _important(
            args,
            f"[pool] random_obj_xy ON jitter=±{float(args.obj_xy_jitter_m):.3f}m "
            "(offsets stored per solution JSON)",
        )
    queue = build_task_queue(manifest)
    write_json(result_dir / "task_queue.json", queue)

    if args.stop_after_solutions:
        _important(args, f"[pool] --stop-after-solutions: wrote candidates + solutions under {result_dir}")
        _important(args, f"[pool] {queue['n_tasks']} task(s) queued; re-run without --stop-after-solutions to sim")
        return

    n_workers = max(1, len(_gpu_ids(args.sim_gpu_ids)) * max(1, int(args.sim_per_gpu)))
    # If any task in a chunk records video, the chunk cannot be headless.
    chunks = write_chunks(
        queue=queue,
        result_dir=result_dir,
        n_chunks=n_workers,
        object_scale=args.object_scale,
        headless=bool(args.headless),
        save_hdf5=bool(args.save_hdf5),
        record_fps=args.record_fps,
        record_every=args.record_every,
        record_keep_frames=args.record_keep_frames,
    )
    record_worker_idx = 0
    if args.record_video and len(chunks) > 1 and not args.record_all_workers:
        record_counts = []
        for path in chunks:
            chunk = load_json(path)
            record_counts.append(sum(1 for t in chunk.get("tasks", []) if t.get("record_video")))
        if any(record_counts):
            record_worker_idx = max(range(len(record_counts)), key=lambda idx: (record_counts[idx], -idx))

    # Patch per-chunk headless after split based on video tasks.
    for ci, path in enumerate(chunks):
        chunk = load_json(path)
        record_disabled_for_chunk = False
        if args.record_video and len(chunks) > 1 and not args.record_all_workers and ci != record_worker_idx:
            for task in chunk.get("tasks", []):
                task["record_video"] = False
            record_disabled_for_chunk = True
        if any(t.get("record_video") for t in chunk.get("tasks", [])):
            chunk["headless"] = False
        elif record_disabled_for_chunk:
            # In parallel video mode, non-recording workers should not open a
            # viewport on the shared Xvfb/DISPLAY.
            chunk["headless"] = True
        write_json(path, chunk)

    if args.record_video and len(chunks) > 1 and not args.record_all_workers:
        _important(
            args,
            f"[pool] record-video with multiple workers: only worker {record_worker_idx} will record "
            "(use --record-all-workers to override)",
        )

    _important(args, f"[pool] {queue['n_tasks']} task(s), {len(chunks)} worker chunk(s)")
    for i, chunk_path in enumerate(chunks):
        chunk = load_json(chunk_path)
        obj_ids_in_chunk = sorted({str(t["obj_id"]) for t in chunk.get("tasks", [])})
        _important(
            args,
            "[pool] "
            f"chunk {i}: tasks={len(chunk.get('tasks', []))} "
            f"objects={','.join(obj_ids_in_chunk)}",
        )
    isaac_py = isaac_python()
    worker_script = PROJ / "sim" / "evaluation" / "run_eval_worker.py"
    gpu_ids = _gpu_ids(args.sim_gpu_ids) or ["0"]
    use_startup_barrier = (not args.no_startup_barrier) and len(chunks) > 1
    ready_dir = start_file = None
    if use_startup_barrier:
        ready_dir, start_file = _prepare_startup_barrier(result_dir)

    launches = []
    ready_files: list[Path] = []
    launch_count_by_gpu: dict[str, int] = {}
    for i, chunk_path in enumerate(chunks):
        chunk = load_json(chunk_path)
        gpu = gpu_ids[i % len(gpu_ids)]
        cmd = [isaac_py, str(worker_script), "--worker-chunk", str(chunk_path)]
        if chunk.get("headless", True):
            cmd.append("--headless")
        launch_idx = launch_count_by_gpu.get(str(gpu), 0)
        launch_count_by_gpu[str(gpu)] = launch_idx + 1
        startup_delay_s = launch_idx * max(0.0, float(args.sim_startup_stagger_s))
        if startup_delay_s > 0:
            cmd.extend(["--startup-delay-s", str(startup_delay_s)])
        if use_startup_barrier:
            assert ready_dir is not None and start_file is not None
            ready_file = ready_dir / f"{chunk_path.stem}_gpu{gpu}.ready"
            ready_files.append(ready_file)
            cmd.extend(
                [
                    "--startup-ready-file",
                    str(ready_file),
                    "--startup-start-file",
                    str(start_file),
                ]
            )
        env = _make_isaac_worker_env(
            os.environ.copy(),
            gpu_id=str(gpu),
            result_dir=result_dir,
            chunk_path=chunk_path,
        )
        log_path = result_dir / "logs" / f"worker_{i:03d}.log"
        _important(
            args,
            f"[pool] worker {i+1}/{len(chunks)} gpu={gpu} "
            f"tasks={len(chunk.get('tasks', []))} startup_delay={startup_delay_s:.0f}s",
        )
        if args.dry_run:
            _important(args, "[pool] " + " ".join(cmd))
            launches.append((i, chunk_path, str(gpu), None, None, log_path))
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
            stdout = log_f
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJ),
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
            )
            launches.append((i, chunk_path, str(gpu), proc, log_f, log_path))

    if use_startup_barrier and not args.dry_run:
        assert start_file is not None
        _wait_for_startup_barrier(args, ready_files=ready_files, start_file=start_file)

    _monitor_worker_progress(args, launches, total_tasks=int(queue["n_tasks"]))

    rows = []
    for i, chunk_path, _gpu, proc, log_f, log_path in launches:
        rc = 0 if proc is None else proc.wait()
        if log_f is not None:
            log_f.close()
        if args.loud and log_path is not None and log_path.is_file():
            print(log_path.read_text(encoding="utf-8"), end="")
        results_path = chunk_path.with_name(f"{chunk_path.stem}_results.json")
        worker_rows = []
        if results_path.is_file():
            worker_rows = load_json(results_path).get("results", [])
        rows.extend(worker_rows)
        chunk = load_json(chunk_path)
        worker_total = len(chunk.get("tasks", []))
        worker_success = sum(1 for r in worker_rows if r.get("success"))
        worker_rate = _success_rate(worker_success, worker_total)
        _important(
            args,
            f"[pool] worker {i+1} returncode={rc} "
            f"success={worker_success}/{worker_total} "
            f"rate={worker_rate:.1%} results={len(worker_rows)}",
        )

    n_success = sum(1 for r in rows if r.get("success"))
    total_rate = _success_rate(n_success, int(queue["n_tasks"]))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(result_dir),
        "n_objects": len(obj_ids),
        "n_tasks": queue["n_tasks"],
        "n_results": len(rows),
        "n_success": n_success,
        "success_rate": total_rate,
        "dry_run": bool(args.dry_run),
        "rows": rows,
    }
    write_json(result_dir / "batch_summary.json", summary)
    eval_summary = _build_eval_summary(
        args=args,
        result_dir=result_dir,
        obj_ids=obj_ids,
        queue=queue,
        rows=rows,
        n_success=n_success,
        total_rate=total_rate,
    )
    write_json(result_dir / "eval_summary.json", eval_summary)
    _important(args, f"[pool] total success={n_success}/{queue['n_tasks']} rate={total_rate:.1%}")
    _important(args, f"[pool] wrote {result_dir / 'batch_summary.json'}")
    _important(args, f"[pool] wrote {result_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()

