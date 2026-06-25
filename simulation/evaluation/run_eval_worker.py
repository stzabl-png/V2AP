#!/usr/bin/env python3
"""Long-lived IsaacSim worker for evaluation task chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation pool IsaacSim worker")
    p.add_argument("--worker-chunk", required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--startup-delay-s", type=float, default=0.0)
    p.add_argument("--startup-ready-file", default=None)
    p.add_argument("--startup-start-file", default=None)
    return p.parse_args()


args = _parse_args()
if args.startup_delay_s > 0:
    print(f"[worker-startup] sleeping {args.startup_delay_s:.1f}s before Isaac launch", flush=True)
    time.sleep(float(args.startup_delay_s))

from isaacsim import SimulationApp  # noqa: E402

_SIM_GPU_ID = int(os.environ.get("ISAAC_SIM_GPU_ID", "0"))
_KIT_GPU_ID = int(os.environ.get("ISAAC_KIT_ACTIVE_GPU", os.environ.get("ISAAC_SIM_GPU_ID", "0")))
_PHYSICAL_GPU_ID = int(os.environ.get("ISAAC_PHYSICAL_GPU_ID", os.environ.get("ISAAC_SIM_GPU_ID", "0")))
_launch_cfg = {
    "headless": args.headless,
    "multi_gpu": False,
    "max_gpu_count": 1,
    "active_gpu": _KIT_GPU_ID,
    "physics_gpu": _SIM_GPU_ID,
}
print(
    "[gpu-isolation] "
    f"physical_gpu={_PHYSICAL_GPU_ID} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
    f"active_gpu={_KIT_GPU_ID} physics_gpu={_SIM_GPU_ID} "
    f"HOME={os.environ.get('HOME')!r} TMPDIR={os.environ.get('TMPDIR')!r}",
    flush=True,
)
simulation_app = SimulationApp(_launch_cfg)

if args.startup_ready_file and args.startup_start_file:
    ready_path = Path(args.startup_ready_file).expanduser().resolve()
    start_path = Path(args.startup_start_file).expanduser().resolve()
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    with ready_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "worker_chunk": str(Path(args.worker_chunk).expanduser().resolve()),
                "physical_gpu": _PHYSICAL_GPU_ID,
                "active_gpu": _KIT_GPU_ID,
                "physics_gpu": _SIM_GPU_ID,
                "ready_at": time.time(),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"[worker-startup] ready file written: {ready_path}", flush=True)
    while not start_path.is_file():
        time.sleep(0.5)
    print(f"[worker-startup] start barrier released: {start_path}", flush=True)

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402
from termcolor import cprint  # noqa: E402

from evaluation.results import append_episode_jsonl, build_episode_record, write_episode_json  # noqa: E402
from evaluation.specs import ExecutionResult, OpenLoopGraspCommand, PolicyOutput  # noqa: E402
from evaluation.solution_gen import load_json  # noqa: E402
from sim.evaluation.curobo_executor import (  # noqa: E402
    execute_open_loop_grasp,
    reset_motion_gen,
    write_robot_gt_hdf5,
)
from sim.evaluation.scene_builder import build_scene_spec, reset_scene_pose, setup_scene, swap_scene_object  # noqa: E402
from sim.evaluation.video_recorder import EpisodeVideoRecorder  # noqa: E402


def _command_from_dict(data: dict) -> OpenLoopGraspCommand:
    return OpenLoopGraspCommand(
        position=np.asarray(data["position"], dtype=np.float64),
        rotation=np.asarray(data["rotation"], dtype=np.float64),
        gripper_width=float(data["gripper_width"]),
        frame=data.get("frame", "object_mesh"),
        ee_frame_convention=data.get("ee_frame_convention", "a2g_grasp_frame"),
        name=data.get("name", "candidate"),
        score=float(data.get("score", 0.0)),
        approach_type=data.get("approach_type", ""),
        is_manual=bool(data.get("is_manual", False)),
        mesh_prerotation_euler=data.get("mesh_prerotation_euler"),
        mesh_prerotation=data.get("mesh_prerotation"),
        metadata=data.get("metadata", {}),
    )


def _policy_output_from_solution(solution: dict) -> PolicyOutput:
    po = solution["policy_output"]
    command = _command_from_dict(po["command"]) if po.get("command") else None
    return PolicyOutput(
        kind=po["kind"],
        command=command,
        actions=po.get("actions"),
        metadata=po.get("metadata", {}),
    )


def _write_progress(path: Path, chunk_id: str, results: list[dict], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_id": chunk_id,
        "completed": len(results),
        "total": total,
        "successes": sum(1 for r in results if r.get("success")),
        "last": results[-1] if results else None,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _execute_task(scene, task: dict, chunk: dict, result_dir: Path):
    solution = load_json(Path(task["solution_path"]))
    policy_output = _policy_output_from_solution(solution)
    if policy_output.kind != "open_loop_grasp" or policy_output.command is None:
        raise RuntimeError(f"unsupported policy output: {policy_output.kind}")

    spec = build_scene_spec(
        obj_id=task["obj_id"],
        episode_id=task["episode_id"],
        dataset=solution.get("dataset"),
        object_scale=float(chunk.get("object_scale", 1.0)),
        sim_z_yaw_deg=float(task["z_yaw_deg"]),
        seed=int(task.get("trial", 0)),
        candidate_hdf5=solution.get("candidate_hdf5"),
        obj_xy_offset=solution.get("obj_xy_offset"),
    )

    if scene is None:
        cprint(f"[worker] setup scene {spec.obj_id} yaw={spec.sim_z_yaw_deg:.0f}", "cyan")
        scene = setup_scene(spec, render=not bool(chunk.get("headless", True)))
        reset_motion_gen()
    elif scene.spec.obj_id != spec.obj_id:
        cprint(f"[worker] swap object {scene.spec.obj_id} -> {spec.obj_id}", "cyan")
        swap_scene_object(scene, spec)
        reset_motion_gen()
    else:
        dx, dy = spec.obj_xy_offset
        cprint(
            f"[worker] reset {spec.obj_id} yaw={spec.sim_z_yaw_deg:.0f} "
            f"xy_offset=({dx:+.3f},{dy:+.3f})",
            "cyan",
        )
        reset_scene_pose(scene, spec)

    recorder = None
    video_path = None
    try:
        if task.get("record_video"):
            video_path = str((result_dir / "episodes" / f"{task['episode_id']}.mp4").resolve())
            recorder = EpisodeVideoRecorder(
                video_path,
                record_every=int(chunk.get("record_every", 3)),
                fps=int(chunk.get("record_fps", 30)),
                keep_frames=bool(chunk.get("record_keep_frames", False)),
            )
            recorder.attach_world(scene.world)
            recorder.start()

        execution = execute_open_loop_grasp(scene, policy_output.command)
        if recorder is not None:
            video_path = recorder.stop()
            execution.video_path = video_path
            execution.video_n_frames = recorder.n_frames
            if recorder.encode_error:
                execution.metadata["video_encode_error"] = recorder.encode_error
    finally:
        if recorder is not None and recorder._active:
            recorder.stop()

    ep_dir = result_dir / "episodes"
    record = build_episode_record(
        scene=spec,
        policy_name=solution.get("policy", "a2g_pdm"),
        policy_output=policy_output,
        execution=execution,
        video_path=video_path,
    )
    json_path = write_episode_json(record, str(ep_dir))
    append_episode_jsonl(record, str(result_dir))
    h5_path = ""
    if chunk.get("save_hdf5"):
        h5_path = write_robot_gt_hdf5(
            result_dir=str(ep_dir),
            scene=spec,
            command=policy_output.command,
            execution=execution,
            policy_name=solution.get("policy", "a2g_pdm"),
        )
    return scene, {
        "task_id": task["task_id"],
        "episode_id": task["episode_id"],
        "obj_id": task["obj_id"],
        "z_yaw_deg": float(task["z_yaw_deg"]),
        "success": bool(execution.success),
        "failure_stage": execution.failure_stage,
        "z_delta_m": execution.z_delta_m,
        "json_path": json_path,
        "hdf5_path": h5_path,
        "video_path": video_path,
        "error": "",
    }


def main() -> None:
    chunk_path = Path(args.worker_chunk).expanduser().resolve()
    chunk = load_json(chunk_path)
    result_dir = Path(chunk["result_dir"]).expanduser().resolve()
    chunk_id = chunk.get("chunk_id", chunk_path.stem)
    tasks = chunk.get("tasks", [])
    results_path = chunk_path.with_name(f"{chunk_path.stem}_results.json")
    progress_path = chunk_path.with_name(f"{chunk_path.stem}_progress.json")

    cprint(f"[worker] {chunk_id}: {len(tasks)} task(s)", "cyan")
    scene = None
    results: list[dict] = []
    completed_task_ids: set[str] = set()
    if results_path.is_file():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            results = list(existing.get("results", []))
            completed_task_ids = {str(r.get("task_id", "")) for r in results if r.get("task_id")}
            if completed_task_ids:
                cprint(
                    f"[worker] resume: found {len(completed_task_ids)} completed task(s) in {results_path.name}",
                    "cyan",
                )
        except Exception as exc:
            cprint(f"[worker] resume: failed to read {results_path.name}: {exc}", "red")
            results = []
            completed_task_ids = set()
    try:
        for task in tasks:
            if str(task.get("task_id", "")) in completed_task_ids:
                continue
            cprint(f"[worker] task {task['task_id']}", "yellow")
            try:
                scene, row = _execute_task(scene, task, chunk, result_dir)
                ok = "SUCCESS" if row["success"] else "FAILED"
                cprint(
                    f"[worker] {ok} {task['task_id']} dz={row.get('z_delta_m')} "
                    f"stage={row.get('failure_stage')}",
                    "green" if row["success"] else "red",
                )
            except Exception as exc:
                row = {
                    "task_id": task["task_id"],
                    "episode_id": task.get("episode_id", ""),
                    "obj_id": task.get("obj_id", ""),
                    "z_yaw_deg": float(task.get("z_yaw_deg", 0.0)),
                    "success": False,
                    "failure_stage": "worker_exception",
                    "z_delta_m": None,
                    "json_path": "",
                    "hdf5_path": "",
                    "video_path": "",
                    "error": str(exc),
                }
                cprint(f"[worker] ERROR {task['task_id']}: {exc}", "red")
            results.append(row)
            completed_task_ids.add(str(row.get("task_id", "")))
            with results_path.open("w", encoding="utf-8") as f:
                json.dump({"chunk_id": chunk_id, "results": results}, f, indent=2, sort_keys=True)
                f.write("\n")
            _write_progress(progress_path, chunk_id, results, len(tasks))
    finally:
        cprint(f"[worker] wrote {results_path}", "green")
        simulation_app.close()


if __name__ == "__main__":
    main()

