"""Generate reusable policy solutions before IsaacSim workers run."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from evaluation.candidate_batch import build_candidate_tasks, run_candidate_batch_generation
from evaluation.episode import build_episode_id, default_candidate_hdf5, pick_record_trials
from evaluation.eval_single import (
    default_candidate_python,
    maybe_generate_candidate,
)
from evaluation.policies.a2g_pdm import A2GPDMPolicy, A2GPDMPolicyConfig
from evaluation.policies.graspnet_baseline import GraspNetBaselinePolicy, GraspNetBaselinePolicyConfig
from evaluation.randomness import DEFAULT_EVAL_SEED, record_trials_rng, resolve_policy_seed
from evaluation.placement import resolve_obj_xy_offset
from evaluation.yaw import resolve_z_yaw_deg


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _generation_namespace(
    *,
    obj_id: str,
    candidate_dir: Path,
    candidate_output: Path,
    mesh_root: str | None,
    mesh: str | None,
    dataset: str | None,
    z_yaw_deg: float,
    candidate_python: str,
) -> Namespace:
    return Namespace(
        generate_candidate=True,
        obj_id=obj_id,
        mesh=mesh,
        mesh_root=mesh_root,
        sam3d_rotated_mesh=False,
        candidate_dir=str(candidate_dir),
        candidate_output=str(candidate_output),
        candidate_python=candidate_python,
        dataset=dataset,
        z_yaw_deg=float(z_yaw_deg),
    )


def ensure_candidate_hdf5(
    *,
    obj_id: str,
    z_yaw_deg: float,
    trial: int,
    candidate_dir: Path | None,
    generate_candidate: bool,
    mesh_root: str | None,
    dataset: str | None,
    result_dir: Path,
    candidate_python: str | None,
) -> str:
    if not generate_candidate:
        if candidate_dir is None:
            raise ValueError("candidate_dir required unless generate_candidate=True")
        path = default_candidate_hdf5(candidate_dir, obj_id, z_yaw_deg)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path.resolve())

    out_dir = result_dir / "candidates" / obj_id
    out_path = out_dir / f"trial_{trial:03d}_yaw{int(round(z_yaw_deg)) % 360:03d}_grasp.hdf5"
    ns = _generation_namespace(
        obj_id=obj_id,
        candidate_dir=out_dir,
        candidate_output=out_path,
        mesh_root=mesh_root,
        mesh=None,
        dataset=dataset,
        z_yaw_deg=z_yaw_deg,
        candidate_python=candidate_python or default_candidate_python(),
    )
    return maybe_generate_candidate(ns)


def make_solution(
    *,
    obj_id: str,
    policy: str,
    trial: int,
    z_yaw_deg: float,
    candidate_hdf5: str,
    selection: str,
    candidate_index: int,
    policy_seed: int | None,
    eval_seed: int = DEFAULT_EVAL_SEED,
    random_obj_xy: bool = False,
    obj_xy_jitter_m: float = 0.05,
) -> dict[str, Any]:
    episode_id = build_episode_id(obj_id, policy, trial, z_yaw_deg)
    dx, dy = resolve_obj_xy_offset(
        random_obj_xy=bool(random_obj_xy),
        obj_xy_jitter_m=float(obj_xy_jitter_m),
        obj_id=obj_id,
        trial=int(trial),
        sim_z_yaw_deg=float(z_yaw_deg),
        eval_seed=int(eval_seed),
    )
    if policy == "graspnet_baseline":
        seed = resolve_policy_seed(eval_seed=eval_seed, policy_seed=policy_seed, trial=trial)
        output = GraspNetBaselinePolicy(
            GraspNetBaselinePolicyConfig(
                candidate_hdf5=candidate_hdf5,
                selection=selection,
                candidate_index=candidate_index,
                seed=seed,
            )
        ).predict(None)
    elif policy == "a2g_pdm":
        seed = resolve_policy_seed(eval_seed=eval_seed, policy_seed=policy_seed, trial=trial)
        output = A2GPDMPolicy(
            A2GPDMPolicyConfig(
                candidate_hdf5=candidate_hdf5,
                selection=selection,
                candidate_index=candidate_index,
                seed=seed,
            )
        ).predict(None)
    else:
        raise ValueError(f"unsupported policy: {policy}")
    return {
        "version": 1,
        "solution_id": episode_id,
        "episode_id": episode_id,
        "obj_id": obj_id,
        "policy": policy,
        "trial": int(trial),
        "eval_seed": int(eval_seed),
        "z_yaw_deg": float(z_yaw_deg),
        "obj_xy_offset": [float(dx), float(dy)],
        "random_obj_xy": bool(random_obj_xy),
        "obj_xy_jitter_m": float(obj_xy_jitter_m),
        "candidate_hdf5": os.path.abspath(candidate_hdf5),
        "selection": selection,
        "candidate_index": int(candidate_index),
        "policy_seed": seed,
        "policy_output": output.to_dict(),
    }


def resolve_yaw_values(
    *,
    obj_id: str,
    z_yaw_deg: float | None,
    z_yaw_grid: list[float] | None,
    z_yaw_random_pool: list[float] | None,
    z_yaw_random: bool,
) -> list[float]:
    """Resolve the yaw outer loop for pool eval.

    Pool semantics are per-yaw: each object runs every yaw value K times, where
    K is --trials-per-obj-yaw. This intentionally differs from eval_batch's
    older trial-cycling behavior.
    """
    if z_yaw_deg is not None:
        return [float(z_yaw_deg) % 360.0]
    if z_yaw_grid:
        return [float(y) % 360.0 for y in z_yaw_grid]
    if z_yaw_random:
        return [float(y) % 360.0 for y in (z_yaw_random_pool or [0.0, 90.0, 180.0, 270.0])]
    return [
        resolve_z_yaw_deg(
            trial=0,
            obj_id=obj_id,
            z_yaw_deg=z_yaw_deg,
            z_yaw_grid=z_yaw_grid,
            z_yaw_random_pool=z_yaw_random_pool,
            z_yaw_random=z_yaw_random,
        )
    ]


def generate_solutions(
    *,
    obj_ids: list[str],
    result_dir: Path,
    trials_per_obj_yaw: int,
    policy: str,
    selection: str,
    candidate_index: int,
    policy_seed: int | None,
    z_yaw_deg: float | None,
    z_yaw_grid: list[float] | None,
    z_yaw_random_pool: list[float] | None,
    z_yaw_random: bool,
    candidate_dir: Path | None,
    generate_candidate_each_trial: bool,
    mesh_root: str | None,
    dataset: str | None,
    record_video: bool,
    record_count_per_object: int | None,
    candidate_python: str | None = None,
    candidate_gpu_ids: str | None = None,
    candidate_batch_multiplier: int = 2,
    candidate_max_batches: int = 10,
    object_scale: float = 1.0,
    no_hard_gate: bool = False,
    no_filtering: bool = False,
    pdm_checkpoint: str | None = None,
    pose_stats: str | None = None,
    affordance_checkpoint: str | None = None,
    candidate_workers: int | None = None,
    candidate_per_gpu: int | None = None,
    reuse_existing: bool = True,
    dry_run: bool = False,
    random_obj_xy: bool = False,
    obj_xy_jitter_m: float = 0.05,
    eval_seed: int = DEFAULT_EVAL_SEED,
) -> dict:
    sol_dir = result_dir / "solutions"
    sol_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    yaw_values_by_obj = {
        obj_id: resolve_yaw_values(
            obj_id=obj_id,
            z_yaw_deg=z_yaw_deg,
            z_yaw_grid=z_yaw_grid,
            z_yaw_random_pool=z_yaw_random_pool,
            z_yaw_random=z_yaw_random,
        )
        for obj_id in obj_ids
    }
    batch_candidate_map: dict[tuple[str, float], str] = {}
    if generate_candidate_each_trial and not dry_run:
        candidate_tasks = build_candidate_tasks(
            obj_ids=obj_ids,
            yaw_values_by_obj=yaw_values_by_obj,
            trials_per_obj_yaw=int(trials_per_obj_yaw),
            result_dir=result_dir,
            mesh_root=mesh_root or "",
            dataset=dataset,
        )
        batch_candidate_map = run_candidate_batch_generation(
            tasks=candidate_tasks,
            result_dir=result_dir,
            mesh_root=mesh_root or "",
            dataset=dataset,
            candidate_python=candidate_python,
            candidate_gpu_ids=candidate_gpu_ids,
            batch_multiplier=int(candidate_batch_multiplier),
            max_batches=int(candidate_max_batches),
            object_scale=float(object_scale),
            no_hard_gate=bool(no_hard_gate),
            no_filtering=bool(no_filtering),
            pdm_checkpoint=pdm_checkpoint,
            pose_stats=pose_stats,
            affordance_checkpoint=affordance_checkpoint,
            candidate_workers=candidate_workers,
            candidate_per_gpu=candidate_per_gpu,
            eval_seed=int(eval_seed),
        )

    n_objs = len(obj_ids)
    for obj_idx, obj_id in enumerate(obj_ids):
        yaw_values = yaw_values_by_obj[obj_id]
        n_obj_episodes = len(yaw_values) * int(trials_per_obj_yaw)
        record_trials = pick_record_trials(
            n_obj_episodes,
            record_count_per_object if record_video else 0,
            record_trials_rng(eval_seed=eval_seed, obj_id=obj_id),
        )
        obj_new = 0
        for yaw_idx, yaw in enumerate(yaw_values):
            for trial in range(int(trials_per_obj_yaw)):
                obj_episode_idx = yaw_idx * int(trials_per_obj_yaw) + trial
                episode_id = build_episode_id(obj_id, policy, trial, yaw)
                sol_path = sol_dir / f"{episode_id}.json"
                if reuse_existing and sol_path.is_file():
                    sol = load_json(sol_path)
                elif dry_run:
                    sol = {
                        "version": 1,
                        "solution_id": episode_id,
                        "episode_id": episode_id,
                        "obj_id": obj_id,
                        "policy": policy,
                        "trial": int(trial),
                        "eval_seed": int(eval_seed),
                        "z_yaw_deg": float(yaw),
                        "obj_xy_offset": [0.0, 0.0],
                        "random_obj_xy": bool(random_obj_xy),
                        "obj_xy_jitter_m": float(obj_xy_jitter_m),
                        "candidate_hdf5": "",
                        "selection": selection,
                        "candidate_index": int(candidate_index),
                        "policy_seed": policy_seed,
                        "policy_output": {"kind": "open_loop_grasp", "command": None, "metadata": {}},
                        "dry_run": True,
                    }
                    write_json(sol_path, sol)
                else:
                    if generate_candidate_each_trial:
                        cand = batch_candidate_map[(obj_id, float(yaw))]
                        solution_selection = "index"
                        solution_index = trial
                    else:
                        cand = ensure_candidate_hdf5(
                            obj_id=obj_id,
                            z_yaw_deg=yaw,
                            trial=trial,
                            candidate_dir=candidate_dir,
                            generate_candidate=False,
                            mesh_root=mesh_root,
                            dataset=dataset,
                            result_dir=result_dir,
                            candidate_python=candidate_python,
                        )
                        solution_selection = selection
                        solution_index = candidate_index
                    sol = make_solution(
                        obj_id=obj_id,
                        policy=policy,
                        trial=trial,
                        z_yaw_deg=yaw,
                        candidate_hdf5=cand,
                        selection=solution_selection,
                        candidate_index=solution_index,
                        policy_seed=policy_seed,
                        eval_seed=int(eval_seed),
                        random_obj_xy=bool(random_obj_xy),
                        obj_xy_jitter_m=float(obj_xy_jitter_m),
                    )
                    write_json(sol_path, sol)
                    obj_new += 1
                rows.append(
                    {
                        "solution_id": sol["solution_id"],
                        "episode_id": sol["episode_id"],
                        "obj_id": obj_id,
                        "trial": trial,
                        "z_yaw_deg": yaw,
                        "solution_path": str(sol_path.resolve()),
                        "record_video": bool(record_video and obj_episode_idx in record_trials),
                    }
                )
        if not dry_run:
            print(
                f"[solutions] {obj_idx + 1}/{n_objs} {obj_id}: "
                f"{n_obj_episodes} episodes ({obj_new} new)",
                flush=True,
                file=sys.stderr,
            )

    manifest = {
        "version": 1,
        "solutions_dir": str(sol_dir.resolve()),
        "n_solutions": len(rows),
        "solutions": rows,
    }
    write_json(sol_dir / "manifest.json", manifest)
    return manifest

