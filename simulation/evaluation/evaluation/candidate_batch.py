"""Parallel candidate generation for eval_pool."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from evaluation.eval_single import (
    candidate_generation_env,
    default_candidate_python,
    resolve_generate_mesh,
)

PROJ = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_gpu_ids(text: str | None) -> list[str]:
    if not text:
        return ["0"]
    return [part.strip() for part in str(text).split(",") if part.strip()] or ["0"]


def _task_key(obj_id: str, z_yaw_deg: float) -> tuple[str, float]:
    return str(obj_id), float(z_yaw_deg)


def _manifest_rows_valid(rows: list[dict]) -> bool:
    if not rows:
        return False
    for row in rows:
        path = Path(str(row.get("output_hdf5", "")))
        if not path.is_file():
            return False
    return True


def _rows_to_mapping(rows: list[dict]) -> dict[tuple[str, float], str]:
    mapping: dict[tuple[str, float], str] = {}
    for row in rows:
        key = _task_key(row["obj_id"], row["z_yaw_deg"])
        mapping[key] = str(Path(row["output_hdf5"]).resolve())
    return mapping


def _expected_keys(tasks: list[dict]) -> set[tuple[str, float]]:
    return {_task_key(t["obj_id"], t["z_yaw_deg"]) for t in tasks}


def _load_existing_candidate_map(
    result_dir: Path,
    tasks: list[dict],
) -> dict[tuple[str, float], str] | None:
    """Reuse completed chunk/merged manifests so a crashed worker exit does not force regen."""
    work_dir = result_dir / "candidate_generation"
    expected = _expected_keys(tasks)
    if not expected:
        return {}

    merged_path = work_dir / "candidate_manifest.json"
    if merged_path.is_file():
        try:
            payload = json.loads(merged_path.read_text(encoding="utf-8"))
            rows = payload.get("tasks", [])
            mapping = _rows_to_mapping(rows)
            if expected <= set(mapping.keys()) and all(Path(p).is_file() for p in mapping.values()):
                return {k: mapping[k] for k in expected}
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    mapping: dict[tuple[str, float], str] = {}
    for manifest_path in sorted(work_dir.glob("candidate_chunk_*_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("tasks", [])
        if not _manifest_rows_valid(rows):
            continue
        mapping.update(_rows_to_mapping(rows))

    if expected <= set(mapping.keys()):
        return {k: mapping[k] for k in expected}
    return None


def _emit_mapping_logs(mapping: dict[tuple[str, float], str], manifest_rows: list[dict]) -> None:
    row_by_key = {_task_key(r["obj_id"], r["z_yaw_deg"]): r for r in manifest_rows}
    for key in sorted(mapping.keys()):
        row = row_by_key.get(key)
        if row is None:
            print(
                f"[candidate-batch] reuse obj={key[0]} yaw={key[1]:.0f} "
                f"path={mapping[key]}",
                flush=True,
            )
            continue
        print(
            "[candidate-batch] done "
            f"obj={row['obj_id']} yaw={float(row['z_yaw_deg']):.0f} "
            f"selected={row.get('n_selected', '?')} pass={row.get('hard_gate_pass_count', '?')} "
            f"forced={row.get('forced_fill_count', '?')} batches={row.get('n_batches_used', '?')} "
            f"rejects={row.get('reject_counts', {})}",
            flush=True,
        )


def _split_even(items: list[dict], n_chunks: int) -> list[list[dict]]:
    chunks = [[] for _ in range(max(1, n_chunks))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [c for c in chunks if c]


def build_candidate_tasks(
    *,
    obj_ids: list[str],
    yaw_values_by_obj: dict[str, list[float]],
    trials_per_obj_yaw: int,
    result_dir: Path,
    mesh_root: str,
    dataset: str | None,
) -> list[dict]:
    tasks: list[dict] = []
    for obj_id in obj_ids:
        for yaw in yaw_values_by_obj[obj_id]:
            yaw_tag = int(round(float(yaw))) % 360
            out_dir = result_dir / "candidates" / obj_id
            out_path = out_dir / f"{obj_id}_yaw{yaw_tag:03d}_pool_grasp.hdf5"
            probe = type(
                "Probe",
                (),
                {
                    "obj_id": obj_id,
                    "dataset": dataset,
                    "mesh": None,
                    "mesh_root": mesh_root,
                    "sam3d_rotated_mesh": False,
                },
            )()
            mesh_path, _ = resolve_generate_mesh(probe)
            tasks.append(
                {
                    "obj_id": obj_id,
                    "z_yaw_deg": float(yaw),
                    "target_candidates": int(trials_per_obj_yaw),
                    "mesh_path": str(mesh_path),
                    "output_hdf5": str(out_path.resolve()),
                }
            )
    return tasks


def run_candidate_batch_generation(
    *,
    tasks: list[dict],
    result_dir: Path,
    mesh_root: str,
    dataset: str | None,
    candidate_python: str | None,
    candidate_gpu_ids: str | None,
    batch_multiplier: int,
    max_batches: int,
    object_scale: float,
    no_hard_gate: bool = False,
    no_filtering: bool = False,
    pdm_checkpoint: str | Path | None = None,
    pose_stats: str | Path | None = None,
    affordance_checkpoint: str | Path | None = None,
    candidate_workers: int | None = None,
    candidate_per_gpu: int | None = None,
    eval_seed: int | None = None,
) -> dict[tuple[str, float], str]:
    if not tasks:
        return {}
    existing = _load_existing_candidate_map(result_dir, tasks)
    if existing is not None:
        print(
            f"[candidate-batch] reusing {len(existing)} existing candidate pool(s) "
            f"under {result_dir / 'candidate_generation'}",
            flush=True,
        )
        work_dir = result_dir / "candidate_generation"
        all_rows: list[dict] = []
        for manifest_path in sorted(work_dir.glob("candidate_chunk_*_manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                all_rows.extend(payload.get("tasks", []))
            except (OSError, json.JSONDecodeError):
                pass
        _emit_mapping_logs(existing, all_rows)
        write_json(
            work_dir / "candidate_manifest.json",
            {
                "version": 1,
                "tasks": [
                    {"obj_id": k[0], "z_yaw_deg": k[1], "candidate_hdf5": v}
                    for k, v in sorted(existing.items())
                ],
            },
        )
        return existing

    gpu_ids = parse_gpu_ids(candidate_gpu_ids)
    if candidate_workers is not None:
        n_workers = max(1, int(candidate_workers))
    elif candidate_per_gpu is not None:
        n_workers = max(1, len(gpu_ids) * max(1, int(candidate_per_gpu)))
    else:
        n_workers = max(1, len(gpu_ids))
    chunks = _split_even(tasks, n_workers)
    work_dir = result_dir / "candidate_generation"
    work_dir.mkdir(parents=True, exist_ok=True)
    python_cmd = candidate_python or default_candidate_python()

    procs = []
    for idx, chunk in enumerate(chunks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        chunk_path = work_dir / f"candidate_chunk_{idx:03d}.json"
        manifest_path = work_dir / f"candidate_chunk_{idx:03d}_manifest.json"
        log_path = work_dir / f"candidate_chunk_{idx:03d}.log"
        write_json(chunk_path, {"tasks": chunk})
        cmd = [
            *shlex.split(python_cmd),
            str(PROJ / "tools" / "batch_pdm_candidates.py"),
            "--tasks-json",
            str(chunk_path),
            "--output-manifest",
            str(manifest_path),
            "--mesh-root",
            str(mesh_root),
            "--batch-multiplier",
            str(int(batch_multiplier)),
            "--max-batches",
            str(int(max_batches)),
            "--object-scale",
            str(float(object_scale)),
        ]
        if dataset:
            cmd.extend(["--dataset", str(dataset)])
        if no_hard_gate:
            cmd.append("--no-hard-gate")
        if no_filtering:
            cmd.append("--no-filtering")
        if pdm_checkpoint:
            cmd.extend(["--pdm-checkpoint", str(Path(pdm_checkpoint).expanduser().resolve())])
        if pose_stats:
            cmd.extend(["--pose-stats", str(Path(pose_stats).expanduser().resolve())])
        if affordance_checkpoint:
            cmd.extend(
                ["--affordance-checkpoint", str(Path(affordance_checkpoint).expanduser().resolve())]
            )
        if eval_seed is not None:
            cmd.extend(["--eval-seed", str(int(eval_seed))])
        env = candidate_generation_env()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_f = log_path.open("w", encoding="utf-8")
        print(
            f"[candidate-batch] worker {idx} gpu={gpu_id} tasks={len(chunk)} log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJ),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((idx, proc, log_f, manifest_path, log_path))

    mapping: dict[tuple[str, float], str] = {}
    for idx, proc, log_f, manifest_path, log_path in procs:
        rc = proc.wait()
        log_f.close()
        manifest_rows: list[dict] = []
        if manifest_path.is_file():
            try:
                with manifest_path.open(encoding="utf-8") as f:
                    manifest_rows = list(json.load(f).get("tasks", []))
            except (OSError, json.JSONDecodeError):
                manifest_rows = []

        if rc != 0:
            if _manifest_rows_valid(manifest_rows):
                print(
                    f"[candidate-batch] worker {idx} exited rc={rc} but manifest is complete; "
                    "treating as success (often CUDA teardown SIGSEGV)",
                    flush=True,
                )
            else:
                tail = ""
                try:
                    tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
                except Exception:
                    pass
                raise RuntimeError(
                    f"candidate batch worker {idx} failed rc={rc}\nLog tail:\n{tail}"
                )

        if not manifest_rows:
            raise RuntimeError(f"candidate batch worker {idx} produced no manifest: {manifest_path}")

        for row in manifest_rows:
            mapping[_task_key(row["obj_id"], row["z_yaw_deg"])] = str(
                Path(row["output_hdf5"]).resolve()
            )
        _emit_mapping_logs(
            {_task_key(r["obj_id"], r["z_yaw_deg"]): str(r["output_hdf5"]) for r in manifest_rows},
            manifest_rows,
        )
    write_json(
        work_dir / "candidate_manifest.json",
        {
            "version": 1,
            "tasks": [
                {"obj_id": k[0], "z_yaw_deg": k[1], "candidate_hdf5": v}
                for k, v in sorted(mapping.items())
            ],
        },
    )
    return mapping

