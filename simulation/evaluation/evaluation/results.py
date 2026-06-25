"""Result writers for evaluation episodes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from evaluation.specs import ExecutionResult, PolicyOutput, SceneSpec


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_episode_record(
    *,
    scene: SceneSpec,
    policy_name: str,
    policy_output: PolicyOutput,
    execution: ExecutionResult,
    video_path: str | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_id": scene.episode_id,
        "obj_id": scene.obj_id,
        "policy": policy_name,
        "success": bool(execution.success),
        "failure_stage": execution.failure_stage,
        "z_delta_m": execution.z_delta_m,
        "scene": scene.to_dict(),
        "policy_output": policy_output.to_dict(),
        "execution": execution.to_dict(),
    }
    if video_path:
        record["video_path"] = video_path
    elif execution.video_path:
        record["video_path"] = execution.video_path
    return record


def write_episode_json(record: dict[str, Any], result_dir: str) -> str:
    _ensure_dir(result_dir)
    episode_id = str(record["episode_id"])
    path = os.path.join(result_dir, f"{episode_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def append_episode_jsonl(record: dict[str, Any], result_dir: str, filename: str = "episodes.jsonl") -> str:
    _ensure_dir(result_dir)
    path = os.path.join(result_dir, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path

