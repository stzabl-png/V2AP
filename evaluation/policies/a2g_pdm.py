"""A2G affordance + PDM policy adapter.

The first evaluation version reads an existing candidate HDF5 and returns one
open-loop grasp command. `evaluation/eval_single.py` can optionally generate
that HDF5 before IsaacSim starts by calling the existing mesh-to-PDM script.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np

from evaluation.policies.base import EvaluationPolicy
from evaluation.specs import OpenLoopGraspCommand, PolicyOutput

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(PROJ, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


SelectionMode = Literal["top", "index", "sample"]


@dataclass
class A2GPDMPolicyConfig:
    candidate_hdf5: str
    selection: SelectionMode = "top"
    candidate_index: int = 0
    seed: int | None = None


class A2GPDMPolicy(EvaluationPolicy):
    name = "a2g_pdm"

    def __init__(self, config: A2GPDMPolicyConfig):
        self.config = config

    def predict(self, context) -> PolicyOutput:
        candidates, metadata = load_candidate_hdf5(self.config.candidate_hdf5)
        if not candidates:
            raise RuntimeError(f"no grasp candidates found in {self.config.candidate_hdf5}")
        selected = select_candidate(
            candidates,
            mode=self.config.selection,
            index=self.config.candidate_index,
            seed=self.config.seed,
        )
        command = OpenLoopGraspCommand(
            position=np.asarray(selected["position"], dtype=np.float64),
            rotation=np.asarray(selected["rotation"], dtype=np.float64),
            gripper_width=float(selected["gripper_width"]),
            frame="object_mesh",
            ee_frame_convention="a2g_grasp_frame",
            name=str(selected["name"]),
            score=float(selected["score"]),
            approach_type=str(selected.get("approach_type", "")),
            is_manual=bool(selected.get("is_manual", False)),
            mesh_prerotation_euler=selected.get("mesh_prerotation_euler"),
            mesh_prerotation=selected.get("mesh_prerotation"),
            metadata={
                "candidate_hdf5": os.path.abspath(self.config.candidate_hdf5),
                "candidate_rank": int(selected.get("rank", -1)),
                "candidate_source": metadata,
            },
        )
        return PolicyOutput(
            kind="open_loop_grasp",
            command=command,
            metadata={
                "policy_name": self.name,
                "selection": self.config.selection,
                "n_candidates": len(candidates),
            },
        )


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _read_gripper_width(ci: h5py.Group) -> float:
    if "gripper_width" in ci.attrs:
        return float(ci.attrs["gripper_width"])
    if "gripper_width" in ci:
        return float(np.asarray(ci["gripper_width"][()]).reshape(()))
    return 0.06


def _mesh_utils():
    from mesh_utils import (
        applied_mesh_prerotation_record,
        read_mesh_prerotation_hdf5_pose,
    )

    return applied_mesh_prerotation_record, read_mesh_prerotation_hdf5_pose


def _read_candidate_group(
    ci: h5py.Group,
    file_handle: h5py.File,
    fallback_prerot,
    read_mesh_prerotation_hdf5_pose,
) -> dict:
    pose_pr = read_mesh_prerotation_hdf5_pose(ci, file_handle) or fallback_prerot
    return {
        "name": _as_str(ci.attrs.get("name", ci.name.rsplit("/", 1)[-1])),
        "score": float(ci.attrs.get("score", 0.0)),
        "position": ci["position"][:],
        "rotation": ci["rotation"][:],
        "gripper_width": _read_gripper_width(ci),
        "approach_type": _as_str(ci.attrs.get("approach_type", "")),
        "is_manual": False,
        "mesh_prerotation_euler": list(pose_pr["euler_xyz_deg"]) if pose_pr else [0.0, 0.0, 0.0],
        "mesh_prerotation": pose_pr,
    }


_CANDIDATE_HDF5_CACHE: dict[str, tuple[list[dict], dict]] = {}


def clear_candidate_hdf5_cache() -> None:
    """Drop cached pool reads (e.g. after regenerating candidates)."""
    _CANDIDATE_HDF5_CACHE.clear()


def load_candidate_hdf5(path: str, *, use_cache: bool = True) -> tuple[list[dict], dict]:
    """Load all supported candidate HDF5 layouts used by the project."""
    abs_path = os.path.abspath(path)
    if use_cache and abs_path in _CANDIDATE_HDF5_CACHE:
        return _CANDIDATE_HDF5_CACHE[abs_path]

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    applied_mesh_prerotation_record, read_mesh_prerotation_hdf5_pose = _mesh_utils()
    candidates: list[dict] = []
    with h5py.File(path, "r") as f:
        if "metadata" in f:
            meta_group = f["metadata"]
            obj_id = _as_str(meta_group.attrs.get("obj_id", f.attrs.get("obj_id", "unknown")))
            dataset = _as_str(meta_group.attrs.get("dataset", ""))
            no_rotation = bool(meta_group.attrs.get("no_rotation", True))
        else:
            obj_id = _as_str(f.attrs.get("object_id", f.attrs.get("obj_id", "unknown")))
            dataset = ""
            no_rotation = True

        fallback_prerot = applied_mesh_prerotation_record(
            obj_id,
            dataset or None,
            no_rotation=no_rotation,
        )

        if "candidates" in f:
            cg = f["candidates"]
            keys = sorted(
                [k for k in cg.keys() if k.startswith("candidate_")],
                key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k,
            )
            for key in keys:
                candidates.append(
                    _read_candidate_group(
                        cg[key],
                        f,
                        fallback_prerot,
                        read_mesh_prerotation_hdf5_pose,
                    )
                )
        elif "candidate_0" in f:
            n_cand = int(f.attrs.get("num_candidates", 0))
            for i in range(n_cand):
                key = f"candidate_{i}"
                if key not in f:
                    continue
                item = _read_candidate_group(
                    f[key],
                    f,
                    fallback_prerot,
                    read_mesh_prerotation_hdf5_pose,
                )
                item["is_manual"] = True
                candidates.append(item)
        elif "grasp" in f:
            gi = f["grasp"]
            candidates.append(
                _read_candidate_group(
                    gi,
                    f,
                    fallback_prerot,
                    read_mesh_prerotation_hdf5_pose,
                )
            )

    candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    for rank, cand in enumerate(candidates):
        cand["rank"] = rank

    metadata = {
        "obj_id": obj_id,
        "dataset": dataset,
        "no_rotation": no_rotation,
    }
    result = (candidates, metadata)
    if use_cache:
        _CANDIDATE_HDF5_CACHE[abs_path] = result
    return result


def select_candidate(
    candidates: list[dict],
    *,
    mode: SelectionMode,
    index: int,
    seed: int | None,
) -> dict:
    if mode == "top":
        return candidates[0]
    if mode == "index":
        # Wrap around with modulo so trial_N cycles through available candidates
        # e.g. 5 candidates + 10 trials: trial_5 → candidate_0, trial_6 → candidate_1…
        wrapped = index % len(candidates)
        return candidates[wrapped]
    if mode == "sample":
        scores = np.asarray([max(float(c.get("score", 0.0)), 0.0) for c in candidates], dtype=np.float64)
        if np.all(scores <= 0):
            probs = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
        else:
            probs = scores / scores.sum()
        from evaluation.randomness import fresh_rng

        rng = fresh_rng() if seed is None else np.random.default_rng(seed)
        return candidates[int(rng.choice(len(candidates), p=probs))]
    raise ValueError(f"unknown selection mode: {mode}")

