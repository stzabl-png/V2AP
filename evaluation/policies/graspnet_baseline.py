"""GraspNet-Baseline policy adapter for eval_pool.

Reads pre-generated GraspNet candidate HDF5 files exactly as A2GPDMPolicy does.
The HDF5 format is identical — this adapter only differs in name and how
candidates are located on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np

from evaluation.policies.base import EvaluationPolicy
from evaluation.specs import OpenLoopGraspCommand, PolicyOutput

# Re-use the existing candidate loader (works with any compliant HDF5)
from evaluation.policies.a2g_pdm import (
    load_candidate_hdf5,
    select_candidate,
    SelectionMode,
)


@dataclass
class GraspNetBaselinePolicyConfig:
    candidate_hdf5: str
    selection: SelectionMode = "top"
    candidate_index: int = 0
    seed: int | None = None


class GraspNetBaselinePolicy(EvaluationPolicy):
    """Evaluate pre-computed GraspNet-Baseline grasps in IsaacSim."""

    name = "graspnet_baseline"

    def __init__(self, config: GraspNetBaselinePolicyConfig):
        self.config = config

    def predict(self, context) -> PolicyOutput:
        candidates, metadata = load_candidate_hdf5(self.config.candidate_hdf5)
        if not candidates:
            # Empty candidate file (e.g. all grasps removed by collision detection).
            # Return a dummy command that will fail in sim — recorded as a failure.
            command = OpenLoopGraspCommand(
                position=np.array([0.0, 0.0, -999.0]),  # unreachable
                rotation=np.eye(3),
                gripper_width=0.08,
                frame="object_mesh",
                ee_frame_convention="a2g_grasp_frame",
                name="no_candidate",
                score=0.0,
                approach_type="",
                is_manual=False,
                mesh_prerotation_euler=[0.0, 0.0, 0.0],
                mesh_prerotation=None,
                metadata={
                    "candidate_hdf5": os.path.abspath(self.config.candidate_hdf5),
                    "empty": True,
                },
            )
            return PolicyOutput(
                kind="open_loop_grasp",
                command=command,
                metadata={
                    "policy_name": self.name,
                    "selection": self.config.selection,
                    "n_candidates": 0,
                    "empty_candidate": True,
                },
            )
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
