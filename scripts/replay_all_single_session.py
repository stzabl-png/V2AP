#!/usr/bin/env python3
"""Replay 97 successful HP+PDM grasps in a single Isaac Sim session.

Usage:
    /home/lyh/isaac-sim-5.0/python.sh scripts/replay_all_single_session.py

Flow:
    1. Start Isaac Sim (GUI)
    2. Build scene with first object
    3. Wait for Enter
    4. Execute grasp → swap object → execute grasp → ... (97 objects)
    5. Close
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ── Parse minimal args before SimulationApp ──
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ── Now safe to import isaacsim modules ──
from termcolor import cprint

from evaluation.policies.a2g_pdm import A2GPDMPolicy, A2GPDMPolicyConfig
from sim.evaluation.curobo_executor import execute_open_loop_grasp
from sim.evaluation.scene_builder import build_scene_spec, setup_scene, swap_scene_object

# ── Load task list ──
TASK_FILE = PROJ / "scripts" / "replay_tasks.json"
with open(TASK_FILE, encoding="utf-8") as f:
    TASKS = json.load(f)

cprint(f"\n{'='*60}", "cyan")
cprint(f"  HP+DiffusionPose Grasp Replay — {len(TASKS)} objects", "cyan")
cprint(f"  Single Isaac Sim session, no restart", "cyan")
cprint(f"{'='*60}\n", "cyan")

# ── Build first scene ──
first = TASKS[0]
spec = build_scene_spec(
    obj_id=first["obj_id"],
    episode_id=first["episode_id"],
    dataset=None,
    object_scale=1.0,
    sim_z_yaw_deg=float(first["z_yaw_deg"]),
    seed=0,
    candidate_hdf5=first["candidate_hdf5"],
    random_obj_xy=True,
    obj_xy_jitter_m=0.05,
    obj_xy_offset=first["obj_xy_offset"],
    eval_seed=42,
)
scene = setup_scene(spec, render=True)
cprint(f"\n[1/{len(TASKS)}] {first['obj_id']} loaded — press Enter to start all grasps...", "yellow")

# ── Wait for Enter (keep rendering) ──
import select
sys.stdin = open("/dev/stdin")
while True:
    scene.world.step(render=True)
    if select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()
        break

cprint("Starting replay sequence...\n", "green")

# ── Execute grasps ──
n_success = 0
n_fail = 0

for i, task in enumerate(TASKS):
    obj_id = task["obj_id"]
    ep_id = task["episode_id"]
    yaw = float(task["z_yaw_deg"])
    hdf5 = task["candidate_hdf5"]
    cidx = task["candidate_index"]

    cprint(f"\n{'─'*50}", "cyan")
    cprint(f"  [{i+1}/{len(TASKS)}] {obj_id}  (yaw={int(yaw)}°)", "cyan")
    cprint(f"{'─'*50}", "cyan")

    # Swap object if not the first
    if i > 0:
        new_spec = build_scene_spec(
            obj_id=obj_id,
            episode_id=ep_id,
            dataset=None,
            object_scale=1.0,
            sim_z_yaw_deg=yaw,
            seed=0,
            candidate_hdf5=hdf5,
            random_obj_xy=True,
            obj_xy_jitter_m=0.05,
            obj_xy_offset=task["obj_xy_offset"],
            eval_seed=42,
        )
        try:
            swap_scene_object(scene, new_spec)
        except Exception as e:
            cprint(f"  ❌ swap failed: {e}", "red")
            n_fail += 1
            continue

    # Load policy
    try:
        policy = A2GPDMPolicy(
            A2GPDMPolicyConfig(
                candidate_hdf5=hdf5,
                selection="index",
                candidate_index=cidx,
                seed=42,
            )
        )
        policy_output = policy.predict(scene)
        if policy_output.kind != "open_loop_grasp" or policy_output.command is None:
            cprint(f"  ❌ policy output invalid", "red")
            n_fail += 1
            continue
    except Exception as e:
        cprint(f"  ❌ policy error: {e}", "red")
        n_fail += 1
        continue

    # Execute grasp
    try:
        execution = execute_open_loop_grasp(scene, policy_output.command)
        if execution.success:
            cprint(f"  ✅ SUCCESS  z_delta={execution.z_delta_m:.3f}m", "green")
            n_success += 1
        else:
            cprint(f"  ❌ FAILED   stage={execution.failure_stage}", "red")
            n_fail += 1
    except Exception as e:
        cprint(f"  ❌ execution error: {e}", "red")
        n_fail += 1

cprint(f"\n{'='*60}", "cyan")
cprint(f"  Done! {n_success} success, {n_fail} failed out of {len(TASKS)}", "cyan")
cprint(f"{'='*60}\n", "cyan")

# Keep window open briefly then close
for _ in range(120):
    scene.world.step(render=True)

simulation_app.close()
