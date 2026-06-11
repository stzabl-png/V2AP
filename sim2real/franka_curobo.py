"""sim2real/franka_curobo.py — self-contained Franka CuRobo wrapper."""
from __future__ import annotations
import os, sys, threading, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from termcolor import cprint

SIM2REAL = Path(__file__).resolve().parent
PROJ     = SIM2REAL.parent
SIM_DIR  = PROJ / "sim"
for _p in [str(PROJ), str(SIM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# curobo src path
for _c in [os.path.expanduser("~/Project/curobo/src"),
           os.path.expanduser("~/curobo/src"),
           "/home/vision/Project/curobo/src"]:
    if os.path.isdir(os.path.join(_c, "curobo")):
        sys.path.insert(0, _c); break

from sim2real.scene_builder import (
    FRANKA_ROBOT_POSITION, FRANKA_ROBOT_ORIENTATION,
    TABLE_A_POSITION, TABLE_SCALE,
)

TCP_OFFSET    = 0.105
PRE_GRASP_OFF = 0.15
LIFT_HEIGHT   = 0.15
_MG = None


def get_franka_base_T():
    yaw = np.deg2rad(FRANKA_ROBOT_ORIENTATION[2])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    T[:3, 3] = FRANKA_ROBOT_POSITION
    return T, np.linalg.inv(T)


def world_to_robot(pos_w, quat_wxyz_w):
    _, Ti = get_franka_base_T()
    pos_r = (Ti @ np.append(pos_w, 1))[:3]
    Rw = Rotation.from_quat([quat_wxyz_w[1],quat_wxyz_w[2],quat_wxyz_w[3],quat_wxyz_w[0]])
    Ri = Rotation.from_matrix(Ti[:3,:3])
    q  = (Ri * Rw).as_quat()
    return pos_r, np.array([q[3],q[0],q[1],q[2]])


def _build_world_cfg(table_pos_r, ground_pos_r):
    return {"cuboid": {
        "table":  {"dims": list(TABLE_SCALE), "pose": [*table_pos_r.tolist(), 1,0,0,0]},
        "ground": {"dims": [20,20,0.1],       "pose": [*ground_pos_r.tolist(), 1,0,0,0]},
    }}


def init_franka_curobo():
    global _MG
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    extra = "/home/vision/isaacsim/kit/python/bin:/usr/local/cuda/bin"
    if extra not in os.environ.get("PATH",""):
        os.environ["PATH"] = extra + ":" + os.environ.get("PATH","")

    _, Ti = get_franka_base_T()
    # Table A position in robot base frame
    table_pos_r  = (Ti @ np.append(TABLE_A_POSITION, 1))[:3]
    ground_pos_r = (Ti @ np.array([0, 0, -0.025, 1]))[:3]

    # Use the same world config builder as the main pipeline
    try:
        from curobo_world import build_world_config_dict
        world_cfg = build_world_config_dict(table_pos_r, ground_pos_r, TABLE_SCALE)
    except Exception:
        # Fallback: inline build
        world_cfg = {"cuboid": {
            "table":  {"dims": list(TABLE_SCALE), "pose": [*table_pos_r.tolist(), 1,0,0,0]},
            "ground": {"dims": [5.0, 5.0, 0.01],  "pose": [*ground_pos_r.tolist(), 1,0,0,0]},
        }}

    cprint(f"   [franka_curobo] table_r={table_pos_r.round(3)}", "yellow")
    cprint("   [franka_curobo] loading MotionGenConfig…", "yellow"); t0=time.time()
    mg_cfg = MotionGenConfig.load_from_robot_config("franka.yml", world_cfg, interpolation_dt=0.02)
    mg     = MotionGen(mg_cfg)

    done, exc_ = threading.Event(), []
    def _wu():
        try: mg.warmup()
        except Exception as e: exc_.append(e)
        finally: done.set()
    def _hb():
        t1=time.time()
        while not done.is_set():
            done.wait(10)
            if not done.is_set():
                print(f"   [warmup] {int(time.time()-t1)}s", flush=True)
    threading.Thread(target=_wu,daemon=True).start()
    threading.Thread(target=_hb,daemon=True).start()
    done.wait()
    if exc_: raise RuntimeError(f"warmup failed: {exc_[0]}")
    cprint(f"   [franka_curobo] ready ✅ ({time.time()-t0:.1f}s)","green")
    _MG = mg
    return mg


def plan_franka(mg, current_q7, pos_w, quat_wxyz_w, label=""):
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CJS
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    pos_r, quat_r = world_to_robot(pos_w, quat_wxyz_w)
    start = CJS.from_position(
        torch.tensor(current_q7, dtype=torch.float32).unsqueeze(0).cuda(),
    )
    goal = Pose.from_list([*pos_r.tolist(), *quat_r.tolist()])
    res  = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=10, enable_graph=True, enable_opt=True))
    ok   = res.success.item() if hasattr(res.success,"item") else bool(res.success)
    if ok:
        traj = res.get_interpolated_plan()
        cprint(f"      [{label}] Franka OK: {traj.position.shape[0]} steps","green")
        return traj.position.cpu().numpy()
    cprint(f"      [{label}] Franka FAILED","red")
    return None
