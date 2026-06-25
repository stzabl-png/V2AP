"""cuRobo-backed open-loop grasp executor for evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from termcolor import cprint

from evaluation.specs import ExecutionResult, OpenLoopGraspCommand, SceneSpec
from sim.evaluation.context import SimEvaluationContext
from sim.evaluation.scene_builder import (
    ROBOT_ORIENTATION,
    ROBOT_POSITION,
    TABLE_POSITION,
    TABLE_SCALE,
    TABLE_TOP_Z,
)

SIM_DIR = Path(__file__).resolve().parents[1]
PROJ_DIR = SIM_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
for _curobo_candidate in [
    os.path.expanduser("~/Project/curobo/src"),
    os.path.expanduser("~/curobo/src"),
    "/home/vision/Project/curobo/src",
]:
    if os.path.isdir(os.path.join(_curobo_candidate, "curobo")):
        sys.path.insert(0, _curobo_candidate)
        break

from curobo_world import build_world_config_dict, object_pose_robot_frame, sync_curobo_world  # noqa: E402


LIFT_HEIGHT = 0.15
TCP_OFFSET = 0.105
PRE_GRASP_OFFSET = 0.15
_CUROBO_MG = None


def reset_motion_gen() -> None:
    """Force cuRobo MotionGen rebuild after object/yaw/world changes."""
    global _CUROBO_MG
    _CUROBO_MG = None


def make_transform(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    return T


def get_robot_base_transform() -> tuple[np.ndarray, np.ndarray]:
    yaw_rad = np.deg2rad(ROBOT_ORIENTATION[2])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = ROBOT_POSITION
    return T, np.linalg.inv(T)


def world_to_robot_pose(pos_w, quat_wxyz_w, T_robot_world):
    pos_r = (T_robot_world @ np.append(pos_w, 1.0))[:3]
    R_w = Rotation.from_quat([quat_wxyz_w[1], quat_wxyz_w[2], quat_wxyz_w[3], quat_wxyz_w[0]])
    R_rw = Rotation.from_matrix(T_robot_world[:3, :3])
    R_r = R_rw * R_w
    q = R_r.as_quat()
    return pos_r, np.array([q[3], q[0], q[1], q[2]])


def transform_grasp_to_world(grasp_pos_obj, grasp_rot_obj, T_world_obj):
    pos_w = (T_world_obj @ np.append(grasp_pos_obj, 1.0))[:3]
    rot_w = T_world_obj[:3, :3] @ grasp_rot_obj
    return pos_w, rot_w


def world_pose_to_object_mesh(pos_world, quat_wxyz_world, obj_pos_world, obj_quat_wxyz, object_scale):
    T_world_obj = make_transform(
        np.asarray(obj_pos_world, dtype=np.float64).reshape(3),
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4),
    )
    T_world_body = make_transform(
        np.asarray(pos_world, dtype=np.float64).reshape(3),
        np.asarray(quat_wxyz_world, dtype=np.float64).reshape(4),
    )
    T_obj_body = np.linalg.inv(T_world_obj) @ T_world_body
    scale = float(object_scale) if object_scale else 1.0
    return (T_obj_body[:3, 3] / scale).astype(np.float32), T_obj_body[:3, :3].astype(np.float32)


def world_point_to_object_mesh(point_world, obj_pos_world, obj_quat_wxyz, object_scale):
    T_world_obj = make_transform(
        np.asarray(obj_pos_world, dtype=np.float64).reshape(3),
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4),
    )
    p_h = np.append(np.asarray(point_world, dtype=np.float64).reshape(3), 1.0)
    scale = float(object_scale) if object_scale else 1.0
    return ((np.linalg.inv(T_world_obj) @ p_h)[:3] / scale).astype(np.float32)


def snapshot_panda_hand_object_mesh(franka, scene: SimEvaluationContext) -> dict[str, Any]:
    pos_w, quat_w = franka.get_cur_ee_pos(local_frame=False)
    obj_pos, obj_quat = scene.obj.get_obj_pos()
    pos_o, rot_o = world_pose_to_object_mesh(pos_w, quat_w, obj_pos, obj_quat, scene.spec.object_scale)
    return {
        "position": pos_o,
        "rotation": rot_o,
        "approach_dir": rot_o[:, 2].copy(),
        "finger_dir": rot_o[:, 1].copy(),
    }


def snapshot_gripper_tips_object_mesh(stage, scene: SimEvaluationContext) -> dict[str, Any]:
    from pxr import UsdGeom

    left_path = "/World/Franka/panda_leftfinger"
    right_path = "/World/Franka/panda_rightfinger"
    left_prim = stage.GetPrimAtPath(left_path)
    right_prim = stage.GetPrimAtPath(right_path)
    if not left_prim.IsValid() or not right_prim.IsValid():
        raise RuntimeError("finger prims not found")

    left_world = np.array(
        UsdGeom.Xformable(left_prim).ComputeLocalToWorldTransform(0).ExtractTranslation(),
        dtype=np.float64,
    )
    right_world = np.array(
        UsdGeom.Xformable(right_prim).ComputeLocalToWorldTransform(0).ExtractTranslation(),
        dtype=np.float64,
    )
    obj_pos, obj_quat = scene.obj.get_obj_pos()
    left_o = world_point_to_object_mesh(left_world, obj_pos, obj_quat, scene.spec.object_scale)
    right_o = world_point_to_object_mesh(right_world, obj_pos, obj_quat, scene.spec.object_scale)
    tips = np.stack([left_o, right_o]).astype(np.float32)
    return {
        "gripper_tips_loc": tips,
        "finger_width_actual": float(np.linalg.norm(tips[0] - tips[1])),
    }


def _base_result(success=False, failure_stage: str | None = None) -> ExecutionResult:
    return ExecutionResult(success=bool(success), failure_stage=failure_stage)


def init_curobo(scene: SimEvaluationContext):
    import os as _os
    import subprocess
    import threading
    import time as _time
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    extra = "/home/vision/isaacsim/kit/python/bin:/usr/local/cuda/bin"
    if extra not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = extra + ":" + _os.environ.get("PATH", "")

    _, T_robot_world = get_robot_base_transform()
    table_pos_r = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    ground_pos_r = (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3]
    scene.metadata["curobo_table_pos_r"] = table_pos_r
    scene.metadata["curobo_ground_pos_r"] = ground_pos_r

    mesh_verts = mesh_faces = mesh_pose = None
    if scene.curobo_mesh_vertices is not None:
        mesh_verts = scene.curobo_mesh_vertices
        mesh_faces = scene.curobo_mesh_faces
        pos_w, quat_wxyz = scene.obj.get_obj_pos()
        mesh_pose = object_pose_robot_frame(pos_w, quat_wxyz, T_robot_world)

    world_config = build_world_config_dict(
        table_pos_r,
        ground_pos_r,
        TABLE_SCALE,
        mesh_vertices=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_pose_robot=mesh_pose,
    )
    load_kwargs = {"interpolation_dt": 0.02}
    if mesh_verts is not None:
        load_kwargs["collision_cache"] = {"obb": 4, "mesh": 4}

    cprint("   [curobo-init] loading MotionGenConfig...", "yellow"); _t0 = _time.time()
    mg_config = MotionGenConfig.load_from_robot_config("franka.yml", world_config, **load_kwargs)
    cprint(f"   [curobo-init] MotionGenConfig loaded in {_time.time()-_t0:.1f}s", "yellow")

    cprint("   [curobo-init] building MotionGen...", "yellow"); _t1 = _time.time()
    mg = MotionGen(mg_config)
    cprint(f"   [curobo-init] MotionGen built in {_time.time()-_t1:.1f}s", "yellow")

    # --- warmup with heartbeat every 10s and timeout ---
    WARMUP_TIMEOUT_S = int(_os.environ.get("CUROBO_WARMUP_TIMEOUT", "1200"))  # 20 min default
    cprint(f"   [curobo-init] warmup() starting (timeout={WARMUP_TIMEOUT_S}s)...", "yellow")
    _warmup_done = threading.Event()
    _warmup_exc: list[Exception] = []

    def _gpu_mem() -> str:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"], timeout=3, text=True
            ).strip().split("\n")[0]
            used, free, util = out.split(",")
            return f"GPU mem={used.strip()}MiB free={free.strip()}MiB util={util.strip()}%"
        except Exception:
            return ""

    def _run_warmup():
        try:
            mg.warmup()
        except Exception as exc:
            _warmup_exc.append(exc)
        finally:
            _warmup_done.set()

    def _heartbeat():
        t0 = _time.time()
        tick = 0
        while not _warmup_done.is_set():
            _warmup_done.wait(timeout=10)
            if not _warmup_done.is_set():
                tick += 1
                elapsed = int(_time.time() - t0)
                gpu_str = _gpu_mem() if tick % 3 == 0 else ""  # sample GPU every 30s
                print(f"   [warmup] {elapsed}s elapsed {gpu_str}", flush=True)
                if elapsed >= WARMUP_TIMEOUT_S:
                    cprint(f"   [warmup] TIMEOUT after {elapsed}s — killing process", "red")
                    _os.kill(_os.getpid(), 9)  # hard kill

    wt = threading.Thread(target=_run_warmup, daemon=True)
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    wt.start()
    wt.join()  # block until warmup finishes or timeout kills us
    _warmup_done.set()  # stop heartbeat

    if _warmup_exc:
        raise RuntimeError(f"cuRobo warmup failed: {_warmup_exc[0]}") from _warmup_exc[0]

    cprint(f"   [curobo-init] cuRobo ready ✅  (total init {_time.time()-_t0:.1f}s)", "green")
    return mg


def _sync_curobo_world_for_scene(
    motion_gen,
    scene: SimEvaluationContext,
    *,
    include_object_mesh: bool,
) -> np.ndarray:
    """Sync cuRobo collision world (table/ground, optional object mesh). Returns T_robot_world."""
    _, T_robot_world = get_robot_base_transform()
    legacy_scene = scene.as_legacy_dict()
    table_pos_r = scene.metadata.get(
        "curobo_table_pos_r",
        (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3],
    )
    ground_pos_r = scene.metadata.get(
        "curobo_ground_pos_r",
        (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3],
    )
    sync_curobo_world(
        motion_gen,
        legacy_scene,
        table_pos_r,
        ground_pos_r,
        TABLE_SCALE,
        T_robot_world,
        include_object_mesh=include_object_mesh and scene.curobo_mesh_vertices is not None,
    )
    scene.update_from_legacy_dict(legacy_scene)
    return T_robot_world


def plan_trajectory(
    motion_gen,
    scene: SimEvaluationContext,
    target_pos_world,
    target_quat_wxyz_world,
    *,
    label: str,
    use_object_mesh: bool,
):
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    T_robot_world = _sync_curobo_world_for_scene(
        motion_gen, scene, include_object_mesh=use_object_mesh
    )

    pos_r, quat_r = world_to_robot_pose(target_pos_world, target_quat_wxyz_world, T_robot_world)
    current_joints = scene.franka.get_joint_positions()[:7]
    start_state = CuJointState.from_position(
        torch.tensor(current_joints, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )
    goal_pose = Pose.from_list(
        [
            float(pos_r[0]),
            float(pos_r[1]),
            float(pos_r[2]),
            float(quat_r[0]),
            float(quat_r[1]),
            float(quat_r[2]),
            float(quat_r[3]),
        ]
    )
    result = motion_gen.plan_single(
        start_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=10, enable_graph=True, enable_opt=True),
    )
    success = result.success.item() if callable(getattr(result.success, "item", None)) else result.success
    if success:
        traj = result.get_interpolated_plan()
        cprint(f"      [{label}] plan OK: {traj.position.shape[0]} steps", "green")
        return traj.position.cpu().numpy()
    cprint(f"      [{label}] plan failed", "red")
    return None


def _try_final_straight_approach(
    motion_gen,
    scene: SimEvaluationContext,
    *,
    target_pos_world: np.ndarray,
    target_quat_wxyz_world: np.ndarray,
    hold_steps: int = 2,
) -> bool:
    """Final straight approach via cuRobo constrained planning (same collision world as final).

    Syncs table + ground only (no object mesh), then runs ``plan_single`` with
    ``PoseCostMetric`` path constraints (same idea as ``MotionGen.plan_grasp`` approach
    segment): orientation and lateral axes held in the grasp frame, motion along the
    approach axis. Includes world + self-collision checking via MotionGen.
    """
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    franka = scene.franka
    world = scene.world
    render = scene.render

    T_robot_world = _sync_curobo_world_for_scene(motion_gen, scene, include_object_mesh=False)

    target_pos_world = np.asarray(target_pos_world, dtype=np.float64).reshape(3)
    target_quat_wxyz_world = np.asarray(target_quat_wxyz_world, dtype=np.float64).reshape(4)
    pos_r, quat_r = world_to_robot_pose(target_pos_world, target_quat_wxyz_world, T_robot_world)
    goal_pose = Pose.from_list(
        [
            float(pos_r[0]),
            float(pos_r[1]),
            float(pos_r[2]),
            float(quat_r[0]),
            float(quat_r[1]),
            float(quat_r[2]),
            float(quat_r[3]),
        ]
    )

    cur_q = franka.get_joint_positions()[:7]
    start_state = CuJointState.from_position(
        torch.tensor(cur_q, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )

    # Default from MotionGen.plan_grasp: lock orientation + lateral motion in grasp frame,
    # free axis aligned with approach (index 5 in [ox,oy,oz, px,py,pz] weight vector).
    approach_constraint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    hold_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=motion_gen.tensor_args.to_device(approach_constraint),
        project_to_goal_frame=True,
    )
    plan_config = MotionGenPlanConfig(
        max_attempts=10,
        enable_graph=True,
        enable_opt=True,
        pose_cost_metric=hold_metric,
    )
    try:
        result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    finally:
        motion_gen.update_pose_cost_metric(PoseCostMetric.reset_metric())

    ok = result.success.item() if hasattr(result.success, "item") else bool(result.success)
    if not ok:
        cprint("      [final-straight] cuRobo constrained approach plan failed", "yellow")
        return False

    traj = result.get_interpolated_plan()
    for joint_pos in traj.position.cpu().numpy():
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        for _ in range(int(max(1, hold_steps))):
            world.step(render=render)
    cprint(f"      [final-straight] cuRobo constrained plan OK: {traj.position.shape[0]} steps", "green")
    return True


def _command_to_world_target(scene: SimEvaluationContext, command: OpenLoopGraspCommand):
    if command.frame != "object_mesh":
        raise ValueError(f"first executor only supports object_mesh commands, got {command.frame}")

    obj_pos_world, obj_quat_wxyz = scene.obj.get_obj_pos()
    T_world_obj = make_transform(obj_pos_world, obj_quat_wxyz)
    grasp_pos_scaled = np.asarray(command.position, dtype=np.float64) * scene.spec.object_scale
    grasp_rot_obj = np.asarray(command.rotation, dtype=np.float64)

    prerot = command.mesh_prerotation_euler
    if prerot and any(abs(float(e)) > 0.5 for e in prerot):
        Rp = Rotation.from_euler("xyz", prerot, degrees=True).as_matrix()
        T_eff = T_world_obj.copy()
        T_eff[:3, :3] = T_world_obj[:3, :3] @ Rp.T
    else:
        T_eff = T_world_obj

    pos_world, rot_world = transform_grasp_to_world(grasp_pos_scaled, grasp_rot_obj, T_eff)
    r_adapt = np.array(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    rot_world = rot_world @ r_adapt
    approach_dir = rot_world[:, 2]
    pos_world = pos_world - approach_dir * TCP_OFFSET
    min_grasp_z = TABLE_TOP_Z + 0.02
    if pos_world[2] < min_grasp_z:
        cprint(f"   grasp target z={pos_world[2]:.3f} below {min_grasp_z:.3f}; clamping", "yellow")
        pos_world[2] = min_grasp_z
    q_xyzw = Rotation.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return pos_world, rot_world, quat_wxyz


def execute_open_loop_grasp(scene: SimEvaluationContext, command: OpenLoopGraspCommand) -> ExecutionResult:
    """Execute one open-loop object-mesh grasp command and return a common result."""
    global _CUROBO_MG

    franka = scene.franka
    world = scene.world
    render = scene.render
    planning = {
        "pregrasp_plan_success": False,
        "direct_plan_success": False,
        "final_plan_success": False,
        "final_plan_mode": "",
        "lift_plan_success": False,
    }

    obj_init, _ = scene.obj.get_obj_pos()
    initial_z = float(obj_init[2])

    try:
        pos_world, rot_world, quat_wxyz = _command_to_world_target(scene, command)
    except Exception as exc:
        res = _base_result(False, "target_transform")
        res.metadata["error"] = str(exc)
        return res

    lift_pos = pos_world.copy()
    lift_pos[2] += LIFT_HEIGHT
    approach_dir = rot_world[:, 2]
    pre_grasp_pos = pos_world - approach_dir * PRE_GRASP_OFFSET

    try:
        if _CUROBO_MG is None:
            _CUROBO_MG = init_curobo(scene)
    except Exception as exc:
        res = _base_result(False, "curobo_init")
        res.metadata["error"] = str(exc)
        return res

    franka.open_gripper()
    for _ in range(30):
        world.step(render=render)

    traj = plan_trajectory(
        _CUROBO_MG,
        scene,
        pre_grasp_pos,
        quat_wxyz,
        label="pre-grasp",
        use_object_mesh=True,
    )
    if traj is not None:
        planning["pregrasp_plan_success"] = True
    else:
        traj = plan_trajectory(
            _CUROBO_MG,
            scene,
            pos_world,
            quat_wxyz,
            label="direct",
            use_object_mesh=True,
        )
        planning["direct_plan_success"] = traj is not None
    if traj is None:
        res = _base_result(False, "pregrasp_plan")
        res.planning = planning
        return res

    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=render)
    for _ in range(10):
        world.step(render=render)

    # Final approach: prefer a straight-line EE approach, then fall back to cuRobo.
    straight_ok = False
    try:
        straight_ok = _try_final_straight_approach(
            _CUROBO_MG,
            scene,
            target_pos_world=pos_world,
            target_quat_wxyz_world=quat_wxyz,
            hold_steps=2,
        )
    except Exception as exc:
        cprint(f"      [final-straight] exception: {exc}", "yellow")
        straight_ok = False

    if straight_ok:
        planning["final_plan_success"] = True
        planning["final_plan_mode"] = "straight"
    else:
        traj_final = plan_trajectory(
            _CUROBO_MG,
            scene,
            pos_world,
            quat_wxyz,
            label="final",
            use_object_mesh=False,
        )
        planning["final_plan_success"] = traj_final is not None
        planning["final_plan_mode"] = "curobo" if traj_final is not None else ""
        if traj_final is None:
            res = _base_result(False, "final_plan")
            res.planning = planning
            return res
        for joint_pos in traj_final:
            gripper = franka.get_joint_positions()[7:9]
            franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
            for _ in range(3):
                world.step(render=render)

    franka.close_gripper()
    force_log = []
    for _ in range(80):
        world.step(render=render)
        force_log.append(franka.get_joint_positions()[7:9].copy())

    executed_at_close = None
    gripper_tips_loc = None
    finger_width_actual = None
    try:
        executed_at_close = snapshot_panda_hand_object_mesh(franka, scene)
    except Exception as exc:
        cprint(f"   panda_hand@close snapshot failed: {exc}", "yellow")
    try:
        tips = snapshot_gripper_tips_object_mesh(world.stage, scene)
        gripper_tips_loc = tips["gripper_tips_loc"]
        finger_width_actual = tips["finger_width_actual"]
    except Exception as exc:
        cprint(f"   gripper tips snapshot failed: {exc}", "yellow")

    traj_lift = plan_trajectory(
        _CUROBO_MG,
        scene,
        lift_pos,
        quat_wxyz,
        label="lift",
        use_object_mesh=False,
    )
    planning["lift_plan_success"] = traj_lift is not None
    if traj_lift is not None:
        franka.close_gripper()
        for joint_pos in traj_lift:
            from omni.isaac.core.utils.types import ArticulationAction

            action = ArticulationAction(
                joint_positions=np.concatenate([joint_pos, np.array([None, None])]),
            )
            franka.apply_action(action)
            for _ in range(2):
                world.step(render=render)

    for _ in range(80):
        world.step(render=render)

    obj_after, _ = scene.obj.get_obj_pos()
    z_delta = float(obj_after[2] - initial_z)
    success = z_delta > 0.03
    executed_post_lift = None
    try:
        executed_post_lift = snapshot_panda_hand_object_mesh(franka, scene)
    except Exception as exc:
        cprint(f"   panda_hand@post_lift snapshot failed: {exc}", "yellow")

    failure_stage = None if success else ("lift_result" if planning["lift_plan_success"] else "lift_plan")
    return ExecutionResult(
        success=success,
        failure_stage=failure_stage,
        z_delta_m=z_delta,
        initial_object_position_world=[float(x) for x in np.asarray(obj_init).reshape(3)],
        final_object_position_world=[float(x) for x in np.asarray(obj_after).reshape(3)],
        gripper_tips_loc=gripper_tips_loc,
        finger_width_actual=finger_width_actual,
        executed_at_close=executed_at_close,
        executed_post_lift=executed_post_lift,
        planning=planning,
        metadata={
            "command_name": command.name,
            "command_score": command.score,
            "finger_log_samples": len(force_log),
        },
    )


def _write_snapshot_group(parent, name: str, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    g = parent.create_group(name)
    g.attrs["frame"] = "object_mesh"
    g.attrs["ee_frame"] = "panda_hand"
    for key in ["position", "rotation", "approach_dir", "finger_dir"]:
        if key in snapshot:
            g.create_dataset(key, data=snapshot[key])


def write_robot_gt_hdf5(
    *,
    result_dir: str,
    scene: SceneSpec,
    command: OpenLoopGraspCommand,
    execution: ExecutionResult,
    policy_name: str,
) -> str:
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{scene.episode_id}_robot_gt.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["obj_id"] = scene.obj_id
        f.attrs["episode_id"] = scene.episode_id
        f.attrs["policy_name"] = policy_name
        f.attrs["success"] = bool(execution.success)
        f.attrs["failure_stage"] = execution.failure_stage or ""
        f.attrs["z_delta_m"] = execution.z_delta_m if execution.z_delta_m is not None else np.nan
        f.attrs["object_scale"] = scene.object_scale
        f.attrs["robot_gt_schema_version"] = 2
        f.attrs["scene_schema_version"] = 1
        f.attrs["executed_pose_frame"] = "object_mesh"
        f.attrs["executed_ee_frame"] = "panda_hand"
        f.attrs["sim_z_yaw_deg"] = scene.sim_z_yaw_deg
        if execution.video_path:
            f.attrs["video_path"] = execution.video_path

        cg = f.create_group("candidate_results")
        ci = cg.create_group("candidate_0")
        ci.attrs["name"] = command.name
        ci.attrs["score"] = command.score
        ci.attrs["success"] = bool(execution.success)
        ci.attrs["gripper_width"] = command.gripper_width
        ci.attrs["approach_type"] = command.approach_type
        ci.create_dataset("grasp_point", data=command.position.astype(np.float32))
        ci.create_dataset("rotation", data=command.rotation.astype(np.float32))
        _write_snapshot_group(ci, "executed_panda_hand_at_close", execution.executed_at_close)
        _write_snapshot_group(ci, "executed_panda_hand_post_lift", execution.executed_post_lift)
        if execution.gripper_tips_loc is not None:
            ci.create_dataset("gripper_tips_loc", data=execution.gripper_tips_loc)
            ci.attrs["finger_width_actual"] = float(execution.finger_width_actual or 0.0)

        if execution.success:
            sg = f.create_group("successful_grasps")
            sg.attrs["count"] = 1
            gi = sg.create_group("grasp_0")
            gi.attrs["name"] = command.name
            gi.attrs["score"] = command.score
            gi.attrs["gripper_width"] = command.gripper_width
            gi.attrs["approach_type"] = command.approach_type
            gi.create_dataset("grasp_point", data=command.position.astype(np.float32))
            gi.create_dataset("rotation", data=command.rotation.astype(np.float32))
            _write_snapshot_group(gi, "executed_panda_hand_at_close", execution.executed_at_close)
            _write_snapshot_group(gi, "executed_panda_hand_post_lift", execution.executed_post_lift)
    return path

