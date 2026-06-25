#!/usr/bin/env python3
"""
Baseline2 — Step 1: Sim Trajectory Recorder
============================================
从 robot_gt_merged 中读取成功抓取位姿，在 Isaac Sim 中 Replay，
逐帧记录 state / action / point_cloud，输出 HDF5 供 DP3 训练。

输出格式 (每个 episode 一个 HDF5):
  state:       (T, 8)       - EEF xyz(3) + quat_wxyz(4) + gripper(1)
  action:      (T, 8)       - 同上，t+1 时刻目标 (最后帧重复)
  point_cloud: (T, 4096, 3) - 物体点云 (虚拟相机 depth + 语义分割)

用法:
  # 单物体
  sim45 sim/record_trajectory.py --obj A02018 --headless

  # 批量全部 OakInk
  sim45 sim/record_trajectory.py --all --dataset oakink --headless

  # 指定 GT 目录
  sim45 sim/record_trajectory.py --all --gt_dir output/robot_gt_r3 --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys, glob, json, time
import numpy as np

# ── 先解析参数，再启动 SimulationApp ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--obj",      type=str,  default=None)
parser.add_argument("--all",      action="store_true")
parser.add_argument("--dataset",  type=str,  default="oakink")
parser.add_argument("--gt_dir",   type=str,  default=None,
                    help="robot_gt 目录 (默认 output/robot_gt_merged_{dataset})")
parser.add_argument("--out_dir",  type=str,  default=None,
                    help="HDF5 输出目录 (默认 Baseline2/data/hdf5/{dataset})")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--n_points", type=int,  default=4096)
parser.add_argument("--force",    action="store_true", help="覆盖已有文件")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})
RENDER_SIM = not args.headless

# ── 项目路径 ──────────────────────────────────────────────────────────────
import h5py, torch
from termcolor import cprint
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.prims import get_prim_at_path
import isaacsim.core.utils.numpy.rotations as rot_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as _cfg

PROJ        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_USD_DIR = os.path.join(PROJ, "output", "obj_usd")
ASSETS_DIR  = os.path.join(PROJ, "sim", "assets_scene")
TCP_OFFSET  = 0.105   # Franka 手腕到指尖中心 (m)
N_POINTS    = args.n_points

# ── cuRobo ────────────────────────────────────────────────────────────────
CUROBO_PATHS = [
    "/home/lyh/curobo/src",
    "/home/lyh/Project/curobo/src",
]
for p in CUROBO_PATHS:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType

_CUROBO_MG = None

def init_curobo():
    global _CUROBO_MG
    if _CUROBO_MG is not None:
        return _CUROBO_MG
    cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": 10, "mesh": 4},
        num_trajopt_seeds=4,
    )
    _CUROBO_MG = MotionGen(cfg)
    _CUROBO_MG.warmup(enable_graph=True)
    return _CUROBO_MG


def plan_to(mg, franka, target_pos, target_quat_wxyz, label=""):
    """cuRobo 规划，返回关节轨迹 list 或 None."""
    goal = CuPose(
        torch.tensor([target_pos], dtype=torch.float32).cuda(),
        torch.tensor([target_quat_wxyz[[1,2,3,0]]], dtype=torch.float32).cuda(),  # wxyz→xyzw
    )
    joints = franka.get_joint_positions()[:7]
    start  = CuJointState.from_position(
        torch.tensor(joints, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )
    cfg = MotionGenPlanConfig(max_attempts=4, timeout=10.0, enable_graph=True)
    res = mg.plan_single(start, goal, cfg)
    if res.success.item():
        return [p.cpu().numpy() for p in res.get_interpolated_plan().position]
    cprint(f"   ❌ cuRobo [{label}] 规划失败", "red")
    return None


# ── 场景工具 ──────────────────────────────────────────────────────────────
def find_obj_usd(obj_id, dataset):
    for ds in [dataset, "oakink", "dexycb", "arctic"]:
        p = os.path.join(OBJ_USD_DIR, ds, f"{obj_id}.usd")
        if os.path.exists(p):
            return p
    # fallback sim/assets/
    p = os.path.join(PROJ, "sim", "assets", f"{obj_id}.usd")
    return p if os.path.exists(p) else None


def setup_scene(obj_id, dataset, obj_scale=1.0):
    world = World(stage_units_in_meters=1.0)
    # 地面场景
    env_usd = os.path.join(ASSETS_DIR, "Collected_default_environment", "default_environment.usd")
    add_reference_to_stage(usd_path=env_usd, prim_path="/World/Env")
    # 物体
    obj_usd = find_obj_usd(obj_id, dataset)
    if obj_usd is None:
        raise FileNotFoundError(f"USD not found for {obj_id}")
    from pxr import UsdGeom, Gf
    obj_prim_path = "/World/Object"
    add_reference_to_stage(usd_path=obj_usd, prim_path=obj_prim_path)
    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import Usd
    stage = simulation_app.context.get_stage()
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(obj_prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.03))
    xform.AddScaleOp().Set(Gf.Vec3d(obj_scale, obj_scale, obj_scale))
    # Franka
    franka = Franka(prim_path="/World/Franka", name="franka")
    world.scene.add(franka)
    # 俯视虚拟相机 (物体点云用)
    cam = Camera(
        prim_path="/World/TopCamera",
        position=np.array([0.5, 0.0, 1.2]),
        frequency=30,
        resolution=(640, 480),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.add_semantic_segmentation_to_frame()
    world.reset()
    set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.5, 0.0, 0.03])
    # warmup render
    for _ in range(20):
        world.step(render=RENDER_SIM)
    return world, franka, cam


# ── 点云提取 ──────────────────────────────────────────────────────────────
def get_object_pointcloud(cam, n_points=4096, obj_semantic_label="Object"):
    """从虚拟相机深度图 + 语义分割提取物体点云."""
    frame = cam.get_current_frame()
    depth  = frame.get("distance_to_image_plane")   # (H, W) float32, 单位 m
    seg    = frame.get("semantic_segmentation")       # (H, W, 4) RGBA uint8

    if depth is None or seg is None:
        return np.zeros((n_points, 3), dtype=np.float32)

    # 物体 mask: semantic label 对应 RGB 中非背景部分
    # Isaac Sim 语义分割: 物体像素有特定颜色，背景为 (0,0,0)
    mask = (seg[..., 0] > 10) | (seg[..., 1] > 10) | (seg[..., 2] > 10)  # 非背景

    H, W = depth.shape
    # 相机内参 (虚拟相机默认 FOV=90°)
    fov  = np.radians(90)
    fx   = W / (2 * np.tan(fov / 2))
    fy   = fx
    cx, cy = W / 2.0, H / 2.0

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = mask & (depth > 0.01) & (depth < 3.0)

    z = depth[valid].astype(np.float32)
    x = ((u[valid] - cx) / fx * z).astype(np.float32)
    y = ((v[valid] - cy) / fy * z).astype(np.float32)
    xyz = np.stack([x, y, z], axis=1)  # (N, 3)

    if len(xyz) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # 下采样至 n_points
    if len(xyz) > n_points:
        idx = np.random.choice(len(xyz), n_points, replace=False)
        xyz = xyz[idx]
    elif len(xyz) < n_points:
        pad = np.zeros((n_points - len(xyz), 3), dtype=np.float32)
        xyz = np.concatenate([xyz, pad], axis=0)
    return xyz


# ── EEF state 提取 ──────────────────────────────────────────────────────
def get_eef_state(franka, gripper_cmd):
    """返回 (8,) = xyz(3) + quat_wxyz(4) + gripper(1)."""
    pos  = franka.get_end_effector_pose().p.cpu().numpy().flatten()   # (3,)
    quat = franka.get_end_effector_pose().q.cpu().numpy().flatten()   # (4,) xyzw
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
    grip = np.array([float(gripper_cmd)], dtype=np.float32)           # 0=open,1=close
    return np.concatenate([pos, quat_wxyz, grip]).astype(np.float32)


# ── 单次抓取 Replay ──────────────────────────────────────────────────────
def replay_grasp(world, franka, cam, mg,
                 grasp_pos, grasp_rot, gripper_width, obj_scale):
    """
    执行一次完整抓取并记录轨迹。
    返回 dict(state, action, point_cloud) 或 None (失败)
    """
    from scipy.spatial.transform import Rotation as R

    states = []
    actions = []
    pcs = []

    def record_frame(gripper_cmd):
        s  = get_eef_state(franka, gripper_cmd)
        pc = get_object_pointcloud(cam, N_POINTS)
        states.append(s)
        pcs.append(pc)

    # 物体位姿 (世界系: 桌面 0.5, 0, 0.03)
    T_obj = np.eye(4)
    T_obj[:3, 3] = np.array([0.5, 0.0, 0.03])

    # 计算世界系抓取位置
    pos_w = T_obj[:3, 3] + grasp_pos * obj_scale
    rot_w = grasp_rot  # 3×3
    approach = rot_w[:, 2]
    pos_wrist = pos_w - approach * TCP_OFFSET
    quat_xyzw = R.from_matrix(rot_w).as_quat()
    quat_wxyz  = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)

    # 预抓取点 (沿 approach 方向退后 12cm)
    pre_pos = pos_wrist - approach * 0.12

    # Phase 1: Home → 预抓取
    traj = plan_to(mg, franka, pre_pos, quat_wxyz, "pre-grasp")
    if traj is None:
        return None
    franka.open_gripper()
    for jp in traj:
        grip = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([jp, grip]))
        world.step(render=RENDER_SIM)
        record_frame(gripper_cmd=0.0)  # 开

    # Phase 2: 预抓取 → 抓取点
    traj2 = plan_to(mg, franka, pos_wrist, quat_wxyz, "grasp")
    if traj2 is None:
        return None
    for jp in traj2:
        grip = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([jp, grip]))
        world.step(render=RENDER_SIM)
        record_frame(gripper_cmd=0.0)

    # Phase 3: 闭合夹爪
    target_w = gripper_width / 2.0
    for _ in range(30):
        cur = franka.get_joint_positions()[7:9]
        new_w = np.clip(cur - 0.005, target_w, 0.04)
        franka.set_joint_positions(
            np.concatenate([franka.get_joint_positions()[:7], new_w, new_w]))
        world.step(render=RENDER_SIM)
        record_frame(gripper_cmd=1.0)  # 闭

    # Phase 4: 抬起 15cm
    lift_pos = pos_wrist.copy()
    lift_pos[2] += 0.15
    traj3 = plan_to(mg, franka, lift_pos, quat_wxyz, "lift")
    if traj3 is not None:
        for jp in traj3:
            grip = franka.get_joint_positions()[7:9]
            franka.set_joint_positions(np.concatenate([jp, grip]))
            world.step(render=RENDER_SIM)
            record_frame(gripper_cmd=1.0)

    # 检查物体是否被抬起
    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom
    stage = simulation_app.context.get_stage()
    prim  = stage.GetPrimAtPath("/World/Object")
    xf    = UsdGeom.Xformable(prim)
    obj_z = float(xf.ComputeLocalToWorldTransform(0).GetRow3(3)[2])
    if obj_z < 0.08:
        return None  # 未抬起

    # 构建 state/action 序列
    T = len(states)
    state_arr  = np.stack(states, axis=0).astype(np.float32)   # (T, 8)
    action_arr = np.concatenate(
        [state_arr[1:], state_arr[-1:]], axis=0).astype(np.float32)  # (T, 8) shifted
    pc_arr     = np.stack(pcs, axis=0).astype(np.float32)      # (T, 4096, 3)

    return {"state": state_arr, "action": action_arr, "point_cloud": pc_arr}


# ── 主流程 ───────────────────────────────────────────────────────────────
def load_successful_grasps(gt_dir, obj_id):
    """从 robot_gt HDF5 加载成功抓取位姿列表."""
    path = os.path.join(gt_dir, f"{obj_id}_robot_gt.hdf5")
    if not os.path.exists(path):
        return []
    grasps = []
    with h5py.File(path, "r") as f:
        grp = f.get("successful_grasps", {})
        for key in grp.keys():
            g = grp[key]
            grasps.append({
                "position":      g["grasp_point"][:].astype(np.float32),
                "rotation":      g["rotation"][:].astype(np.float32),
                "gripper_width": float(g.attrs.get("gripper_width", 0.04)),
            })
    return grasps


def get_obj_scale(obj_id, dataset):
    import json
    scale_path = os.path.join(
        PROJ, "data_hub", "ProcessedData", "obj_meshes", dataset, obj_id, "scale.json")
    if os.path.exists(scale_path):
        with open(scale_path) as f:
            return float(json.load(f)["scale_factor"])
    return 1.0


def process_object(obj_id, dataset, gt_dir, out_dir, mg):
    scale  = get_obj_scale(obj_id, dataset)
    grasps = load_successful_grasps(gt_dir, obj_id)
    if not grasps:
        cprint(f"  ⚠️  {obj_id}: 无成功抓取数据", "yellow")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    episodes_saved = 0

    try:
        world, franka, cam = setup_scene(obj_id, dataset, obj_scale=scale)
    except FileNotFoundError as e:
        cprint(f"  ❌ {obj_id}: {e}", "red")
        return 0

    for ep_idx, grasp in enumerate(grasps):
        out_path = os.path.join(out_dir, f"{ep_idx:04d}.hdf5")
        if os.path.exists(out_path) and not args.force:
            episodes_saved += 1
            continue

        # 重置场景
        world.reset()
        franka.initialize()
        for _ in range(10):
            world.step(render=RENDER_SIM)

        result = replay_grasp(
            world, franka, cam, mg,
            grasp["position"], grasp["rotation"],
            grasp["gripper_width"], scale)

        if result is None:
            cprint(f"    [{ep_idx}] ❌ Replay 失败", "red")
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("state",       data=result["state"])
            f.create_dataset("action",      data=result["action"])
            f.create_dataset("point_cloud", data=result["point_cloud"])
            f.attrs["obj_id"]   = obj_id
            f.attrs["episode"]  = ep_idx
            f.attrs["n_frames"] = len(result["state"])
        episodes_saved += 1
        cprint(f"    [{ep_idx}] ✅ {len(result['state'])} frames → {out_path}", "green")

    simulation_app.close()
    return episodes_saved


def main():
    DATASETS = ["oakink", "dexycb"]
    dataset  = args.dataset

    gt_dir  = args.gt_dir or os.path.join(
        PROJ, "output", f"robot_gt_merged_{dataset}")
    out_base = args.out_dir or os.path.join(
        PROJ, "Baseline2", "data", "hdf5", dataset)

    # 构建物体列表
    if args.obj:
        obj_list = [args.obj]
    elif args.all:
        gt_files = glob.glob(os.path.join(gt_dir, "*_robot_gt.hdf5"))
        obj_list = sorted(
            os.path.basename(f).replace("_robot_gt.hdf5", "") for f in gt_files)
    else:
        print("用法: sim45 sim/record_trajectory.py --obj A02018 --headless")
        print("      sim45 sim/record_trajectory.py --all --dataset oakink --headless")
        return

    cprint("=" * 60, "cyan")
    cprint(f"  Baseline2 Trajectory Recorder", "cyan")
    cprint(f"  Dataset:  {dataset}", "cyan")
    cprint(f"  GT dir:   {gt_dir}", "cyan")
    cprint(f"  Output:   {out_base}", "cyan")
    cprint(f"  Objects:  {len(obj_list)}", "cyan")
    cprint("=" * 60, "cyan")

    mg = init_curobo()
    total = 0
    for i, obj_id in enumerate(obj_list):
        cprint(f"\n[{i+1}/{len(obj_list)}] {obj_id}", "white")
        out_dir = os.path.join(out_base, obj_id)
        n = process_object(obj_id, dataset, gt_dir, out_dir, mg)
        total += n

    cprint(f"\n{'='*60}", "cyan")
    cprint(f"  完成! 共记录 {total} 个 episode", "cyan")
    cprint(f"  输出: {out_base}", "cyan")
    cprint(f"{'='*60}", "cyan")


if __name__ == "__main__":
    main()
