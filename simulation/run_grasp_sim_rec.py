#!/usr/bin/env python3
"""
Pipeline Stage B + per-try viewport 录屏 (run_grasp_sim 同款逻辑)
================================================================
与 run_grasp_sim.py 行为一致，额外在每个 candidate try 录制 mp4。

运行 (服务器 headed，建议 Xvfb):
    export DISPLAY=:99
    Xvfb :99 -screen 0 1920x1080x24 &
    $ISAAC_SIM_PATH/python.sh sim/run_grasp_sim_rec.py \\
        --hdf5 output/grasp_collect_no_rot/candidates/smoke_ycb_legacy/ycb_dex_01.hdf5 \\
        --record-video output/grasp_sim_videos/smoke_ycb \\
        --max-candidates 5

    # 跳过 15cm pre-grasp，只 direct + final approach:
    $ISAAC_SIM_PATH/python.sh sim/run_grasp_sim_rec.py \\
        --hdf5 ... --record-video ... --skip-pre-grasp

Pipeline:
    HDF5 → 搭建场景 → 坐标变换 → cuRobo 规划 → Franka 执行抓取 → 每 try 一个 mp4
"""
from isaacsim import SimulationApp
import argparse
import os
import sys
import json
import glob
import shutil
import subprocess

# Parse args before SimulationApp (因为 SimulationApp 修改了 sys.argv)
parser = argparse.ArgumentParser(
    description="Pipeline Stage B: Isaac Sim Grasp Execution + per-try video",
)
parser.add_argument("--hdf5", type=str, required=True, help="Stage A 输出的 HDF5 路径")
parser.add_argument("--headless", action="store_true", help="无头模式 (录屏时建议 headed + Xvfb)")
parser.add_argument("--object_scale", type=float, default=1.0, help="物体缩放 (默认 1.0)")
parser.add_argument("--save-result", action="store_true", default=True,
                    help="保存 Robot GT 结果到 HDF5 (默认开启)")
parser.add_argument("--result-dir", type=str, default=None,
                    help="结果保存目录 (默认 sim/output/robot_gt/)")
parser.add_argument("--max-candidates", type=int, default=None,
                    help="最多尝试的候选数 (默认全部)")
parser.add_argument("--record-video", type=str, required=True,
                    help="录屏输出根目录，如 output/grasp_sim_videos/smoke_ycb")
parser.add_argument("--record-every", type=int, default=3,
                    help="每 N 次 world.step 抓 1 帧 viewport")
parser.add_argument("--record-fps", type=int, default=30, help="合成 mp4 帧率")
parser.add_argument("--record-keep-frames", action="store_true",
                    help="保留 PNG 帧目录 (默认 encode 后删除)")
parser.add_argument("--record-include-reset", action="store_true",
                    help="try 结束后把场景重置也录进同一段 (默认不含)")
parser.add_argument(
    "--skip-pre-grasp",
    action="store_true",
    help="跳过 15cm pre-grasp，直接 cuRobo 规划到抓取点 (object mesh)，再 final approach",
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})


# render=True 在非 headless 模式显示流畅运动；录屏时 step 强制 render
RENDER_SIM = not args.headless

import numpy as np
import h5py
import torch
from termcolor import cprint
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)

# ── cuRobo 路径: 按机器自动查找 ──
for _curobo_candidate in [
    os.path.expanduser('~/Project/curobo/src'),          # 本机
    os.path.expanduser('~/curobo/src'),                  # 服务器备用
    '/home/vision/Project/curobo/src',                   # 服务器绝对路径
]:
    if os.path.isdir(os.path.join(_curobo_candidate, 'curobo')):
        sys.path.insert(0, _curobo_candidate)
        break
from env_config.robot.Franka import Franka
from env_config.room.Real_Ground import Real_Ground
from env_config.rigid.RigidObject import RigidObject
from curobo_world import prepare_curobo_mesh, build_world_config_dict, sync_curobo_world, object_pose_robot_frame


# ============================================================
# Scene Config (和 Replay_Grasp / Scene_Env 一致)
# ============================================================
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80

OBJECT_POSITION = [0.0, 0.55, TABLE_TOP_Z]
OBJECT_ORIENTATION = [0.0, 0.0, 0.0]

APPROACH_HEIGHT = 0.15  # 物体上方 15cm
LIFT_HEIGHT = 0.15      # 提起 15cm (Franka workspace 限制)

# 每物体旋转覆盖 (object_rotation_overrides.json)
_OVERRIDE_FILE = os.path.join(os.path.dirname(__file__), 'object_rotation_overrides.json')
try:
    import json as _json
    with open(_OVERRIDE_FILE) as _f:
        OBJECT_ROTATION_OVERRIDES = _json.load(_f)
    OBJECT_ROTATION_OVERRIDES = {k: v for k, v in OBJECT_ROTATION_OVERRIDES.items() if not k.startswith('_')}
except Exception:
    OBJECT_ROTATION_OVERRIDES = {}


# ============================================================
# Per-try viewport recording (route 1: PNG → ffmpeg mp4)
# ============================================================
class TryVideoRecorder:
    """每个 candidate try：start → capture on world.step → stop → mp4."""

    def __init__(self, root_dir, record_every, fps, keep_frames):
        self.root_dir = os.path.abspath(root_dir)
        self.record_every = max(1, int(record_every))
        self.fps = max(1, int(fps))
        self.keep_frames = keep_frames
        self._active = False
        self._frames_dir = None
        self._frame_idx = 0
        self._step_counter = 0
        self._viewport = None
        self._vu = None
        self._orig_world_step = None
        self._manifest = []
        os.makedirs(self.root_dir, exist_ok=True)

    def _ensure_viewport(self):
        if self._viewport is not None:
            return
        import omni.kit.viewport.utility as vu
        self._vu = vu
        self._viewport = vu.get_active_viewport()
        if self._viewport is None:
            raise RuntimeError("No active viewport — use headed mode (DISPLAY + Xvfb) for recording")

    def attach_world(self, world):
        self._orig_world_step = world.step

        def step_with_capture(render=True):
            use_render = render or self._active
            self._orig_world_step(render=use_render)
            if self._active:
                self._maybe_capture()

        world.step = step_with_capture

    def _maybe_capture(self):
        self._step_counter += 1
        if self._step_counter % self.record_every != 0:
            return
        self._ensure_viewport()
        path = os.path.join(self._frames_dir, f"f_{self._frame_idx:05d}.png")
        self._vu.capture_viewport_to_file(self._viewport, path)
        self._frame_idx += 1

    def start(self, obj_id, attempt_idx, cand_name):
        self._ensure_viewport()
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in cand_name)
        base = f"{attempt_idx + 1:02d}_{slug}"
        self._frames_dir = os.path.join(self.root_dir, obj_id, base, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        for p in glob.glob(os.path.join(self._frames_dir, "*.png")):
            os.remove(p)
        self._frame_idx = 0
        self._step_counter = 0
        self._active = True
        self._current_base = base
        self._current_mp4 = os.path.join(self.root_dir, obj_id, f"{base}.mp4")
        cprint(f"  📹 REC start → {self._current_mp4}", "magenta")
        for _ in range(3):
            self._maybe_capture()

    def stop(self, success, score, approach_type):
        if not self._active:
            return None
        self._active = False
        n_frames = self._frame_idx
        mp4_path = self._current_mp4
        if n_frames > 0:
            self._encode_mp4()
        else:
            cprint(f"  📹 REC skip (0 frames): {mp4_path}", "yellow")
        if not self.keep_frames and self._frames_dir and os.path.isdir(self._frames_dir):
            shutil.rmtree(os.path.dirname(self._frames_dir), ignore_errors=True)
        entry = {
            "video": mp4_path,
            "name": self._current_base,
            "n_frames": n_frames,
            "success": bool(success),
            "score": float(score),
            "approach_type": str(approach_type),
        }
        self._manifest.append(entry)
        cprint(f"  📹 REC done  {n_frames} frames → {mp4_path}", "magenta")
        return mp4_path

    def _encode_mp4(self):
        pattern = os.path.join(self._frames_dir, "f_%05d.png")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(self.fps),
            "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            self._current_mp4,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            cprint("  ❌ ffmpeg not found; PNG frames kept in " + self._frames_dir, "red")
        except subprocess.CalledProcessError as e:
            cprint(f"  ❌ ffmpeg failed: {e}; frames in {self._frames_dir}", "red")

    def write_manifest(self, obj_id, hdf5_path):
        manifest_path = os.path.join(self.root_dir, obj_id, "manifest.jsonl")
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for row in self._manifest:
                row = dict(row)
                row["obj_id"] = obj_id
                row["hdf5"] = os.path.abspath(hdf5_path)
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
        cprint(f"  📹 manifest → {manifest_path}", "magenta")


# ============================================================
# Scene Setup
# ============================================================
def setup_scene(obj_id, object_scale):
    """搭建仿真场景."""
    world = World(backend="numpy")

    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    set_camera_view(
        eye=[0.0, 4.5, 3.5], target=[0.0, 0.0, 0.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")

    # ── 蓝色格子地板: 加载 default_environment.usd (视觉) ────────────────────
    _sim_dir = os.path.dirname(os.path.abspath(__file__))
    _env_usd = os.path.join(_sim_dir, "assets_scene",
                            "Collected_default_environment", "default_environment.usd")
    delete_prim("/World/Environment")
    if os.path.exists(_env_usd):
        from isaacsim.core.utils.stage import add_reference_to_stage as _add_ref
        _add_ref(usd_path=_env_usd, prim_path="/World/Environment")

    # ── 物理地板: FixedCuboid (Isaac Sim 5.0 fix, async 崩溃用同步方案) ──────
    delete_prim("/World/Ground")
    FixedCuboid(
        prim_path="/World/Ground", name="ground",
        position=np.array([0.0, 0.0, -0.025]),
        scale=np.array([20.0, 20.0, 0.05]),
        size=1.0, visible=False,
    )

    delete_prim("/World/Table")
    FixedCuboid(
        prim_path="/World/Table", name="table",
        position=TABLE_POSITION,
        orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
        scale=TABLE_SCALE, size=1.0, visible=True,
    )


    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))

    world.reset()



    for _ in range(50):
        world.step(render=RENDER_SIM)

    franka.open_gripper()
    for _ in range(10):
        world.step(render=RENDER_SIM)

    # 加载物体 USD — 搜索顺序: output/obj_usd/{dataset}/ → sim/assets/
    a2g_root = os.path.dirname(SIM_DIR)
    obj_usd_root = os.path.join(a2g_root, 'output', 'obj_usd')
    datasets_order = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']
    usd_search_paths = (
        [os.path.join(obj_usd_root, ds, f'{obj_id}.usd') for ds in datasets_order]
        + [os.path.join(SIM_DIR, 'assets', f'{obj_id}.usd')]
    )
    usd_path = next((p for p in usd_search_paths if os.path.exists(p)), None)

    if usd_path is None:
        cprint(f"❌ USD not found: {obj_id}.usd", "red")
        cprint(f"   搜索路径: output/obj_usd/{{dataset}}/  sim/assets/", "yellow")
        cprint(f"   先运行: python3 tools/convert_obj_usd.py --obj {obj_id}", "yellow")
        return None

    # 读取每物体覆盖
    _override = OBJECT_ROTATION_OVERRIDES.get(obj_id, None)
    obj_orientation = list(OBJECT_ORIENTATION)

    # ── z_offset 优先级: meta JSON > override config > 启发式 ────────────────
    # meta JSON 由 convert_obj_usd.py 生成（精确值：使物体底面落在 Z=0）
    _meta_path = usd_path.replace('.usd', '_meta.json')
    if os.path.exists(_meta_path):
        with open(_meta_path) as _mf:
            _meta = json.load(_mf)
        obj_z_offset = float(_meta.get('z_offset_m', 0.075 * object_scale))
        cprint(f"   z_offset (meta): {obj_id} → {obj_z_offset*100:.1f}cm", "cyan")
    elif isinstance(_override, dict) and 'z_offset' in _override:
        obj_z_offset = _override['z_offset']
        cprint(f"   z_offset (override): {obj_id} → {obj_z_offset*100:.1f}cm", "cyan")
    else:
        obj_z_offset = 0.075 * object_scale   # 7.5cm 启发式

    if isinstance(_override, dict) and 'rotation' in _override:
        obj_orientation = _override['rotation']
        cprint(f"   ⮻ 朝向覆盖: {obj_id} → {obj_orientation}°", "cyan")

    obj_pos = list(OBJECT_POSITION)
    obj_pos[2] += obj_z_offset

    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")

    obj = RigidObject(
        world, usd_path=usd_path,
        pos=np.array(obj_pos), ori=np.array(obj_orientation),
        scale=np.array([object_scale] * 3), mass=0.05,
    )

    # 碰撞 + 摩擦力设置
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, UsdShade
    stage = world.stage

    # 1. 创建高摩擦材料
    material_path = "/World/PhysicsMaterials/BottleMaterial"
    UsdShade.Material.Define(stage, material_path)
    mat_prim = stage.GetPrimAtPath(material_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.CreateStaticFrictionAttr(1.0)    # 高静摩擦
    physics_mat.CreateDynamicFrictionAttr(0.8)   # 高动摩擦
    physics_mat.CreateRestitutionAttr(0.0)       # 不弹跳
    cprint(f"   ✅ Physics material: friction=1.0/0.8", "green")

    # 2. 物体碰撞 + 绑定摩擦材料
    obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
    for prim in Usd.PrimRange(obj_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.GetApproximationAttr().Set("convexHull")
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col_api.GetContactOffsetAttr().Set(0.02)
            col_api.GetRestOffsetAttr().Set(0.001)
            # 绑定摩擦材料
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                UsdShade.Material(mat_prim),
                UsdShade.Tokens.weakerThanDescendants,
                "physics"
            )

    # 3. 给夹爪指尖也加摩擦
    finger_material_path = "/World/PhysicsMaterials/FingerMaterial"
    UsdShade.Material.Define(stage, finger_material_path)
    finger_mat_prim = stage.GetPrimAtPath(finger_material_path)
    finger_physics_mat = UsdPhysics.MaterialAPI.Apply(finger_mat_prim)
    finger_physics_mat.CreateStaticFrictionAttr(1.2)   # 更高摩擦 (橡胶指尖)
    finger_physics_mat.CreateDynamicFrictionAttr(1.0)
    finger_physics_mat.CreateRestitutionAttr(0.0)

    for finger_name in ["panda_leftfinger", "panda_rightfinger"]:
        finger_path = f"/World/Franka/{finger_name}"
        finger_prim = stage.GetPrimAtPath(finger_path)
        if finger_prim.IsValid():
            for child in Usd.PrimRange(finger_prim):
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
                    binding = UsdShade.MaterialBindingAPI.Apply(child)
                    binding.Bind(
                        UsdShade.Material(finger_mat_prim),
                        UsdShade.Tokens.weakerThanDescendants,
                        "physics"
                    )
            cprint(f"   ✅ Finger friction on {finger_name}", "green")

    for _ in range(100):
        world.step(render=RENDER_SIM)

    # ── cuRobo 规划用物体 mesh（规划层碰撞，不影响 PhysX）────────────────────
    scene_out = {"world": world, "franka": franka, "obj": obj}
    mesh_info = prepare_curobo_mesh(stage, obj.rigid_prim_path)
    if mesh_info is not None:
        scene_out["curobo_mesh_vertices"] = mesh_info["vertices"]
        scene_out["curobo_mesh_faces"]    = mesh_info["faces"]
        cprint(
            f"   ✅ cuRobo object mesh: {mesh_info['n_faces']} faces"
            f" (raw {mesh_info['n_faces_raw']})", "green"
        )
    else:
        cprint("   ⚠️ cuRobo object mesh: 提取失败，仅用桌面+地面", "yellow")

    cprint("✅ Scene Ready", "green")
    return scene_out


# ============================================================
# Coordinate Transforms
# ============================================================
def get_robot_base_transform():
    """T_world_robot 和 T_robot_world."""
    yaw_rad = np.deg2rad(ROBOT_ORIENTATION[2])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = ROBOT_POSITION
    return T, np.linalg.inv(T)


def world_to_robot_pose(pos_w, quat_wxyz_w, T_robot_world):
    """世界坐标 → 机器人底座坐标."""
    pos_r = (T_robot_world @ np.append(pos_w, 1.0))[:3]

    R_w = Rotation.from_quat([quat_wxyz_w[1], quat_wxyz_w[2],
                               quat_wxyz_w[3], quat_wxyz_w[0]])
    R_rw = Rotation.from_matrix(T_robot_world[:3, :3])
    R_r = R_rw * R_w
    q = R_r.as_quat()  # xyzw
    quat_r = np.array([q[3], q[0], q[1], q[2]])  # wxyz

    return pos_r, quat_r


def transform_grasp_to_world(grasp_pos_obj, grasp_rot_obj, T_world_obj):
    """OBJ 坐标 → 世界坐标."""
    pos_w = (T_world_obj @ np.append(grasp_pos_obj, 1.0))[:3]
    rot_w = T_world_obj[:3, :3] @ grasp_rot_obj
    return pos_w, rot_w


def make_transform(pos, quat_wxyz):
    """位置+四元数 → 4x4 变换矩阵."""
    T = np.eye(4)
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    return T


def world_pose_to_object_mesh(pos_world, quat_wxyz_world, obj_pos_world, obj_quat_wxyz,
                              object_scale):
    """世界系位姿 → 物体 mesh 系 (与候选 grasp_point 同一 scale 约定)."""
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
    pos_obj = (T_obj_body[:3, 3] / scale).astype(np.float32)
    rot_obj = T_obj_body[:3, :3].astype(np.float32)
    return pos_obj, rot_obj


def snapshot_panda_hand_object_mesh(franka, scene, object_scale):
    """当前 panda_hand 位姿，物体 mesh 局部系 (实时物体 pose)."""
    pos_w, quat_w = franka.get_cur_ee_pos(local_frame=False)
    obj_pos, obj_quat = scene["obj"].get_obj_pos()
    pos_o, rot_o = world_pose_to_object_mesh(
        pos_w, quat_w, obj_pos, obj_quat, object_scale,
    )
    return {
        'position': pos_o,
        'rotation': rot_o,
        'approach_dir': rot_o[:, 2].copy(),
        'finger_dir': rot_o[:, 1].copy(),
    }


def world_point_to_object_mesh(point_world, obj_pos_world, obj_quat_wxyz, object_scale):
    """世界系点 → 物体 mesh 局部系 (与 grasp_point / executed_panda_hand 同一 scale 约定)."""
    T_world_obj = make_transform(
        np.asarray(obj_pos_world, dtype=np.float64).reshape(3),
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4),
    )
    p_h = np.append(np.asarray(point_world, dtype=np.float64).reshape(3), 1.0)
    p_obj = (np.linalg.inv(T_world_obj) @ p_h)[:3]
    scale = float(object_scale) if object_scale else 1.0
    return (p_obj / scale).astype(np.float32)


def snapshot_gripper_tips_object_mesh(stage, scene, object_scale):
    """
    与 executed_panda_hand_at_close 同时刻：左右指尖在物体 mesh 局部系。
    finger 连杆原点作为指尖代理 (panda_leftfinger / panda_rightfinger)。
    """
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
    obj_pos, obj_quat = scene["obj"].get_obj_pos()
    left_o = world_point_to_object_mesh(left_world, obj_pos, obj_quat, object_scale)
    right_o = world_point_to_object_mesh(right_world, obj_pos, obj_quat, object_scale)
    tips = np.stack([left_o, right_o]).astype(np.float32)
    width = float(np.linalg.norm(tips[0] - tips[1]))
    return {'gripper_tips_loc': tips, 'finger_width_actual': width}


def write_gripper_tips_loc_hdf5(parent, gripper_tips_loc, finger_width_actual):
    """gripper_tips_loc: (2,3) 左/右指尖，物体 mesh 系，at_close 时刻。"""
    if gripper_tips_loc is None:
        parent.attrs['has_gripper_tips'] = False
        return
    parent.create_dataset('gripper_tips_loc', data=gripper_tips_loc)
    parent.attrs['finger_width_actual'] = float(finger_width_actual)
    parent.attrs['has_gripper_tips'] = True
    parent.attrs['gripper_tips_frame'] = 'object_mesh'
    parent.attrs['gripper_tips_snapshot'] = 'at_close'


def write_executed_panda_hand_hdf5(parent, snapshot, label):
    """label: 'at_close' | 'post_lift'."""
    if snapshot is None:
        return
    g = parent.create_group(f'executed_panda_hand_{label}')
    g.attrs['frame'] = 'object_mesh'
    g.attrs['ee_frame'] = 'panda_hand'
    g.attrs['snapshot'] = label
    g.create_dataset('position', data=snapshot['position'])
    g.create_dataset('rotation', data=snapshot['rotation'])
    g.create_dataset('approach_dir', data=snapshot['approach_dir'])
    g.create_dataset('finger_dir', data=snapshot['finger_dir'])


def _grasp_result_base(success=False):
    return {
        'success': success,
        'gripper_tips_loc': None,
        'finger_width_actual': None,
        'executed_at_close': None,
        'executed_post_lift': None,
    }


# ============================================================
# cuRobo Motion Planning
# ============================================================
_CUROBO_MG = None

def init_curobo(scene=None):
    """初始化 cuRobo MotionGen (含物体 mesh 障碍物)."""
    import os as _os
    # ── 确保 ninja + nvcc 在 PATH，Isaac Sim python.sh 会覆盖环境 ──
    _extra = '/home/vision/isaacsim/kit/python/bin:/usr/local/cuda/bin'
    if _extra not in _os.environ.get('PATH', ''):
        _os.environ['PATH'] = _extra + ':' + _os.environ.get('PATH', '')

    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    _, T_robot_world = get_robot_base_transform()

    # 桌面障碍物 (变换到机器人底座坐标)
    table_pos_r  = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    ground_pos_r = (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3]

    # 记录到 scene 供 plan_trajectory 复用
    if scene is not None:
        scene["curobo_table_pos_r"]  = table_pos_r
        scene["curobo_ground_pos_r"] = ground_pos_r

    # 物体 mesh 障碍物 (可选)
    mesh_verts = mesh_faces = mesh_pose = None
    if scene is not None and scene.get("curobo_mesh_vertices") is not None:
        mesh_verts  = scene["curobo_mesh_vertices"]
        mesh_faces  = scene["curobo_mesh_faces"]
        pos_w, quat_wxyz = scene["obj"].get_obj_pos()
        mesh_pose = object_pose_robot_frame(pos_w, quat_wxyz, T_robot_world)

    world_config = build_world_config_dict(
        table_pos_r, ground_pos_r, TABLE_SCALE,
        mesh_vertices=mesh_verts, mesh_faces=mesh_faces, mesh_pose_robot=mesh_pose,
    )

    load_kwargs = {"interpolation_dt": 0.02}
    if mesh_verts is not None:
        load_kwargs["collision_cache"] = {"obb": 4, "mesh": 4}

    mg_config = MotionGenConfig.load_from_robot_config(
        "franka.yml", world_config, **load_kwargs,
    )
    mg = MotionGen(mg_config)

    cprint("   → cuRobo warmup...", "yellow")
    mg.warmup()
    if mesh_verts is not None:
        cprint("   ✅ cuRobo ready (table + ground + object mesh)", "green")
    else:
        cprint("   ✅ cuRobo ready (table + ground only)", "green")
    return mg


def plan_trajectory(motion_gen, franka, target_pos_world, target_quat_wxyz_world,
                    label="", scene=None, use_object_mesh=True):
    """cuRobo 规划无碰撞轨迹.

    use_object_mesh=True  → pre-grasp/approach 阶段：cuRobo 包含物体 mesh 障碍
    use_object_mesh=False → final approach/lift  阶段：仅桌面+地面，不阻挡夹爪接触
    """
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    _, T_robot_world = get_robot_base_transform()

    # 每次规划前同步 cuRobo 碰撞世界
    if scene is not None:
        table_pos_r  = scene.get("curobo_table_pos_r",
                                 (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3])
        ground_pos_r = scene.get("curobo_ground_pos_r",
                                 (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3])
        include_mesh = use_object_mesh and scene.get("curobo_mesh_vertices") is not None
        sync_curobo_world(
            motion_gen, scene,
            table_pos_r, ground_pos_r,
            TABLE_SCALE, T_robot_world,
            include_object_mesh=include_mesh,
        )
        if not use_object_mesh and scene.get("curobo_mesh_vertices") is not None:
            cprint(f"      [{label}] cuRobo world: table+ground only (object mesh cleared)", "cyan")
    pos_r, quat_r = world_to_robot_pose(target_pos_world, target_quat_wxyz_world, T_robot_world)

    euler_r = Rotation.from_quat([quat_r[1], quat_r[2], quat_r[3], quat_r[0]]).as_euler('xyz', degrees=True)

    cprint(f"      [{label}] target (world): pos=[{target_pos_world[0]:.4f}, {target_pos_world[1]:.4f}, {target_pos_world[2]:.4f}]", "magenta")
    cprint(f"      [{label}] target (robot): pos=[{pos_r[0]:.4f}, {pos_r[1]:.4f}, {pos_r[2]:.4f}]", "magenta")
    cprint(f"      [{label}] target (robot): euler=[{euler_r[0]:.1f}°, {euler_r[1]:.1f}°, {euler_r[2]:.1f}°]", "magenta")
    cprint(f"      [{label}] target (robot): quat_wxyz=[{quat_r[0]:.4f}, {quat_r[1]:.4f}, {quat_r[2]:.4f}, {quat_r[3]:.4f}]", "magenta")

    current_joints = franka.get_joint_positions()[:7]
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]

    start_state = CuJointState.from_position(
        torch.tensor(current_joints, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=joint_names,
    )

    goal_pose = Pose.from_list([
        float(pos_r[0]), float(pos_r[1]), float(pos_r[2]),
        float(quat_r[0]), float(quat_r[1]), float(quat_r[2]), float(quat_r[3]),
    ])

    plan_config = MotionGenPlanConfig(
        max_attempts=10, enable_graph=True, enable_opt=True,
    )

    result = motion_gen.plan_single(start_state, goal_pose, plan_config)

    _suc = result.success
    if callable(getattr(_suc, 'item', None)):
        _suc = _suc.item()
    if _suc:
        traj = result.get_interpolated_plan()
        cprint(f"      [{label}] ✅ Plan OK: {traj.position.shape[0]} steps", "green")
        return traj.position.cpu().numpy()

    # ---- 详细失败诊断 ----
    cprint(f"      [{label}] ❌ Plan FAILED. Diagnostics:", "red")

    # 1. 检查 IK 是否可达
    try:
        ik_result = motion_gen.ik_solver.solve_single(goal_pose)
        if ik_result.success.item():
            cprint(f"      [{label}]   IK: ✅ 可达 (位置误差={ik_result.position_error[0].item()*1000:.2f}mm)", "yellow")
        else:
            cprint(f"      [{label}]   IK: ❌ 不可达! 这个位姿在机械臂工作空间外", "red")
            cprint(f"      [{label}]   IK 位置误差: {ik_result.position_error[0].item()*1000:.2f}mm", "red")
    except Exception as e:
        cprint(f"      [{label}]   IK check error: {e}", "red")

    # 2. 检查 result 的其他属性
    try:
        if hasattr(result, 'valid_query') and result.valid_query is not None:
            cprint(f"      [{label}]   valid_query: {result.valid_query.item()}", "yellow")
        if hasattr(result, 'status') and result.status is not None:
            cprint(f"      [{label}]   status: {result.status}", "yellow")
        if hasattr(result, 'attempts') and result.attempts is not None:
            cprint(f"      [{label}]   attempts: {result.attempts}", "yellow")
        if hasattr(result, 'position_error') and result.position_error is not None:
            cprint(f"      [{label}]   position_error: {result.position_error.item()*1000:.2f}mm", "yellow")
        if hasattr(result, 'rotation_error') and result.rotation_error is not None:
            cprint(f"      [{label}]   rotation_error: {result.rotation_error.item():.4f}rad", "yellow")
        if hasattr(result, 'cspace_error') and result.cspace_error is not None:
            cprint(f"      [{label}]   cspace_error: {result.cspace_error.item():.6f}", "yellow")
    except Exception as e:
        cprint(f"      [{label}]   (error reading result attrs: {e})", "yellow")

    # 3. 打印桌面位置供参考
    table_pos_r = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    cprint(f"      [{label}]   Table in robot frame: z={table_pos_r[2]:.4f}", "yellow")
    cprint(f"      [{label}]   Target z in robot frame: {pos_r[2]:.4f}", "yellow")
    cprint(f"      [{label}]   距离桌面: {(pos_r[2] - table_pos_r[2])*100:.1f}cm", "yellow")

    return None


def execute_trajectory(franka, world, traj):
    """执行关节轨迹."""
    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=RENDER_SIM)


# ============================================================
# Main Grasp Execution
# ============================================================
def execute_grasp(scene, grasp_pos_obj, grasp_rot_obj, gripper_width, object_scale,
                  is_manual=False, mesh_prerotation_euler=None, skip_pre_grasp=False):
    """完整抓取流程.
    is_manual: 手动标注时 position=指尖中心 → sim 里减 TCP偏移
               自动生成时 position=panda_hand 已含偏移 → 不再减 (防双重)
    mesh_prerotation_euler: 生成抓取时预旋转角 (degrees)。除去 T_world_obj 中的对应旋转
    """
    global _CUROBO_MG

    franka = scene["franka"]
    world = scene["world"]

    # ---- 获取物体世界位姿 ----
    obj_pos_world, obj_quat_wxyz = scene["obj"].get_obj_pos()
    T_world_obj = make_transform(obj_pos_world, obj_quat_wxyz)

    # scale 影响: OBJ 坐标中的位置需要乘以 scale
    grasp_pos_scaled = grasp_pos_obj * object_scale

    # 如果 grasp 在预旋转 mesh 上生成, 除去 T_world_obj 中的预旋转防双重旋转
    if mesh_prerotation_euler and any(abs(e) > 0.5 for e in mesh_prerotation_euler):
        _Rp = Rotation.from_euler('xyz', mesh_prerotation_euler, degrees=True).as_matrix()
        T_eff = T_world_obj.copy()
        T_eff[:3, :3] = T_world_obj[:3, :3] @ _Rp.T
    else:
        T_eff = T_world_obj

    pos_world, rot_world = transform_grasp_to_world(grasp_pos_scaled, grasp_rot_obj, T_eff)

    # ⭐ 关键: 坐标系约定转换
    # 我们的抓取坐标系:     x = 夹爪开合,  y = 沿瓶身(竖直),  z = 接近
    # Franka panda_hand:     y = 夹爪开合,                      z = 接近
    # 需要绕 Z 轴(接近方向)旋转 -90°, 让我们的 x → panda_hand 的 y
    R_adapt = np.array([
        [0, 1, 0],   # new_x = old_y (沿瓶身 → panda x)
        [-1, 0, 0],  # new_y = -old_x (夹爪开合 → panda y, 取反保持右手系)
        [0, 0, 1],   # new_z = old_z (接近方向不变)
    ], dtype=np.float64)
    rot_world = rot_world @ R_adapt

    # TCP 偏移处理 — position = 接触中点, 减去 TCP_OFFSET 得到 panda_hand 腕部位置
    # Franka: panda_hand → 指尖 = 10.5cm (固定段 6.5cm + 夹爪活动段 4cm)
    TCP_OFFSET = 0.105
    approach_dir = rot_world[:, 2]   # 接近方向 = 旋转矩阵第3列
    pos_world = pos_world - approach_dir * TCP_OFFSET

    # Z 安全限制
    MIN_GRASP_Z = TABLE_TOP_Z + 0.02
    if pos_world[2] < MIN_GRASP_Z:
        cprint(f"   ⚠️ Z={pos_world[2]:.3f} 太低 (需 >{MIN_GRASP_Z:.3f}), clamp up", "yellow")
        pos_world[2] = MIN_GRASP_Z

    # 旋转矩阵 → 四元数 wxyz
    q_xyzw = Rotation.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    euler = Rotation.from_matrix(rot_world).as_euler('xyz', degrees=True)

    lift_pos = pos_world.copy()
    lift_pos[2] += LIFT_HEIGHT

    cprint(f"\n🤖 Executing grasp:", "cyan")
    cprint(f"   World pos:  [{pos_world[0]:.4f}, {pos_world[1]:.4f}, {pos_world[2]:.4f}]", "cyan")
    cprint(f"   Euler:      [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]", "cyan")
    cprint(f"   Gripper:    {gripper_width*100:.1f}cm", "cyan")

    # 记录初始物体 Z
    obj_init, _ = scene["obj"].get_obj_pos()
    initial_z = obj_init[2]

    # ---- 初始化 cuRobo (含物体 mesh 障碍物) ----
    if _CUROBO_MG is None:
        try:
            _CUROBO_MG = init_curobo(scene)
        except Exception as e:
            cprint(f"   ❌ cuRobo init failed: {e}", "red")
            return False

    # ---- 打开夹爪 ----
    franka.open_gripper()
    for _ in range(30):
        world.step(render=RENDER_SIM)

    if skip_pre_grasp:
        cprint(f"   → [1/5] Direct approach (--skip-pre-grasp, with object mesh)...", "yellow")
        traj = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz,
                               label="direct", scene=scene, use_object_mesh=True)
        if traj is None:
            cprint(f"   ❌ Direct approach 规划失败", "red")
            return False
        cprint(f"   ✅ Direct approach trajectory: {len(traj)} steps", "green")
    else:
        # ---- Phase 1: 规划到预抓取点 (Pre-grasp) ----
        approach_dir = rot_world[:, 2]
        pre_grasp_offset = 0.15  # 15cm
        pre_grasp_pos = pos_world - approach_dir * pre_grasp_offset

        cprint(f"   → [1/5] Planning to pre-grasp point (with object mesh)...", "yellow")
        cprint(f"      pre-grasp: [{pre_grasp_pos[0]:.4f}, {pre_grasp_pos[1]:.4f}, {pre_grasp_pos[2]:.4f}]", "magenta")
        traj = plan_trajectory(_CUROBO_MG, franka, pre_grasp_pos, quat_wxyz,
                               label="pre-grasp", scene=scene, use_object_mesh=True)

        if traj is None:
            cprint(f"   → Pre-grasp 失败, 直接规划到抓取点...", "yellow")
            traj = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz,
                                   label="direct", scene=scene, use_object_mesh=True)

        if traj is None:
            cprint(f"   ❌ cuRobo 规划全部失败", "red")
            return False

        cprint(f"   ✅ Pre-grasp trajectory: {len(traj)} steps", "green")

    # ---- Phase 2: 执行 approach 轨迹 (pre-grasp 或 direct) ----
    move_label = "direct" if skip_pre_grasp else "pre-grasp"
    cprint(f"   → [2/5] Moving to {move_label}...", "yellow")
    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=RENDER_SIM)

    for _ in range(10):
        world.step(render=RENDER_SIM)

    # ---- Phase 3: 最后接近 (清除物体 mesh — 让夹爪能碰到物体) ----
    cprint(f"   → [3/5] Final approach (table+ground only, no object mesh)...", "yellow")
    traj_final = plan_trajectory(_CUROBO_MG, franka, pos_world, quat_wxyz,
                                 label="final", scene=scene, use_object_mesh=False)
    if traj_final is not None:
        for joint_pos in traj_final:
            gripper = franka.get_joint_positions()[7:9]
            franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
            for _ in range(3):
                world.step(render=RENDER_SIM)
    else:
        cprint(f"   → Final approach 失败", "red")
        return _grasp_result_base(success=False)


    # ---- Phase 4: 闭合夹爪 + 力传感器 ----
    cprint(f"   → [4/5] Closing gripper with force sensing...", "yellow")

    # 设置接触力传感器 (PhysX contact report)
    from pxr import UsdPhysics, PhysxSchema
    stage = world.stage
    left_finger_path = "/World/Franka/panda_leftfinger"
    right_finger_path = "/World/Franka/panda_rightfinger"

    # 开启 contact report
    for finger_path in [left_finger_path, right_finger_path]:
        prim = stage.GetPrimAtPath(finger_path)
        if prim.IsValid():
            cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            cr.CreateThresholdAttr(0.0)  # 报告所有接触力
            cprint(f"      ✅ Contact sensor on {finger_path.split('/')[-1]}", "green")

    franka.close_gripper()

    # 闭合过程中记录力
    force_log = []
    for step in range(80):
        world.step(render=RENDER_SIM)

        # 读取接触力
        from omni.physx import get_physx_scene_query_interface
        try:
            contact_data = get_physx_scene_query_interface().overlap_shape_any(
                left_finger_path, None)
        except:
            pass

        # 记录夹爪位置
        finger_pos = franka.get_joint_positions()[7:9]
        if step % 20 == 0:
            cprint(f"      Step {step}: fingers=[{finger_pos[0]:.4f}, {finger_pos[1]:.4f}]", "cyan")
        force_log.append(finger_pos.copy())

    # 读取闭合后的实际手指位置 (仅用于诊断)
    finger_pos_after = franka.get_joint_positions()[7:9]
    cprint(f"      夹爪位置: [{finger_pos_after[0]:.4f}, {finger_pos_after[1]:.4f}] (宽{(finger_pos_after[0]+finger_pos_after[1])*100:.2f}cm)", "magenta")

    # ⭐ 不再用 set_joint_positions 控制夹爪！
    # close_gripper() 的控制器会持续施加闭合力, 这才是真正产生物理夹持力的方式
    # set_joint_positions 是运动学瞬移, 不产生力

    # 检查夹爪是否在变化 (松开?)
    if len(force_log) > 10:
        early = np.mean([f[0] for f in force_log[:10]])
        late = np.mean([f[0] for f in force_log[-10:]])
        delta = late - early
        if delta > 0.001:
            cprint(f"      ⚠️ 夹爪在过程中张开了 {delta*100:.2f}cm!", "red")
        else:
            cprint(f"      ✅ 夹爪稳定 (变化 {delta*100:.3f}cm)", "green")

    gripper_tips_loc = None
    finger_width_actual = None
    try:
        executed_at_close = snapshot_panda_hand_object_mesh(franka, scene, object_scale)
        p = executed_at_close['position']
        cprint(
            f"   📌 panda_hand@at_close (obj): [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]",
            "magenta",
        )
    except Exception as e:
        executed_at_close = None
        cprint(f"   ⚠️ panda_hand@at_close 记录失败: {e}", "yellow")

    try:
        tips_snap = snapshot_gripper_tips_object_mesh(world.stage, scene, object_scale)
        gripper_tips_loc = tips_snap['gripper_tips_loc']
        finger_width_actual = tips_snap['finger_width_actual']
        cprint(
            f"   📐 gripper_tips@at_close (obj): L={gripper_tips_loc[0]}  "
            f"R={gripper_tips_loc[1]}  width={finger_width_actual*100:.2f}cm",
            "magenta",
        )
    except Exception as e:
        cprint(f"   ⚠️ gripper_tips@at_close 记录失败: {e}", "yellow")

    # ---- Phase 5: 提起 (物体 mesh 仍清除 — 物体随夹爪一起移动) ----
    cprint(f"   → [5/5] Planning lift (no object mesh)...", "yellow")
    traj_lift = plan_trajectory(_CUROBO_MG, franka, lift_pos, quat_wxyz,
                                label="lift", scene=scene, use_object_mesh=False)

    if traj_lift is not None:
        cprint(f"   ✅ Lift trajectory: {len(traj_lift)} steps", "green")
        # close_gripper() 只需调一次, 控制器会持续施力
        franka.close_gripper()
        for joint_pos in traj_lift:
            # 用 apply_action 或直接设 DC target, 只设手臂
            from omni.isaac.core.utils.types import ArticulationAction
            action = ArticulationAction(
                joint_positions=np.concatenate([joint_pos, np.array([None, None])]),
            )
            franka.apply_action(action)
            for _ in range(2):
                world.step(render=RENDER_SIM)
    else:
        cprint(f"   ⚠️ Lift planning failed, skipping", "yellow")

    # 稳定 — 夹爪控制器持续施力
    for _ in range(80):
        world.step(render=RENDER_SIM)

    # ---- 检查结果 ----
    obj_after, _ = scene["obj"].get_obj_pos()
    z_delta = obj_after[2] - initial_z
    success = z_delta > 0.03

    cprint(f"   📍 obj Z: {initial_z:.4f} → {obj_after[2]:.4f} (Δ={z_delta:.4f}m)", "cyan")

    result = _grasp_result_base(success=success)
    result['executed_at_close'] = executed_at_close
    result['gripper_tips_loc'] = gripper_tips_loc
    result['finger_width_actual'] = finger_width_actual

    try:
        executed_post_lift = snapshot_panda_hand_object_mesh(franka, scene, object_scale)
        p = executed_post_lift['position']
        cprint(
            f"   📌 panda_hand@post_lift (obj): [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]",
            "magenta",
        )
    except Exception as e:
        executed_post_lift = None
        cprint(f"   ⚠️ panda_hand@post_lift 记录失败: {e}", "yellow")
    result['executed_post_lift'] = executed_post_lift

    if success:
        cprint(f"   ✅ GRASP SUCCESS!", "green", "on_green")
    else:
        cprint(f"   ❌ GRASP FAILED", "red")

    return result


# ============================================================
# Main
# ============================================================
def main():
    h5_path = args.hdf5
    if not os.path.exists(h5_path):
        cprint(f"❌ HDF5 not found: {h5_path}", "red")
        cprint(f"   先运行: python Pipeline/generate_grasp_pose.py --mesh ...", "yellow")
        simulation_app.close()
        return

    # ---- 读取 HDF5 ----
    cprint("=" * 60, "cyan")
    cprint("Pipeline Stage B: Isaac Sim Grasp Execution (+ per-try video)", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"  HDF5: {h5_path}", "cyan")
    cprint(f"  Record: {args.record_video}  every={args.record_every}  fps={args.record_fps}", "magenta")
    if args.skip_pre_grasp:
        cprint("  Approach: direct only (--skip-pre-grasp), then final (no object mesh)", "magenta")

    file_prerot = None
    _no_rotation = True
    with h5py.File(h5_path, 'r') as f:
        # 兼容两种格式: 自动生成 vs 手动标注
        if "metadata" in f:
            # v2 自动生成格式
            obj_id = f["metadata"].attrs["obj_id"]
            n_contact = f["affordance"].attrs.get("n_contact", 0)
        else:
            # 手动标注格式
            obj_id = f.attrs.get('object_id', f.attrs.get('obj_id', 'unknown'))
            n_contact = 0

        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _tools = os.path.join(_proj, "tools")
        if _tools not in sys.path:
            sys.path.insert(0, _tools)
        from mesh_utils import (
            applied_mesh_prerotation_record,
            read_mesh_prerotation_hdf5_pose,
        )

        _meta = f.get("metadata", None)
        _no_rotation = bool(_meta.attrs.get("no_rotation", True)) if _meta is not None else True
        _dataset = (
            str(_meta.attrs.get("dataset", ""))
            if _meta is not None and "dataset" in _meta.attrs
            else None
        ) or None
        file_prerot = applied_mesh_prerotation_record(
            obj_id, _dataset, no_rotation=_no_rotation,
        )

        grasp_candidates = []

        if "candidates" in f:
            # v2 自动生成: candidates/candidate_N
            n_cand = f["candidates"].attrs["n_candidates"]
            for i in range(n_cand):
                ci = f[f"candidates/candidate_{i}"]
                pose_pr = read_mesh_prerotation_hdf5_pose(ci, f) or file_prerot
                sim_prerot_euler = list(pose_pr["euler_xyz_deg"])
                grasp_candidates.append({
                    "name": ci.attrs["name"],
                    "score": ci.attrs["score"],
                    "position": ci["position"][:],
                    "rotation": ci["rotation"][:],
                    "gripper_width": ci.attrs["gripper_width"],
                    "approach_type": ci.attrs.get("approach_type", "raycast"),
                    "is_manual": False,          # 自动生成: position=panda_hand, 不减 TCP 偏移
                    "mesh_prerotation_euler": sim_prerot_euler,
                    "mesh_prerotation": pose_pr,
                })
            cprint(
                f"  Candidates: {n_cand} (v2 auto, no_rotation={_no_rotation})",
                "cyan",
            )

        elif "candidate_0" in f:
            # 手动标注格式: candidate_N 在根目录
            n_cand = f.attrs.get('num_candidates', 0)
            for i in range(n_cand):
                key = f'candidate_{i}'
                if key not in f: continue
                ci = f[key]
                grasp_candidates.append({
                    "name": ci.attrs.get("name", f"manual_{i}"),
                    "score": ci.attrs.get("score", 80.0),
                    "position": ci["position"][:],
                    "rotation": ci["rotation"][:],
                    "gripper_width": ci["gripper_width"],
                    "approach_type": ci.attrs.get("approach_type", "horizontal"),
                    "is_manual": True,           # 手动标注: position=指尖中心, 需减 TCP 偏移
                    "mesh_prerotation_euler": [0.0, 0.0, 0.0],
                    "mesh_prerotation": file_prerot,
                })
            cprint(f"  Candidates: {len(grasp_candidates)} (manual annotation, is_manual=True)", "cyan")

        else:
            # v1 兼容: 只有一个位姿
            grasp_candidates.append({
                "name": "legacy",
                "score": 0.0,
                "position": f["grasp/position"][:],
                "rotation": f["grasp/rotation"][:],
                "gripper_width": f["grasp"].attrs["gripper_width"],
                "approach_type": "horizontal",
                "mesh_prerotation": file_prerot,
                "mesh_prerotation_euler": list(file_prerot["euler_xyz_deg"]),
            })
            cprint(f"  Candidates: 1 (v1 legacy)", "cyan")

    if file_prerot is None:
        from mesh_utils import applied_mesh_prerotation_record
        file_prerot = applied_mesh_prerotation_record(obj_id, no_rotation=True)

    cprint(f"  Object:    {obj_id}", "cyan")
    cprint(f"  no_rotation (file default): {_no_rotation}", "cyan")
    cprint(f"  Contacts:  {n_contact}", "cyan")
    cprint(f"  Scale:     {args.object_scale}x", "cyan")

    # 按评分降序排列（纯分数，不加方向权重）
    grasp_candidates.sort(key=lambda c: c["score"], reverse=True)
    if args.max_candidates is not None and args.max_candidates > 0:
        grasp_candidates = grasp_candidates[: args.max_candidates]
        cprint(f"  Limited to top {len(grasp_candidates)} candidates (--max-candidates)", "cyan")

    for i, c in enumerate(grasp_candidates):
        R = c["rotation"]
        approach = R[:, 2] if R.shape == (3, 3) else [0, 0, 0]
        approach_str = f"({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})"
        marker = "⭐" if i == 0 else "  "
        cprint(f"  {marker} [{i+1}] {c['name']:>16s}  score={c['score']:5.1f}  "
               f"approach={approach_str}  gripper={c['gripper_width']*100:.1f}cm", "cyan")


    # ---- 搭建场景 ----
    cprint(f"\n📦 Setting up scene...", "yellow")
    scene = setup_scene(obj_id, args.object_scale)
    if scene is None:
        simulation_app.close()
        return

    recorder = TryVideoRecorder(
        args.record_video,
        args.record_every,
        args.record_fps,
        args.record_keep_frames,
    )
    recorder.attach_world(scene["world"])

    cprint("🎥 场景就绪，直接开始抓取+录屏", "cyan")

    # ---- 逐候选尝试抓取 ----
    success = False
    candidate_results = []  # 记录每个候选的结果
    winning_candidate = None

    for attempt, cand in enumerate(grasp_candidates):
        cprint(f"\n{'='*40}", "yellow")
        cprint(f"  🔄 Attempt {attempt+1}/{len(grasp_candidates)}: {cand['name']} (score={cand['score']:.1f})", "yellow")
        cprint(f"{'='*40}", "yellow")

        recorder.start(obj_id, attempt, cand["name"])

        try:
            # 自动生成 (is_manual=False): position=panda_hand, 不减 TCP
            # 手动标注 (is_manual=True):  position=指尖中心, 减 TCP
            is_manual = cand.get("is_manual", True)  # 默认 True (安全: 背景兼容)
            grasp_result = execute_grasp(
                scene,
                cand["position"],
                cand["rotation"],
                cand["gripper_width"],
                args.object_scale,
                is_manual=is_manual,
                mesh_prerotation_euler=cand.get("mesh_prerotation_euler", None),
                skip_pre_grasp=args.skip_pre_grasp,
            )
            # ── fix: execute_grasp 在 cuRobo 初始化失败时可能返回 False ──
            if not isinstance(grasp_result, dict):
                grasp_result = _grasp_result_base(success=bool(grasp_result))
            success = grasp_result['success']
        except Exception as e:
            cprint(f"  ❌ Error: {e}", "red")
            success = False
            grasp_result = _grasp_result_base(success=False)

        video_path = None
        if not args.record_include_reset:
            video_path = recorder.stop(
                success, cand["score"], cand.get("approach_type", "unknown"),
            )

        candidate_results.append({
            'name': cand['name'],
            'score': cand['score'],
            'success': success,
            'grasp_point': cand.get('position', np.zeros(3)),
            'rotation': cand.get('rotation', np.eye(3)),
            'gripper_width': cand['gripper_width'],
            'approach_type': cand.get('approach_type', 'unknown'),
            'gripper_tips_loc': grasp_result.get('gripper_tips_loc'),
            'finger_width_actual': grasp_result.get('finger_width_actual'),
            'mesh_prerotation': cand.get('mesh_prerotation', file_prerot),
            'video_path': video_path,
            'executed_at_close': grasp_result.get('executed_at_close'),
            'executed_post_lift': grasp_result.get('executed_post_lift'),
        })

        if success:
            cprint(f"\n  ✅ SUCCESS with candidate: {cand['name']}", "green")
            if winning_candidate is None:
                winning_candidate = cand  # 记录第一个成功的
        else:
            cprint(f"  ❌ FAILED with candidate: {cand['name']}", "red")

        # 每次尝试后都重置场景 (最后一个除外)
        if attempt < len(grasp_candidates) - 1:
            cprint(f"  → 重置场景, 尝试下一个候选...", "yellow")
            # 读取每物体朝向+高度覆盖 (和 setup_scene 保持一致)
            _ovr = OBJECT_ROTATION_OVERRIDES.get(obj_id, None)
            if isinstance(_ovr, dict):
                reset_z_offset = _ovr.get('z_offset', 0.075 * args.object_scale)
                reset_ori = _ovr.get('rotation', list(OBJECT_ORIENTATION))
            else:
                reset_z_offset = 0.075 * args.object_scale
                reset_ori = list(OBJECT_ORIENTATION)
            reset_pos = list(OBJECT_POSITION)
            reset_pos[2] += reset_z_offset
            # 设定位姿并清零速度
            scene["obj"].set_obj_pose(np.array(reset_pos), ori=np.array(reset_ori))
            try:
                scene["obj"].rigid.set_linear_velocity(np.zeros(3))
                scene["obj"].rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
            # 机械臂回 home
            home_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
            scene["franka"].set_joint_positions(home_joints)
            # 等物理稳定 (多等几步确保鼠标落稳)
            for _ in range(150):
                scene["world"].step(render=RENDER_SIM)

        if args.record_include_reset:
            video_path = recorder.stop(
                success, cand["score"], cand.get("approach_type", "unknown"),
            )
            candidate_results[-1]["video_path"] = video_path

    recorder.write_manifest(obj_id, h5_path)

    # ---- 保存 Robot GT 结果 (所有成功的都保存) ----
    successful_grasps = [cr for cr in candidate_results if cr['success']]
    any_success = len(successful_grasps) > 0

    if args.save_result:
        result_dir = args.result_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "robot_gt")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{obj_id}_robot_gt.hdf5")

        _tools = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
        )
        if _tools not in sys.path:
            sys.path.insert(0, _tools)
        from mesh_utils import write_mesh_prerotation_hdf5

        with h5py.File(result_path, 'w') as rf:
            rf.attrs['obj_id'] = obj_id
            rf.attrs['success'] = any_success
            rf.attrs['n_candidates_tried'] = len(candidate_results)
            rf.attrs['n_candidates_total'] = len(grasp_candidates)
            rf.attrs['n_successful'] = len(successful_grasps)
            rf.attrs['object_scale'] = args.object_scale
            rf.attrs['robot_gt_schema_version'] = 2
            rf.attrs['executed_pose_frame'] = 'object_mesh'
            rf.attrs['executed_ee_frame'] = 'panda_hand'

            # 兼容: 保留 winning_candidate (第一个成功的)
            if winning_candidate is not None:
                wg = rf.create_group('winning_candidate')
                wg.attrs['name'] = winning_candidate['name']
                wg.attrs['score'] = winning_candidate['score']
                wg.attrs['gripper_width'] = winning_candidate['gripper_width']
                wg.attrs['approach_type'] = winning_candidate.get('approach_type', '')
                wg.create_dataset('grasp_point', data=winning_candidate['position'])
                wg.create_dataset('rotation', data=winning_candidate['rotation'])
                wg.create_dataset('approach_dir', data=winning_candidate['rotation'][:, 2])
                wg.create_dataset('finger_dir', data=winning_candidate['rotation'][:, 0])
                wpr = winning_candidate.get('mesh_prerotation', file_prerot)
                write_mesh_prerotation_hdf5(wg, wpr)
                wc = next(
                    (cr for cr in candidate_results if cr['name'] == winning_candidate['name']),
                    None,
                )
                if wc is not None:
                    write_executed_panda_hand_hdf5(wg, wc.get('executed_at_close'), 'at_close')
                    write_executed_panda_hand_hdf5(wg, wc.get('executed_post_lift'), 'post_lift')
                    write_gripper_tips_loc_hdf5(
                        wg, wc.get('gripper_tips_loc'), wc.get('finger_width_actual'),
                    )

            # ★ 所有成功的抓取都保存为 GT
            sg = rf.create_group('successful_grasps')
            sg.attrs['count'] = len(successful_grasps)
            for i, cr in enumerate(successful_grasps):
                gi = sg.create_group(f'grasp_{i}')
                gi.attrs['name'] = cr['name']
                gi.attrs['score'] = cr['score']
                gi.attrs['gripper_width'] = cr['gripper_width']
                gi.attrs['approach_type'] = cr['approach_type']
                gi.create_dataset('grasp_point', data=cr['grasp_point'])
                gi.create_dataset('rotation', data=cr['rotation'])
                gi.create_dataset('approach_dir', data=cr['rotation'][:, 2])
                gi.create_dataset('finger_dir', data=cr['rotation'][:, 0])
                write_gripper_tips_loc_hdf5(
                    gi, cr.get('gripper_tips_loc'), cr.get('finger_width_actual'),
                )
                write_mesh_prerotation_hdf5(
                    gi, cr.get('mesh_prerotation', file_prerot),
                )
                write_executed_panda_hand_hdf5(gi, cr.get('executed_at_close'), 'at_close')
                write_executed_panda_hand_hdf5(gi, cr.get('executed_post_lift'), 'post_lift')

            # 所有候选的结果 (含失败的)
            cg = rf.create_group('candidate_results')
            for i, cr in enumerate(candidate_results):
                ci = cg.create_group(f'candidate_{i}')
                ci.attrs['name'] = cr['name']
                ci.attrs['score'] = cr['score']
                ci.attrs['success'] = cr['success']
                ci.attrs['gripper_width'] = cr['gripper_width']
                ci.attrs['approach_type'] = cr['approach_type']
                if cr.get('video_path'):
                    ci.attrs['video_path'] = cr['video_path']
                ci.create_dataset('grasp_point', data=cr['grasp_point'])
                ci.create_dataset('rotation', data=cr['rotation'])
                write_mesh_prerotation_hdf5(
                    ci, cr.get("mesh_prerotation", file_prerot),
                )
                write_executed_panda_hand_hdf5(ci, cr.get('executed_at_close'), 'at_close')
                write_executed_panda_hand_hdf5(ci, cr.get('executed_post_lift'), 'post_lift')
                write_gripper_tips_loc_hdf5(
                    ci, cr.get('gripper_tips_loc'), cr.get('finger_width_actual'),
                )

        cprint(f"\n  📁 Saved: {result_path}  ({len(successful_grasps)} 个成功GT)", "green")
        cprint(f"  📹 Videos: {os.path.join(args.record_video, obj_id)}/", "magenta")

    # ---- 等待观察 ----
    if not args.headless:
        n_success = len(successful_grasps)
        cprint(f"\n{'=' * 60}", "cyan")
        cprint(f"  结果: {'✅ SUCCESS' if any_success else '❌ ALL FAILED'}  ({n_success}/{len(candidate_results)} 成功, {n_success} 条GT)", "green" if any_success else "red")
        for cr in candidate_results:
            icon = '✅' if cr['success'] else '❌'
            cprint(f"    {icon} {cr['name']:>12s}  score={cr['score']:.1f}", "green" if cr['success'] else "red")
        cprint(f"  保持 5 秒...", "cyan")
        cprint(f"{'=' * 60}", "cyan")
        hold_arm = scene["franka"].get_joint_positions()[:7]
        for _ in range(250):  # ~5秒
            all_j = scene["franka"].get_joint_positions()
            all_j[:7] = hold_arm
            scene["franka"].set_joint_positions(all_j)
            scene["franka"].close_gripper()
            scene["world"].step(render=RENDER_SIM)

    simulation_app.close()


main()

