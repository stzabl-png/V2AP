#!/usr/bin/env python3
"""
sim/gt_replay_ikpd_v2.py — Gate 3 with per-session sim setup + parallel-to-table pre-position.

What changed vs gt_replay_ikpd.py:
  1. Per-session config (sim_origin_W, Franka base) — frame 0 EE is reachable
  2. 3-stage pre-position to bridge Franka home → trajectory state[0]:
       (a) cartesian interp:   home_EE → (state[0].pos, parallel-to-table orientation)
       (b) orientation slerp:  hold pos, rotate to state[0].quat
       (c) ready to replay
  3. Object spawn uses identity orientation (master_chef_can is cylindrically symmetric;
     true GT orientation matters for asymmetric objects in later phases)

Stack stays IK (Lula) + PhysX implicit PD.

Usage:
  /home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/gt_replay_ikpd_v2.py \\
      --session 142601 --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--session", default="142601",
                    choices=["142601", "142646", "142724", "142815", "142844"],
                    help="DexYCB session ID (5 subject-09 master_chef_can sessions)")
parser.add_argument("--traj", default=None, help="override HDF5 path")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--usd", default="/home/accelerator/UCB_Project/output/obj_usd_cad/ycb/ycb_dex_01.usd")
parser.add_argument("--phys-per-action", type=int, default=5)
parser.add_argument("--ik-pos-tol", type=float, default=0.005)
parser.add_argument("--ik-ori-tol", type=float, default=0.05)
parser.add_argument("--position-only", action="store_true",
                    help="ignore orientation in IK — pure positional control (sanity test)")
parser.add_argument("--grip-delay", type=int, default=0,
                    help="delay gripper close by N frames after grasp_onset_idx (lets fingertips enter can volume)")
parser.add_argument("--video", default=None, help="save PNG frames here (omitted = no recording)")
parser.add_argument("--video-every", type=int, default=3, help="capture 1 frame every N sim steps")
args, _ = parser.parse_known_args()

# ── per-session config ───────────────────────────────────────────────────────
# state[0]_G summary (from /tmp/diag_state0_phase1.py):
#   142601: pos=(+0.34,-0.10,+0.08)  ang=149°
#   142646: pos=(+0.31,+0.01,+0.24)  ang=142°
#   142724: pos=(+0.19,-0.63,+0.05)  ang=140°    (Y=-63cm is huge!)
#   142815: pos=(+0.18,-0.30,+0.22)  ang=173°
#   142844: pos=(+0.26,-0.13,+0.29)  ang=126°
# Goal: pick sim_origin (object xy) + Franka base s.t. state[0]_simW is in Franka's reach
# (~30-40cm from base in +X), and Franka faces -X toward object.
SESSION_CONFIG = {
    "142601": {
        "traj":       "/tmp/gt_replay_ycb01_session142601_cam_master.hdf5",
        "sim_origin": (0.0, 0.30, 0.80),                # object spawn xy + table z
        "robot_pos":  (0.69, 0.20, 0.80),               # Franka base, ~35cm in +X of state[0]
        "robot_ori":  (0.0, 0.0, 180.0),                # face -X toward object
    },
    "142646": {"traj": None, "sim_origin": (0.0, 0.30, 0.80), "robot_pos": (0.66, 0.31, 0.80), "robot_ori": (0., 0., 180.)},
    "142724": {"traj": None, "sim_origin": (0.0, 0.30, 0.80), "robot_pos": (0.54, -0.33, 0.80), "robot_ori": (0., 0., 180.)},
    "142815": {"traj": None, "sim_origin": (0.0, 0.30, 0.80), "robot_pos": (0.53, 0.0, 0.80),  "robot_ori": (0., 0., 180.)},
    "142844": {"traj": None, "sim_origin": (0.0, 0.30, 0.80), "robot_pos": (0.61, 0.17, 0.80), "robot_ori": (0., 0., 180.)},
}
cfg = SESSION_CONFIG[args.session]
TRAJ = args.traj or cfg["traj"]
SIM_ORIGIN_XY = cfg["sim_origin"]
ROBOT_POS = list(cfg["robot_pos"])
ROBOT_ORI = list(cfg["robot_ori"])

TABLE_POS = [0, 1.0, 0.75]; TABLE_SCALE = [2, 2, 0.1]; TABLE_TOP_Z = 0.80
SETTLE_INIT = 50
PREP_STEPS_CART = 200     # cartesian interp from home → (state[0].pos, parallel-to-table)
PREP_HOLD_A = 50          # hold at end of A — was 200 but PD settles within 30 (avoid visual stall)
PREP_STEPS_SLERP = 100    # orientation slerp to state[0].quat at fixed pos
PREP_HOLD_B = 50          # hold at end of B — was 200
HOLD_BEFORE_GRIP = 50     # extra settle before closing gripper — was 150
HOLD_AFTER_GRIP = 100     # let grasp grip the object before lift starts

sim_app = SimulationApp({"headless": args.headless})

import numpy as np, h5py
from scipy.spatial.transform import Rotation, Slerp
from termcolor import cprint
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
import omni.replicator.core as rep

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject


# ── helpers ──────────────────────────────────────────────────────────────────
def quat_wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])
def quat_xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])

# CONVENTION FIX: retarget builds R with local +X = opening axis (thumb→index),
# local +Z = approach. Franka panda_hand has local +Y = opening, local +Z = approach.
# So data and robot differ by a 90° rotation around the local +Z (approach) axis.
# Empirically verified: panda_hand FK shows fingers separated along +Y by 0.05m, not +X.
# We post-multiply every retarget quat by a -90° rotation around local +Z so that
# what was retarget's +X (= opening direction in world) becomes Franka's +Y (= opening dir).
_RETARGET_TO_FRANKA_R = Rotation.from_euler("z", -90, degrees=True)
def retarget_to_franka_quat(q_wxyz_retarget):
    r_retarget = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q_wxyz_retarget)))
    r_franka = r_retarget * _RETARGET_TO_FRANKA_R     # right-multiply = local-frame post-rotation
    return quat_xyzw_to_wxyz(r_franka.as_quat())

def parallel_to_table_quat(ee_pos_W, obj_pos_W):
    """Quat (wxyz) — approach horizontal toward object, opening axis horizontal ⊥ approach.
    Builds R with retarget convention (local +X = opening, +Z = approach), THEN applies
    retarget→Franka swap so what we send to IK matches Franka's panda_hand axis convention."""
    delta = np.asarray(obj_pos_W) - np.asarray(ee_pos_W)
    delta_h = delta.copy(); delta_h[2] = 0.0
    if np.linalg.norm(delta_h) < 1e-6:
        delta_h = np.array([1., 0., 0.])
    ez = delta_h / np.linalg.norm(delta_h)
    ex = np.cross(np.array([0., 0., 1.]), ez)
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0., 1., 0.])
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    R = np.column_stack([ex, ey, ez])
    q_xyzw_retarget = Rotation.from_matrix(R).as_quat()
    return retarget_to_franka_quat(quat_xyzw_to_wxyz(q_xyzw_retarget))


# ── load trajectory ──────────────────────────────────────────────────────────
with h5py.File(TRAJ, "r") as h:
    states  = h["state"][:].copy()
    actions = h["action"][:].copy()
    grasp_onset_idx = int(h.attrs["grasp_onset_idx"])
    n_steps = int(h.attrs["n_steps"])
# Apply retarget→Franka 90° axis swap to all quats in trajectory
for arr in (states, actions):
    for k in range(arr.shape[0]):
        arr[k, 3:7] = retarget_to_franka_quat(arr[k, 3:7])
cprint(f"[{args.session}] loaded {n_steps} steps, gripper onset @ {grasp_onset_idx}", "cyan")
cprint(f"  state[0]:  pos={states[0,:3].round(3)} quat={states[0,3:7].round(3)} grip={states[0,7]:.0f}  (axis-swap applied)", "cyan")
cprint(f"  config:    sim_origin={SIM_ORIGIN_XY} robot_pos={ROBOT_POS} robot_ori={ROBOT_ORI}", "cyan")


# ── scene ────────────────────────────────────────────────────────────────────
world = World(backend="numpy")
phys = world.get_physics_context()
phys.enable_ccd(True); phys.enable_gpu_dynamics(True); phys.set_broadphase_type("gpu")
phys.enable_stablization(True); phys.set_solver_type("TGS")
set_camera_view(eye=[1.5, 1.5, 1.5], target=[0, 0.4, 0.85], camera_prim_path="/OmniverseKit_Persp")

# ── video recording (PNG frames; ffmpeg to mp4 after) ────────────────────────
VIDEO_DIR = args.video
_video_idx = 0
_video_step = 0
if VIDEO_DIR:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    # clean old frames
    for p in os.listdir(VIDEO_DIR):
        if p.endswith(".png"): os.remove(os.path.join(VIDEO_DIR, p))
    import omni.kit.viewport.utility as _vu
    _viewport = _vu.get_active_viewport()
    cprint(f"📹 video recording on → {VIDEO_DIR}/  every {args.video_every} steps", "magenta")

def _capture_step():
    global _video_idx, _video_step
    if not VIDEO_DIR: return
    _video_step += 1
    if _video_step % args.video_every != 0: return
    _vu.capture_viewport_to_file(_viewport, os.path.join(VIDEO_DIR, f"f_{_video_idx:05d}.png"))
    _video_idx += 1

# wrap world.step so every call may capture a frame
_orig_world_step = world.step
def world_step_with_capture(render=True):
    _orig_world_step(render=render)
    _capture_step()
world.step = world_step_with_capture

delete_prim("/Replicator/DomeLight_Xform")
rep.create.light(position=[0, 0, 0], light_type="dome")
GroundPlane(prim_path="/World/defaultGroundPlane", z_position=0,
            physics_material=PhysicsMaterial(prim_path="/World/PM/g",
                                             static_friction=0.5, dynamic_friction=0.5, restitution=0.8),
            visual_material=None)
delete_prim("/World/Table")
FixedCuboid(prim_path="/World/Table", name="table", position=TABLE_POS,
            orientation=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
            scale=TABLE_SCALE, size=1.0, visible=True)
delete_prim("/World/Franka")
franka = Franka(world, np.array(ROBOT_POS), np.array(ROBOT_ORI))
world.reset()
for _ in range(SETTLE_INIT): world.step(render=True)
franka.open_gripper()

ik = KinematicsSolver(franka, end_effector_frame_name="panda_hand")
# CRITICAL: Lula IK uses world-frame targets, but it doesn't auto-discover the robot's
# world-frame base pose. Default = (0,0,0) identity — wrong when ROBOT_POS != origin.
# Must explicitly inform it of where the robot is in the world.
_franka_base_quat_wxyz = euler_angles_to_quat(np.array(ROBOT_ORI), degrees=True)
ik._kinematics.set_robot_base_pose(np.array(ROBOT_POS, dtype=np.float64),
                                    np.asarray(_franka_base_quat_wxyz, dtype=np.float64))
cprint(f"✓ Lula KinematicsSolver ready  (base_W={ROBOT_POS}, ori={ROBOT_ORI}°, ee_frame=panda_hand)", "green")

def measure_ee_W():
    """EE pose at panda_hand frame in WORLD coords (consistent with IK frame).
    Franka.get_cur_ee_pos returns panda_rightfinger which is offset 5-10cm — don't mix."""
    p, R = ik._kinematics_solver.compute_forward_kinematics("panda_hand", franka.get_joint_positions()[:7])
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.asarray(p), quat_xyzw_to_wxyz(q_xyzw)


# ── object spawn ─────────────────────────────────────────────────────────────
sim_origin_W = np.array([SIM_ORIGIN_XY[0], SIM_ORIGIN_XY[1], TABLE_TOP_Z])
HALF_CAN_HEIGHT = 0.070
obj_init_pos = [sim_origin_W[0], sim_origin_W[1], TABLE_TOP_Z + HALF_CAN_HEIGHT]
for i in range(10): delete_prim(f"/World/Rigid/rigid_{i}")
delete_prim("/World/Rigid/rigid")
obj = RigidObject(world, usd_path=args.usd, pos=np.array(obj_init_pos),
                  ori=np.array([0., 0., 0.]), scale=np.array([1., 1., 1.]), mass=0.1)

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
stage = world.stage
mat = "/World/PM/obj"
UsdShade.Material.Define(stage, mat)
pm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat))
pm.CreateStaticFrictionAttr(1.0); pm.CreateDynamicFrictionAttr(0.8); pm.CreateRestitutionAttr(0.0)
obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
for prim in Usd.PrimRange(obj_prim):
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim)
        mc = UsdPhysics.MeshCollisionAPI.Apply(prim); mc.GetApproximationAttr().Set("convexHull")
        co = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        co.GetContactOffsetAttr().Set(0.02); co.GetRestOffsetAttr().Set(0.001)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat)),
            UsdShade.Tokens.weakerThanDescendants, "physics")
for _ in range(SETTLE_INIT): world.step(render=True)

obj_pos_settled, _ = obj.get_obj_pos()
initial_obj_z = float(obj_pos_settled[2])
# re-zero sim_origin to settled object xy (tiny drift possible)
sim_origin_W = np.array([obj_pos_settled[0], obj_pos_settled[1], TABLE_TOP_Z])
cprint(f"\nobject settled at {obj_pos_settled.round(4)}; sim_origin_W={sim_origin_W.round(3)}", "cyan")

ee_home_W, q_home = measure_ee_W()
cprint(f"Franka home: pos={ee_home_W.round(3)} quat={q_home.round(3)}", "cyan")


# ── OFFLINE IK: precompute the full qpos trajectory before sim runs ──────────
# Why: per-step online IK has 3 problems we want to fix:
#   (a) wrist-flip / branch-switching when IK has multiple solutions (no warm-start chain)
#   (b) failures only surface after sim already stepped → wasted work
#   (c) IK calls inside the physics loop block CPU; offline batches once
# We build ONE qpos sequence covering pre-position A + B + 73 trajectory frames,
# using Lula's warm_start (current joint positions at call time) for continuity.
# At sim run time, we just apply_action(qpos[k]) per step. Zero IK in the hot loop.

ARM_DOF = 7  # panda_joint1..7

def precompute_ik_sequence(targets_pose, seed_qpos):
    """targets_pose: list of (pos_W, quat_wxyz, label_str)
    seed_qpos: initial joint positions (used as warm-start for the FIRST IK).
    Subsequent IKs warm-start from the PREVIOUS IK's qpos → forces a continuous IK branch.
    Returns: (qpos_list, ok_list, where qpos_list[i] is np.array(ARM_DOF,) or None if failed)."""
    qpos_list, ok_list = [], []
    cur_seed = np.array(seed_qpos[:ARM_DOF], dtype=np.float64)
    for i, (pos, quat, _label) in enumerate(targets_pose):
        kw = dict(target_position=np.asarray(pos, dtype=np.float64),
                  position_tolerance=args.ik_pos_tol)
        if not args.position_only and quat is not None:
            kw["target_orientation"] = np.asarray(quat, dtype=np.float64)
            kw["orientation_tolerance"] = args.ik_ori_tol
        # Lula's IK warm-starts from the current articulation joint positions; to bias
        # toward the previous solution, we temporarily set the articulation to cur_seed.
        franka.set_joint_positions(np.concatenate([cur_seed, franka.get_joint_positions()[ARM_DOF:]]))
        action, success = ik.compute_inverse_kinematics(**kw)
        if success:
            qpos = np.asarray(action.joint_positions[:ARM_DOF], dtype=np.float64)
            qpos_list.append(qpos); ok_list.append(True)
            cur_seed = qpos          # next IK warm-starts from this qpos
        else:
            qpos_list.append(None); ok_list.append(False)
    return qpos_list, ok_list

def analyze_qpos_continuity(qpos_list, label):
    """Report max joint-step between consecutive frames (wrist flips show up as huge jumps)."""
    valid = [(i, q) for i, q in enumerate(qpos_list) if q is not None]
    if len(valid) < 2: return
    max_step_per_joint = np.zeros(ARM_DOF); max_step_frame = -1; max_step_norm = 0.0
    for k in range(1, len(valid)):
        _, q_prev = valid[k-1]; _, q_curr = valid[k]
        step = np.abs(q_curr - q_prev)
        if step.max() > max_step_per_joint.max():
            max_step_per_joint = step; max_step_frame = valid[k][0]; max_step_norm = np.linalg.norm(step)
    cprint(f"   [{label}] joint-step max: |Δ|_∞={max_step_per_joint.max():.2f}rad ({np.rad2deg(max_step_per_joint.max()):.0f}°) "
           f"on joint{int(np.argmax(max_step_per_joint))+1} at frame {max_step_frame}; "
           f"per-joint max={np.round(np.rad2deg(max_step_per_joint),0).astype(int).tolist()}°",
           "green" if max_step_per_joint.max() < 0.5 else ("yellow" if max_step_per_joint.max() < 1.0 else "red"))


# ── build target sequences (Cartesian targets, before IK) ────────────────────
state0_pos_W = states[0, :3] + sim_origin_W
state0_quat = states[0, 3:7]
pre_quat = parallel_to_table_quat(state0_pos_W, sim_origin_W)

# Stage A: cartesian interp home → (state[0].pos, parallel-to-table)
key_times = [0.0, 1.0]
key_rots_A = Rotation.from_quat(np.vstack([quat_wxyz_to_xyzw(q_home), quat_wxyz_to_xyzw(pre_quat)]))
slerp_A = Slerp(key_times, key_rots_A)
targets_A = []
for k in range(PREP_STEPS_CART):
    alpha = (k + 1) / PREP_STEPS_CART
    p = (1 - alpha) * ee_home_W + alpha * state0_pos_W
    q = quat_xyzw_to_wxyz(slerp_A([alpha]).as_quat()[0])
    targets_A.append((p, q, f"A_{k}"))

# Stage B: orientation slerp at state[0].pos (parallel-to-table → state[0].quat)
key_rots_B = Rotation.from_quat(np.vstack([quat_wxyz_to_xyzw(pre_quat), quat_wxyz_to_xyzw(state0_quat)]))
slerp_B = Slerp(key_times, key_rots_B)
targets_B = []
for k in range(PREP_STEPS_SLERP):
    alpha = (k + 1) / PREP_STEPS_SLERP
    q = quat_xyzw_to_wxyz(slerp_B([alpha]).as_quat()[0])
    targets_B.append((state0_pos_W, q, f"B_{k}"))

# Trajectory: 73 frames from HDF5
targets_traj = [(actions[t, :3] + sim_origin_W, actions[t, 3:7], f"T_{t}") for t in range(n_steps)]

# ── precompute (this is the offline IK batch — no sim time spent here) ──────
cprint(f"\n🧮 Offline IK precompute  (stages: A={len(targets_A)}  B={len(targets_B)}  Traj={len(targets_traj)})", "yellow")
home_qpos = franka.get_joint_positions()[:ARM_DOF].copy()
qpos_A, ok_A = precompute_ik_sequence(targets_A, home_qpos)
qpos_B, ok_B = precompute_ik_sequence(targets_B,
                                       (next((q for q in reversed(qpos_A) if q is not None), home_qpos)))
qpos_traj, ok_traj = precompute_ik_sequence(targets_traj,
                                             (next((q for q in reversed(qpos_B) if q is not None), home_qpos)))

# IMPORTANT: precompute mutated joint_positions to seed each IK; restore real home before sim runs
franka.set_joint_positions(np.concatenate([home_qpos, franka.get_joint_positions()[ARM_DOF:]]))

# Stats / sanity
def _ok_str(oks): return f"{sum(oks)}/{len(oks)} ({100*sum(oks)/max(len(oks),1):.0f}%)"
cprint(f"   IK success — A: {_ok_str(ok_A)}  B: {_ok_str(ok_B)}  Traj: {_ok_str(ok_traj)}", "cyan")
analyze_qpos_continuity(qpos_A, "stage A")
analyze_qpos_continuity(qpos_B, "stage B")
analyze_qpos_continuity(qpos_traj, "trajectory")

if not all(q is not None for q in qpos_traj):
    bad = [i for i, q in enumerate(qpos_traj) if q is None]
    cprint(f"   ⚠️  trajectory has {len(bad)} unreachable frames: {bad[:10]}{'...' if len(bad)>10 else ''}", "red")


# ── online: drive PhysX through the cached qpos sequences (zero IK in loop) ──
from isaacsim.core.utils.types import ArticulationAction

def drive_qpos(qpos, n_phys_steps):
    """Set joint position target → step physics. Skip if qpos is None (IK failed for this frame)."""
    if qpos is not None:
        franka._articulation_controller.apply_action(
            ArticulationAction(joint_positions=np.concatenate([qpos, np.array([np.nan, np.nan])])))
    for _ in range(n_phys_steps): world.step(render=True)

# Stage A
cprint(f"\n🚀 Pre-position A: drive {PREP_STEPS_CART} cached qpos targets (1 phys/step)", "yellow")
for qp in qpos_A: drive_qpos(qp, 1)
for _ in range(PREP_HOLD_A):                       # hold final A target while PD converges
    drive_qpos(qpos_A[-1] if qpos_A[-1] is not None else home_qpos, 1)
ee_A, _ = measure_ee_W()
dist_A = float(np.linalg.norm(ee_A - state0_pos_W))
cprint(f"   end of A: ee={ee_A.round(3)} dist={dist_A*100:.1f}cm", "green" if dist_A < 0.02 else "yellow")

# Stage B
cprint(f"\n🌀 Pre-position B: drive {PREP_STEPS_SLERP} cached qpos (1 phys/step)", "yellow")
for qp in qpos_B: drive_qpos(qp, 1)
for _ in range(PREP_HOLD_B):
    drive_qpos(qpos_B[-1] if qpos_B[-1] is not None else (qpos_A[-1] if qpos_A[-1] is not None else home_qpos), 1)
ee_B, q_B = measure_ee_W()
dist_B = float(np.linalg.norm(ee_B - state0_pos_W))
quat_err_B_deg = np.rad2deg(2 * np.arccos(min(1.0, abs(np.dot(q_B, state0_quat)))))
cprint(f"   end of B: ee={ee_B.round(3)} dist={dist_B*100:.1f}cm quat_err={quat_err_B_deg:.0f}°",
       "green" if (dist_B < 0.02 and quat_err_B_deg < 20) else "yellow")

# re-zero obj z (prep motion may have nudged it)
obj_pos_post, _ = obj.get_obj_pos()
initial_obj_z = float(obj_pos_post[2])
cprint(f"   object after prep: {obj_pos_post.round(3)}  (initial_obj_z={initial_obj_z:.4f})", "cyan")


# ── Trajectory replay (cached qpos, gripper logic unchanged) ─────────────────
cprint(f"\n🎬 Replay {n_steps} cached qpos targets (phys_per_action={args.phys_per_action})", "green")
gripper_closed = False
ik_fail_count = sum(1 for q in qpos_traj if q is None)
ee_track_errs_mm = []
last_valid_qpos = qpos_B[-1] if qpos_B[-1] is not None else home_qpos

for t in range(n_steps):
    a_grip = float(actions[t, 7])
    if (not gripper_closed) and a_grip >= 0.5 and t >= (grasp_onset_idx + args.grip_delay):
        last_qp = qpos_traj[t-1] if (t > 0 and qpos_traj[t-1] is not None) else last_valid_qpos
        cprint(f"  step {t:3d}: HOLDING at last cached qpos before gripper close...", "magenta")
        for _ in range(HOLD_BEFORE_GRIP): drive_qpos(last_qp, 1)
        ee_grip, _ = measure_ee_W()
        obj_grip, _ = obj.get_obj_pos()
        cprint(f"    ee at close: {ee_grip.round(3)}  obj: {obj_grip.round(3)}  "
               f"|ee-obj|_xy={np.linalg.norm(ee_grip[:2] - obj_grip[:2])*100:.1f}cm  "
               f"ee_z-obj_z={(ee_grip[2]-obj_grip[2])*100:+.1f}cm", "magenta")
        franka.close_gripper(); gripper_closed = True
        for _ in range(HOLD_AFTER_GRIP): world.step(render=True)
        continue

    qp = qpos_traj[t]
    if qp is not None: last_valid_qpos = qp
    drive_qpos(qp if qp is not None else last_valid_qpos, args.phys_per_action)

    ee_now_W, _ = measure_ee_W()
    target_pos_W = actions[t, :3] + sim_origin_W
    track_err_mm = float(np.linalg.norm(ee_now_W - target_pos_W) * 1000.0)
    ee_track_errs_mm.append(track_err_mm)
    if t % 10 == 0:
        obj_now_W, _ = obj.get_obj_pos()
        cprint(f"  t={t:3d} target={target_pos_W.round(3)} ee={ee_now_W.round(3)} "
               f"track={track_err_mm:4.0f}mm obj_z={obj_now_W[2]:.3f}", "cyan")


# ── result ───────────────────────────────────────────────────────────────────
for _ in range(30): world.step(render=True)
obj_pos_final, _ = obj.get_obj_pos()
dz = float(obj_pos_final[2]) - initial_obj_z

cprint(f"\n{'=' * 60}", "yellow")
cprint(f"=== Gate 3 v2 RESULT (session {args.session}) ===", "yellow")
cprint(f"  initial obj z: {initial_obj_z:.4f}", "yellow")
cprint(f"  final obj z:   {obj_pos_final[2]:.4f}", "yellow")
cprint(f"  dz: {dz * 100:+.1f} cm", "yellow")
cprint(f"  Offline IK — A: {_ok_str(ok_A)}  B: {_ok_str(ok_B)}  Traj: {_ok_str(ok_traj)}", "yellow")
cprint(f"  Pre-position A end_pos_err={dist_A*100:.1f}cm   B end_pos_err={dist_B*100:.1f}cm quat_err={quat_err_B_deg:.0f}°", "yellow")
cprint(f"  Replay unreachable frames (skipped): {ik_fail_count}/{n_steps}", "yellow")
if ee_track_errs_mm:
    cprint(f"  Replay EE tracking: avg {np.mean(ee_track_errs_mm):.0f}mm  "
           f"max {np.max(ee_track_errs_mm):.0f}mm  p50 {np.median(ee_track_errs_mm):.0f}mm",
           "cyan")

if dz > 0.03:
    cprint(f"  ✅ GATE 3 PASS: dz > 3cm", "green")
elif dz > 0.005:
    cprint(f"  ⚠️  PARTIAL: object moved {dz * 100:.1f}cm but didn't fully lift", "yellow")
else:
    cprint(f"  ❌ FAIL: dz≈0", "red")

sim_app.close()
