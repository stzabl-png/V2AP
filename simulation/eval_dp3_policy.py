#!/usr/bin/env python3
"""
sim/eval_dp3_policy.py
======================
Closed-loop IsaacSim evaluation for Baseline1's DP3 policy.

For each (object × N init poses):
  1. Spawn object USD on table, Franka in home pose
  2. Episode loop (closed-loop):
       a. Read sim's object pose + EE pose
       b. Build 4096-pt object-centric point cloud (SAM3D mesh × sim_pose, minus frame-0 centroid)
       c. POST observation to DP3 inference server → get 8 action steps
       d. Execute each action via Franka RMPFlow + gripper command
  3. Success = obj_z_after - obj_z_init > 3cm

Requires:
  • DP3 inference server running on http://127.0.0.1:8765 (see Baseline1/eval/dp3_inference_server.py)
  • USDs at output/obj_usd/{ycb,oakink}/{obj_id}.usd
  • R_align at Baseline1/assets/sam3d_align/[oakink/]{obj_id}.json
  • SAM3D meshes at data_hub/ProcessedData/obj_meshes/{ds}/{obj_id}/mesh.ply

Usage (in env_isaaclab):
  /home/accelerator/miniforge3/envs/env_isaaclab/bin/python sim/eval_dp3_policy.py \\
    --objects A16013 ycb_dex_01 --n-rollouts 5 --headless
"""
from isaacsim import SimulationApp
import argparse, os, sys, json, time

parser = argparse.ArgumentParser(description="Baseline1 DP3 closed-loop sim eval")
parser.add_argument("--objects", nargs="+", required=True,
                    help="object IDs to test (e.g. A16013 ycb_dex_01)")
parser.add_argument("--n-rollouts", type=int, default=3, help="rollouts per object")
parser.add_argument("--max-steps", type=int, default=80, help="max DP3 steps per rollout (8 sub-actions each)")
parser.add_argument("--server-url", default="http://127.0.0.1:8765")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--result-dir", default="output/dp3_eval")
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import trimesh
import requests
from scipy.spatial.transform import Rotation
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

SIM_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SIM_DIR)
sys.path.insert(0, SIM_DIR)
from env_config.robot.Franka import Franka
from env_config.rigid.RigidObject import RigidObject
# Use IsaacSim's bundled GroundPlane directly — partner's Real_Ground requires
# sim/assets_scene/.../default_environment.usd (gitignored, not in our repo)
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial

# ── Scene constants (match partner's run_grasp_sim.py for cross-comparability) ──
ROBOT_POSITION      = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION   = [0.0, 0.0, 90.0]
TABLE_POSITION      = [0.0, 1.0, 0.75]
TABLE_ORIENTATION   = [0.0, 0.0, 0.0]
TABLE_SCALE         = [2.0, 2.0, 0.1]
TABLE_TOP_Z         = 0.80
OBJECT_POSITION     = [0.0, 0.55, TABLE_TOP_Z]
SUCCESS_DZ          = 0.03                  # 3 cm — matches partner's success criterion
SETTLE_STEPS_INIT   = 50
SETTLE_STEPS_GRIP   = 20

# Per-object init overrides (z offset + orientation) — partner's calibration
_OVERRIDE_FILE = os.path.join(SIM_DIR, "object_rotation_overrides.json")
OBJECT_OVERRIDES = {}
if os.path.exists(_OVERRIDE_FILE):
    with open(_OVERRIDE_FILE) as f:
        d = json.load(f)
    OBJECT_OVERRIDES = {k: v for k, v in d.items() if not k.startswith("_")}

CANONICAL_ROT_FILE = os.path.join(SIM_DIR, "canonical_rotation.json")
CANONICAL_ROT = {}
if os.path.exists(CANONICAL_ROT_FILE):
    with open(CANONICAL_ROT_FILE) as f:
        d = json.load(f)
    CANONICAL_ROT = {k: v for k, v in d.items() if not k.startswith("_")}


# ── helpers ───────────────────────────────────────────────────────────────────
def find_usd(obj_id):
    for ds in ("oakink", "ycb"):
        p = os.path.join(PROJ_DIR, "output", "obj_usd", ds, f"{obj_id}.usd")
        if os.path.exists(p):
            return p, ds
    return None, None


def get_object_scale(obj_id, ds):
    """Multiplicative factor SAM3D-mesh → real metres for *both* USD spawn and PC sampling.

    YCB: scale_factor in mesh's scale.json (already baked into convert_batch_usd for USDs;
         we still need it for load_sam3d_pts to get PC in metres).
    OakInk: no scale.json → prescale lives in Baseline1/assets/sam3d_align/oakink/{obj}.json
            (NOT baked into USDs — must be applied at runtime to BOTH the sim object scale
            AND the sampled PC). Mismatch here was the cause of the dz=0 silent failure.
    """
    if ds == "ycb":
        scale_path = os.path.join(PROJ_DIR, "data_hub", "ProcessedData", "obj_meshes",
                                  ds, obj_id, "scale.json")
        if os.path.exists(scale_path):
            return float(json.load(open(scale_path)).get("scale_factor", 1.0))
        return 1.0
    elif ds == "oakink":
        ra_path = os.path.join(PROJ_DIR, "Baseline1", "assets", "sam3d_align",
                               "oakink", f"{obj_id}.json")
        if os.path.exists(ra_path):
            return float(json.load(open(ra_path)).get("prescale", 1.0))
        return 1.0
    return 1.0


def load_sam3d_pts(obj_id, ds, n_points=4096):
    """Sample N points from SAM3D mesh, applying scale_factor and canonical_rotation
    so the points match the USD geometry that sim physics tracks."""
    mesh_path = os.path.join(PROJ_DIR, "data_hub", "ProcessedData", "obj_meshes",
                             ds, obj_id, "mesh.ply")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    sf = get_object_scale(obj_id, ds)   # ycb: scale_factor; oakink: prescale; else 1.0
    if sf != 1.0:
        mesh.vertices = mesh.vertices * sf
    # canonical_rotation: Euler XYZ deg applied to mesh (matches what USD does)
    if obj_id in CANONICAL_ROT:
        euler = CANONICAL_ROT[obj_id]
        R_can = Rotation.from_euler("XYZ", euler, degrees=True).as_matrix()
        mesh.vertices = mesh.vertices @ R_can.T
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)            # in USD local frame


def wxyz_to_R(q):
    """quaternion wxyz → 3x3 rotation matrix."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def R_to_wxyz(R):
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def pose_to_T(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(quat_wxyz)
    T[:3, 3]  = pos
    return T


def sample_obj_pc_world(sam3d_pts_local, obj_world_pose_T):
    """Transform pre-sampled local points to world frame using sim's object pose."""
    h = np.concatenate([sam3d_pts_local, np.ones((sam3d_pts_local.shape[0], 1))], axis=1)
    return (obj_world_pose_T @ h.T).T[:, :3].astype(np.float32)


def build_obs(sam3d_pts_local, obj_pos, obj_quat_wxyz, ee_pos, ee_quat_wxyz, grip_state, centroid_t0):
    """One-frame observation in object-centric frame (centroid-subtracted)."""
    T_obj = pose_to_T(obj_pos, obj_quat_wxyz)
    pc_world = sample_obj_pc_world(sam3d_pts_local, T_obj)
    pc_oc = pc_world - centroid_t0
    ee_pos_oc = ee_pos - centroid_t0
    agent = np.concatenate([ee_pos_oc, ee_quat_wxyz, [grip_state]]).astype(np.float32)
    return pc_oc, agent


def query_policy(server_url, pc_obs, ap_obs, timeout=5.0):
    """POST observation to inference server. pc_obs (T,N,3); ap_obs (T,D)."""
    r = requests.post(f"{server_url}/predict",
                      json={"point_cloud": pc_obs.tolist(),
                            "agent_pos":   ap_obs.tolist()},
                      timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return np.asarray(d["action"], dtype=np.float32)         # (n_action, 8)


def get_policy_info(server_url):
    return requests.get(f"{server_url}/info", timeout=5).json()


# ── scene setup (lean, no friction-tuning fluff) ─────────────────────────────
def setup_scene(usd_path, obj_id, ds):
    world = World(backend="numpy")
    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    set_camera_view(eye=[0.0, 4.5, 3.5], target=[0.0, 0.0, 0.0],
                    camera_prim_path="/OmniverseKit_Persp")
    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")
    physics_material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/ground_mat",
        static_friction=0.5, dynamic_friction=0.5, restitution=0.8,
    )
    GroundPlane(prim_path="/World/defaultGroundPlane", name="default_ground_plane",
                z_position=0, physics_material=physics_material, visual_material=None)

    delete_prim("/World/Table")
    FixedCuboid(prim_path="/World/Table", name="table",
                position=TABLE_POSITION,
                orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True)

    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))
    world.reset()
    for _ in range(SETTLE_STEPS_INIT):
        world.step(render=True)
    franka.open_gripper()

    # object placement (use partner's overrides if present)
    ovr = OBJECT_OVERRIDES.get(obj_id, {})
    z_off = ovr.get("z_offset", 0.075)
    obj_euler = ovr.get("rotation", [0.0, 0.0, 0.0])
    obj_pos_init = list(OBJECT_POSITION); obj_pos_init[2] += z_off
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")
    s = get_object_scale(obj_id, ds)   # ycb: 1.0 (already baked into USD); oakink: prescale ~0.1-0.3
    obj = RigidObject(world, usd_path=usd_path,
                      pos=np.array(obj_pos_init), ori=np.array(obj_euler),
                      scale=np.array([s, s, s]), mass=0.05)
    if s != 1.0:
        cprint(f"  [scale fix] {obj_id} spawned with USD scale={s:.4f} "
               f"(SAM3D→metres; was 1.0 → object was {1/s:.1f}× too big)", "cyan")

    # Collision setup (copied from partner's run_grasp_sim.py) — without this the
    # object's mesh has no collision shape and falls through the table.
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    stage = world.stage
    mat_path = "/World/PhysicsMaterials/ObjMaterial"
    UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    physics_mat.CreateStaticFrictionAttr(1.0)
    physics_mat.CreateDynamicFrictionAttr(0.8)
    physics_mat.CreateRestitutionAttr(0.0)
    obj_prim = stage.GetPrimAtPath(obj.rigid_prim_path)
    for prim in Usd.PrimRange(obj_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.GetApproximationAttr().Set("convexHull")
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col_api.GetContactOffsetAttr().Set(0.02)
            col_api.GetRestOffsetAttr().Set(0.001)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)),
                         UsdShade.Tokens.weakerThanDescendants, "physics")
    # Finger friction
    finger_mat_path = "/World/PhysicsMaterials/FingerMaterial"
    UsdShade.Material.Define(stage, finger_mat_path)
    finger_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(finger_mat_path))
    finger_mat.CreateStaticFrictionAttr(1.2)
    finger_mat.CreateDynamicFrictionAttr(1.0)
    finger_mat.CreateRestitutionAttr(0.0)
    for finger_name in ("panda_leftfinger", "panda_rightfinger"):
        for child in Usd.PrimRange(stage.GetPrimAtPath(f"/World/Franka/{finger_name}")):
            if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(child).Bind(
                    UsdShade.Material(stage.GetPrimAtPath(finger_mat_path)),
                    UsdShade.Tokens.weakerThanDescendants, "physics")

    for _ in range(SETTLE_STEPS_INIT):
        world.step(render=True)
    return world, franka, obj


# ── one rollout ───────────────────────────────────────────────────────────────
def rollout_one(world, franka, obj, sam3d_pts_local, info, server_url,
                max_dp3_steps, rng):
    """One closed-loop rollout. Returns dict(success, n_steps, ...)."""
    n_obs    = int(info["n_obs_steps"])    # typically 2
    n_action = int(info["n_action_steps"]) # typically 8

    # initial state
    obj_pos0, obj_q0 = obj.get_obj_pos()
    ee_pos0, ee_q0 = franka.get_cur_ee_pos()
    initial_obj_z = float(obj_pos0[2])

    # frame-0 centroid (object-centric origin)
    T_obj0 = pose_to_T(obj_pos0, obj_q0)
    pc0_world = sample_obj_pc_world(sam3d_pts_local, T_obj0)
    centroid_t0 = pc0_world.mean(axis=0).astype(np.float64)
    cprint(f"  init: obj@{np.round(obj_pos0,3)} ee@{np.round(ee_pos0,3)} centroid@{np.round(centroid_t0,3)}", "cyan")

    # sliding-window obs buffer (start with frame 0 repeated)
    pc_buf = [build_obs(sam3d_pts_local, obj_pos0, obj_q0, ee_pos0, ee_q0,
                        grip_state=0.0, centroid_t0=centroid_t0)] * n_obs

    gripper_state = 0.0   # 0=open, 1=closed
    grip_action_done = False

    step_log = []
    for step in range(max_dp3_steps):
        # stack the last n_obs observations
        pc_obs = np.stack([b[0] for b in pc_buf])
        ap_obs = np.stack([b[1] for b in pc_buf])

        try:
            actions = query_policy(server_url, pc_obs, ap_obs)
        except Exception as e:
            cprint(f"  ❌ policy server error: {e}", "red")
            return {"success": False, "n_steps": step, "error": str(e)}

        for sub_idx in range(actions.shape[0]):
            a = actions[sub_idx]
            a_pos_world = (a[:3] + centroid_t0).astype(np.float64)
            a_quat_wxyz = a[3:7].astype(np.float64)
            a_grip      = float(a[7])

            # gripper command (once-only transition: open→closed)
            if (not grip_action_done) and a_grip >= 0.5:
                cprint(f"    step {step}.{sub_idx}: close gripper (action.grip={a_grip:.3f})", "magenta")
                franka.close_gripper()
                gripper_state = 1.0
                grip_action_done = True
                # close_gripper blocks 20 sim steps; don't apply rmpflow this sub-step
                continue

            # apply 1 rmpflow step toward target
            franka.Rmpflow_Step_Action(a_pos_world, a_quat_wxyz)
            world.step(render=True)

        # update obs buffer with newest 2 (or n_obs) world observations
        obj_pos, obj_q = obj.get_obj_pos()
        ee_pos, ee_q = franka.get_cur_ee_pos()
        new_obs = build_obs(sam3d_pts_local, obj_pos, obj_q, ee_pos, ee_q,
                            grip_state=gripper_state, centroid_t0=centroid_t0)
        pc_buf = pc_buf[1:] + [new_obs]

        step_log.append({
            "step": step, "obj_z": float(obj_pos[2]),
            "ee_pos": ee_pos.tolist(), "gripper": gripper_state,
        })

        # early exit: object lifted >5cm above initial
        if obj_pos[2] - initial_obj_z > 0.05:
            cprint(f"    step {step}: obj lifted {(obj_pos[2]-initial_obj_z)*100:.1f}cm → early success", "green")
            break

    # final settle (let physics catch up)
    for _ in range(30):
        world.step(render=True)
    obj_pos_final, _ = obj.get_obj_pos()
    dz = float(obj_pos_final[2] - initial_obj_z)
    success = dz > SUCCESS_DZ
    cprint(f"  result: obj_z {initial_obj_z:.3f} → {float(obj_pos_final[2]):.3f}  Δz={dz*100:+.1f}cm  {'✅' if success else '❌'}", "green" if success else "red")
    return {"success": success, "n_steps": len(step_log), "dz": dz, "log": step_log}


def main():
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(PROJ_DIR, args.result_dir), exist_ok=True)

    cprint(f"DP3 server: {args.server_url}", "cyan")
    info = get_policy_info(args.server_url)
    cprint(f"  horizon={info['horizon']}  n_obs={info['n_obs_steps']}  n_action={info['n_action_steps']}", "cyan")

    all_results = {}
    for obj_id in args.objects:
        usd_path, ds = find_usd(obj_id)
        if usd_path is None:
            cprint(f"❌ USD not found for {obj_id} (output/obj_usd/{{ycb,oakink}}/)", "red")
            continue
        cprint(f"\n=== {obj_id} ({ds}) === USD: {usd_path}", "yellow")

        sam3d_pts_local = load_sam3d_pts(obj_id, ds, n_points=4096)
        cprint(f"  SAM3D pts: {sam3d_pts_local.shape}  bbox: {np.round(sam3d_pts_local.min(0),3)}→{np.round(sam3d_pts_local.max(0),3)}", "cyan")

        world, franka, obj = setup_scene(usd_path, obj_id, ds)
        n_succ = 0
        per_run = []
        for k in range(args.n_rollouts):
            cprint(f"  --- rollout {k+1}/{args.n_rollouts} ---", "yellow")
            # randomize object initial position (small jitter)
            jx = float(rng.uniform(-0.03, 0.03))
            jy = float(rng.uniform(-0.03, 0.03))
            ovr = OBJECT_OVERRIDES.get(obj_id, {})
            z_off = ovr.get("z_offset", 0.075)
            obj_pos_jitter = [OBJECT_POSITION[0] + jx,
                              OBJECT_POSITION[1] + jy,
                              OBJECT_POSITION[2] + z_off]
            obj.set_obj_pose(np.array(obj_pos_jitter),
                             np.array(ovr.get("rotation", [0.0, 0.0, 0.0])))
            franka.open_gripper()
            for _ in range(SETTLE_STEPS_INIT):
                world.step(render=True)

            result = rollout_one(world, franka, obj, sam3d_pts_local, info,
                                 args.server_url, args.max_steps, rng)
            per_run.append({"rollout": k, **{k_: v for k_, v in result.items() if k_ != "log"}})
            n_succ += int(result["success"])

        all_results[obj_id] = {"n_total": args.n_rollouts, "n_success": n_succ,
                               "rate": n_succ / args.n_rollouts, "runs": per_run}
        cprint(f"\n  {obj_id}: {n_succ}/{args.n_rollouts} = {n_succ/args.n_rollouts*100:.0f}%", "green" if n_succ > 0 else "red")

    out_path = os.path.join(PROJ_DIR, args.result_dir, f"eval_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "policy_info": info, "results": all_results}, f, indent=2)
    cprint(f"\nwrote results → {out_path}", "cyan")

    # overall summary
    total = sum(r["n_total"] for r in all_results.values())
    succ  = sum(r["n_success"] for r in all_results.values())
    cprint(f"\n=== overall: {succ}/{total} = {succ/total*100:.0f}% ===" if total else "no runs", "green")
    simulation_app.close()


if __name__ == "__main__":
    main()
