#!/usr/bin/env python3
"""
sim2real/run_dual_robot.py  –  Dual-robot retarget visualisation

  Table A (x=0)  : Franka + Gripper  → raw grasp candidate
  Table B (+2m X): Dexmate + Sharpa  → retargeted pose

Usage:
  /home/lyh/isaac-sim/python.sh sim2real/run_dual_robot.py \
      --obj-id A16013 \
      --candidate-hdf5 output/graspnet_rank2/A16013_grasp.hdf5
"""
from __future__ import annotations
import argparse, sys, os, traceback
from pathlib import Path

# ── pre-SimulationApp path setup ─────────────────────────────────────────────
PROJ    = Path(__file__).resolve().parents[1]
SIM_DIR = PROJ / "sim"
for _p in [str(PROJ), str(SIM_DIR), str(PROJ / "sim2real")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--obj-id",           required=True)
    p.add_argument("--candidate-hdf5",   required=True)
    p.add_argument("--candidate-index",  type=int,   default=0)
    p.add_argument("--z-yaw-deg",        type=float, default=0.0)
    p.add_argument("--object-scale",     type=float, default=1.0)
    p.add_argument("--ee-retarget-yaml", default=None)
    p.add_argument("--table-x-offset",  type=float, default=2.0)
    p.add_argument("--headless",         action="store_true")
    p.add_argument("--hold-steps",       type=int,   default=300)
    return p


def main():
    args, _ = build_parser().parse_known_args()
    headless = args.headless or not os.environ.get("DISPLAY")

    from isaacsim import SimulationApp
    sim_app = SimulationApp({
        "headless": headless,
        # RTX Realtime quality fixes (RTX 50-series / Ubuntu artifact mitigation)
        # DLSS Quality (0=Performance, 1=Balanced, 2=Quality, 3=UltraQuality, 4=DLAA)
        "rtx/post/dlss/execMode": 2,
        # Firefly filter: suppress black-pixel spike artifacts from low-sample RT
        "rtx/rtpt/fireflyFilter/enabled": True,
    })
    print("[sim2real] IsaacSim started", flush=True)

    try:
        _run(args, headless)
    except Exception:
        print("\n[sim2real] ═══ FATAL ERROR ═══", flush=True)
        traceback.print_exc()
    finally:
        sim_app.close()


def _run(args, headless):
    # Re-ensure paths after Isaac Sim may reset sys.path
    for _p in [str(PROJ), str(SIM_DIR), str(PROJ / "sim2real")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # Apply RTX rendering quality settings at runtime
    try:
        import carb
        s = carb.settings.get_settings()
        s.set("/rtx/post/dlss/execMode", 2)          # DLSS Quality
        s.set("/rtx/rtpt/fireflyFilter/enabled", True) # suppress black pixel spikes
        s.set("/rtx/post/aa/op", 4)                   # DLAA (fallback AA)
    except Exception:
        pass  # carb not available in headless; harmless

    import numpy as np
    import h5py
    from termcolor import cprint
    from scipy.spatial.transform import Rotation

    from isaacsim.core.api              import World
    from isaacsim.core.api.objects      import FixedCuboid
    from isaacsim.core.utils.prims      import delete_prim
    from isaacsim.core.utils.rotations  import euler_angles_to_quat
    from isaacsim.core.utils.viewports  import set_camera_view
    from isaacsim.core.utils.stage      import add_reference_to_stage
    import omni.replicator.core as rep

    # ── sim2real-local imports (no sim.evaluation dependency) ────────────────
    from sim2real.scene_builder import (
        FRANKA_ROBOT_POSITION, FRANKA_ROBOT_ORIENTATION,
        TABLE_A_POSITION, TABLE_B_POSITION,
        TABLE_ORIENTATION, TABLE_SCALE, TABLE_TOP_Z, TABLE_X_OFFSET,
        DEXMATE_ROBOT_POSITION, DEXMATE_ROBOT_ORIENTATION,
        DEXMATE_URDF_PATH, DEXMATE_USD_PATH,
        SHARPA_USD_PATH, SHARPA_URDF_PATH,
        resolve_object_placement,
    )
    from sim2real.retarget_utils import (
        make_transform, rot_to_quat_wxyz,
        candidate_to_world, franka_ee_world, dexmate_ee_world,
        pre_grasp_pos, lift_pos, load_T_ee_pinch,
    )
    from sim2real.franka_curobo import (
        init_franka_curobo, plan_franka, TCP_OFFSET,
    )
    from sim2real.dexmate_curobo import (
        init_dexmate_curobo, plan_dexmate_trajectory,
    )

    # env_config lives inside sim/
    sys.path.insert(0, str(SIM_DIR))
    from env_config.rigid.RigidObject import RigidObject
    from env_config.robot.Franka      import Franka

    render = not headless

    # ── 1. Load candidate ────────────────────────────────────────────────────
    cprint("[sim2real] loading candidate…", "cyan")
    with h5py.File(args.candidate_hdf5, "r") as f:
        grp   = f.get("candidates") or f
        keys  = sorted(k for k in grp.keys() if not k.startswith("_"))
        idx   = min(args.candidate_index, len(keys)-1)
        cand  = grp[keys[idx]]
        pos_o = np.array(cand["position"], dtype=np.float64).reshape(3)
        rot_o = np.array(cand["rotation"], dtype=np.float64).reshape(3,3)
        score = float(cand.attrs.get("score", 0.0))
        prerot= list(cand.attrs.get("mesh_prerotation_euler", [0.,0.,0.]))
        meta_scale = float(f["metadata"].attrs.get("scale_factor",1.0)) if "metadata" in f else 1.0
    scale = args.object_scale if args.object_scale != 1.0 else meta_scale
    cprint(f"   {keys[idx]}  score={score:.4f}  scale={scale:.4f}", "cyan")

    # ── 2. Object placement ──────────────────────────────────────────────────
    pl_a = resolve_object_placement(args.obj_id, scale, args.z_yaw_deg, x_offset=0.0)
    pl_b = resolve_object_placement(args.obj_id, scale, args.z_yaw_deg, x_offset=TABLE_X_OFFSET)

    # ── 3. World ─────────────────────────────────────────────────────────────
    world   = World(backend="numpy")
    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    set_camera_view(
        eye=[TABLE_X_OFFSET/2, 5.5, 4.0],
        target=[TABLE_X_OFFSET/2, 0.5, TABLE_TOP_Z],
        camera_prim_path="/OmniverseKit_Persp",
    )
    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0,0,0], light_type="dome")

    env_usd = SIM_DIR / "assets_scene" / "Collected_default_environment" / "default_environment.usd"
    delete_prim("/World/Environment")
    if env_usd.exists():
        add_reference_to_stage(usd_path=str(env_usd), prim_path="/World/Environment")

    delete_prim("/World/Ground")
    FixedCuboid(prim_path="/World/Ground", name="ground",
                position=np.array([TABLE_X_OFFSET/2, 0., -0.025]),
                scale=np.array([TABLE_X_OFFSET+4., 20., 0.05]),
                size=1.0, visible=False)

    delete_prim("/World/TableA")
    FixedCuboid(prim_path="/World/TableA", name="table_a",
                position=TABLE_A_POSITION,
                orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True)

    delete_prim("/World/TableB")
    FixedCuboid(prim_path="/World/TableB", name="table_b",
                position=TABLE_B_POSITION,
                orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
                scale=TABLE_SCALE, size=1.0, visible=True)

    # ── Franka ───────────────────────────────────────────────────────────────
    delete_prim("/World/Franka")
    franka = Franka(world, np.array(FRANKA_ROBOT_POSITION), np.array(FRANKA_ROBOT_ORIENTATION))

    # ── Dexmate ──────────────────────────────────────────────────────────────
    from pxr import UsdGeom, Gf
    stage = world.stage
    dexmate_ok  = False
    dexmate_path = "/World/DexmateVega"  # target logical path

    if Path(DEXMATE_USD_PATH).exists():
        delete_prim("/World/DexmateVega")
        add_reference_to_stage(usd_path=DEXMATE_USD_PATH, prim_path="/World/DexmateVega")
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/DexmateVega"))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*DEXMATE_ROBOT_POSITION))
        xf.AddRotateXYZOp().Set(Gf.Vec3d(*DEXMATE_ROBOT_ORIENTATION))
        dexmate_ok = True
        cprint("[sim2real] Dexmate from USD", "green")
    elif Path(DEXMATE_URDF_PATH).exists():
        try:
            import omni.kit.commands
            from isaacsim.asset.importer.urdf import _urdf
            import_config = _urdf.ImportConfig()
            import_config.merge_fixed_joints = False
            import_config.fix_base           = True   # keep robot fixed in place
            # URDFParseAndImportFile returns (status, prim_path_on_stage)
            result = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=DEXMATE_URDF_PATH,
                import_config=import_config,
                dest_path="",
            )
            # result is (True, "/vega_1") — do NOT rename, joints use absolute refs
            prim_path_str = result[1] if isinstance(result, (tuple, list)) else str(result)
            if not prim_path_str:
                raise RuntimeError("URDF import returned empty prim path")
            dexmate_path = prim_path_str
            prim = stage.GetPrimAtPath(prim_path_str)
            if not prim.IsValid():
                raise RuntimeError(f"Prim not found at {prim_path_str}")
            # Translate + rotate to Table B position
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*DEXMATE_ROBOT_POSITION))
            xf.AddRotateXYZOp().Set(Gf.Vec3d(*DEXMATE_ROBOT_ORIENTATION))
            dexmate_ok = True
            cprint(f"[sim2real] Dexmate from URDF → {prim_path_str}", "green")
        except Exception as e:
            cprint(f"[sim2real] Dexmate URDF import failed: {e}", "red")
    else:
        cprint(f"[sim2real] Dexmate URDF not found: {DEXMATE_URDF_PATH}", "red")

    # ── Sharpa — attach to actual R_ee prim path ──────────────────────────────
    sharpa_ok = False
    if dexmate_ok and Path(SHARPA_USD_PATH).exists():
        # R_ee is a link inside the articulation; its stage path follows the robot root
        r_ee_candidates = [
            f"{dexmate_path}/R_ee",
            f"{dexmate_path}/links/R_ee",
            f"{dexmate_path}/R_arm_l8",  # fallback: last right arm link
        ]
        r_ee_prim = None
        for candidate in r_ee_candidates:
            p = stage.GetPrimAtPath(candidate)
            if p.IsValid():
                r_ee_prim = p; break
        if r_ee_prim is not None:
            sharpa_attach = f"{r_ee_prim.GetPath()}/SharpaRight"
            add_reference_to_stage(usd_path=SHARPA_USD_PATH, prim_path=sharpa_attach)
            sharpa_ok = True
            cprint(f"[sim2real] Sharpa Wave → {sharpa_attach}", "green")
        else:
            cprint("[sim2real] R_ee prim not found, Sharpa skipped", "yellow")
    elif dexmate_ok:
        cprint(f"[sim2real] Sharpa USD not found: {SHARPA_USD_PATH}", "yellow")

    # ── Physics init + objects ────────────────────────────────────────────────
    world.reset()
    for _ in range(60): world.step(render=render)
    franka.open_gripper()
    for _ in range(10): world.step(render=render)

    # ── Objects ───────────────────────────────────────────────────────────────
    obj_a = RigidObject(world, usd_path=pl_a["usd_path"],
                        pos=np.array(pl_a["pos"]), ori=np.array(pl_a["ori"]),
                        scale=np.array([scale]*3), mass=0.05)
    _apply_object_physics(world.stage, obj_a, mat_suffix="A")

    obj_b = RigidObject(world, usd_path=pl_b["usd_path"],
                        pos=np.array(pl_b["pos"]), ori=np.array(pl_b["ori"]),
                        scale=np.array([scale]*3), mass=0.05)
    _apply_object_physics(world.stage, obj_b, mat_suffix="B")

    for _ in range(150): world.step(render=render)
    cprint("[sim2real] Scene ready ✅", "green")


    # ── 4. Compute grasp targets ─────────────────────────────────────────────
    obj_a_pos, obj_a_quat = obj_a.get_obj_pos()
    T_world_obj = make_transform(obj_a_pos, obj_a_quat)

    g_pos_w, g_rot_w, g_quat_w = candidate_to_world(
        pos_o, rot_o, T_world_obj, object_scale=scale, mesh_prerotation_euler=prerot)

    f_pos_w, f_rot_w, f_quat_w = franka_ee_world(g_pos_w, g_rot_w, TABLE_TOP_Z)

    T_ee_pinch = load_T_ee_pinch(args.ee_retarget_yaml)
    d_pos_w, d_rot_w, d_quat_w = dexmate_ee_world(
        g_pos_w, g_rot_w, T_ee_pinch,
        robot_base_position=DEXMATE_ROBOT_POSITION,
        robot_base_yaw_deg=DEXMATE_ROBOT_ORIENTATION[2],
        table_top_z=TABLE_TOP_Z)
    d_pos_b = d_pos_w.copy(); d_pos_b[0] += TABLE_X_OFFSET

    f_pre  = pre_grasp_pos(f_pos_w, f_rot_w)
    f_lift = lift_pos(f_pos_w)
    d_pre  = pre_grasp_pos(d_pos_b, d_rot_w)
    d_lift = lift_pos(d_pos_b)

    cprint(f"   pinch_world  = {g_pos_w.round(4)}", "cyan")
    cprint(f"   franka_EE    = {f_pos_w.round(4)}", "cyan")
    cprint(f"   dexmate_R_ee = {d_pos_b.round(4)}", "cyan")

    # Pinch markers
    try:
        from isaacsim.core.api.objects import VisualSphere
        VisualSphere(prim_path="/World/PinchA", name="pinch_a",
                     position=g_pos_w, radius=0.013, color=np.array([1.,0.1,0.1]))
        pb = g_pos_w.copy(); pb[0] += TABLE_X_OFFSET
        VisualSphere(prim_path="/World/PinchB", name="pinch_b",
                     position=pb, radius=0.013, color=np.array([1.,0.1,0.1]))
    except Exception as e:
        cprint(f"[sim2real] VisualSphere skipped: {e}", "yellow")

    # ── 5. Init CuRobo (Franka) ───────────────────────────────────────────────
    franka_mg = init_franka_curobo()

    # ── 6. Init CuRobo (Dexmate) ─────────────────────────────────────────────
    dex_mg = None
    if dexmate_ok:
        try:
            dex_mg = init_dexmate_curobo()
        except Exception as e:
            cprint(f"[sim2real] Dexmate CuRobo failed: {e}", "red")

    # ── Joint helpers ─────────────────────────────────────────────────────────
    HOME_Q = np.array([-0.84,-0.51,-0.37,-1.30,0.65,0.29,0.03])
    def get_dex_q(): return HOME_Q.copy()
    def set_dex_q(q): pass   # placeholder; ArticulationView needs physics init

    # ── 7. Execute ────────────────────────────────────────────────────────────
    def run_phase(label, f_target, d_target):
        cprint(f"\n[sim2real] ── {label} ──", "yellow")
        q7 = franka.get_joint_positions()[:7]
        tf = plan_franka(franka_mg, q7, f_target, f_quat_w, label=f"franka-{label}")
        td = None
        if dex_mg is not None:
            td = plan_dexmate_trajectory(dex_mg, get_dex_q(), d_target, d_quat_w, label=f"dex-{label}")
        _step_both(world, franka, tf, td, set_dex_q, render)

    run_phase("pre-grasp", f_pre,  d_pre)
    run_phase("grasp",     f_pos_w, d_pos_b)

    franka.close_gripper()
    cprint("\n[sim2real] ── HOLDING (inspect alignment) ──", "green")
    for _ in range(args.hold_steps): world.step(render=render)

    run_phase("lift", f_lift, d_lift)
    for _ in range(args.hold_steps): world.step(render=render)

    cprint("\n[sim2real] Done — close window or Ctrl-C to exit.", "green")
    while True: world.step(render=render)


def _step_both(world, franka, traj_f, traj_d, set_dex, render, spw=3):
    import numpy as np
    nf = len(traj_f) if traj_f is not None else 0
    nd = len(traj_d) if traj_d is not None else 0
    n  = max(nf, nd, 1)
    for i in range(n):
        if traj_f is not None:
            q = traj_f[min(i, nf-1)]
            franka.set_joint_positions(np.concatenate([q, franka.get_joint_positions()[7:9]]))
        if traj_d is not None:
            set_dex(traj_d[min(i, nd-1)])
        for _ in range(spw): world.step(render=render)


def _apply_object_physics(stage, obj, mat_suffix="") -> None:
    """Mirror of sim/evaluation/scene_builder._apply_object_physics_materials."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    mat_path = f"/World/PhysicsMaterials/ObjMat{mat_suffix}"
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pm.CreateStaticFrictionAttr(1.0)
    pm.CreateDynamicFrictionAttr(0.8)
    pm.CreateRestitutionAttr(0.0)

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
            binding.Bind(
                UsdShade.Material(mat_prim),
                UsdShade.Tokens.weakerThanDescendants, "physics",
            )


if __name__ == "__main__":
    main()
