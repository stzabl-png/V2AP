"""IsaacSim scene construction for evaluation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from termcolor import cprint

from evaluation.specs import SceneSpec
from sim.evaluation.context import SimEvaluationContext

SIM_DIR = Path(__file__).resolve().parents[1]
PROJ_DIR = SIM_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
from isaacsim.core.utils.prims import delete_prim  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

from curobo_world import prepare_curobo_mesh  # noqa: E402
from env_config.rigid.RigidObject import RigidObject  # noqa: E402
from env_config.robot.Franka import Franka  # noqa: E402


ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]
TABLE_POSITION = [0.0, 1.0, 0.75]
TABLE_ORIENTATION = [0.0, 0.0, 0.0]
TABLE_SCALE = [2.0, 2.0, 0.1]
TABLE_TOP_Z = 0.80
OBJECT_POSITION = [0.0, 0.55, TABLE_TOP_Z]
OBJECT_ORIENTATION = [0.0, 0.0, 0.0]

OVERRIDE_FILE = SIM_DIR / "object_rotation_overrides.json"
try:
    with open(OVERRIDE_FILE, encoding="utf-8") as f:
        _raw_overrides = json.load(f)
    OBJECT_ROTATION_OVERRIDES = {
        k: v for k, v in _raw_overrides.items() if not str(k).startswith("_")
    }
except Exception:
    OBJECT_ROTATION_OVERRIDES = {}


def find_obj_usd_path(obj_id: str) -> str | None:
    """Search output/obj_usd then sim/assets for an object USD."""
    obj_usd_root = PROJ_DIR / "output" / "obj_usd"
    datasets_order = ["oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3", "unseen", "egodex"]
    usd_search_paths = (
        [obj_usd_root / ds / f"{obj_id}.usd" for ds in datasets_order]
        + [SIM_DIR / "assets" / f"{obj_id}.usd"]
    )
    return next((str(p) for p in usd_search_paths if p.exists()), None)


def resolve_object_placement(
    obj_id: str,
    object_scale: float,
    sim_z_yaw_deg: float = 0.0,
    obj_xy_offset: tuple[float, float] | list[float] | None = None,
) -> dict:
    """Resolve table placement using the same conventions as run_grasp_sim.py."""
    override = OBJECT_ROTATION_OVERRIDES.get(obj_id, None)
    obj_orientation = list(OBJECT_ORIENTATION)

    usd_path = find_obj_usd_path(obj_id)
    if usd_path is None:
        raise FileNotFoundError(f"USD not found for placement resolve: {obj_id}")

    meta_path = usd_path.replace(".usd", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        obj_z_offset = float(meta.get("z_offset_m", 0.075 * object_scale))
    elif isinstance(override, dict) and "z_offset" in override:
        obj_z_offset = float(override["z_offset"])
    else:
        obj_z_offset = 0.075 * object_scale

    if isinstance(override, dict) and "rotation" in override:
        obj_orientation = list(override["rotation"])

    yaw = float(sim_z_yaw_deg)
    if abs(yaw) > 1e-9:
        obj_orientation[2] = float(obj_orientation[2]) + yaw

    dx, dy = 0.0, 0.0
    if obj_xy_offset is not None:
        off = np.asarray(obj_xy_offset, dtype=np.float64).reshape(2)
        dx, dy = float(off[0]), float(off[1])
    obj_pos = list(OBJECT_POSITION)
    obj_pos[0] += dx
    obj_pos[1] += dy
    obj_pos[2] += obj_z_offset
    return {
        "pos": obj_pos,
        "ori": obj_orientation,
        "z_offset": obj_z_offset,
        "obj_xy_offset": [dx, dy],
        "sim_z_yaw_deg": yaw,
        "usd_path": usd_path,
    }


def _euler_xyz_deg_to_wxyz(euler_deg: list[float]) -> list[float]:
    q_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def build_scene_spec(
    *,
    obj_id: str,
    episode_id: str,
    dataset: str | None,
    object_scale: float,
    sim_z_yaw_deg: float,
    seed: int,
    candidate_hdf5: str | None = None,
    obj_xy_offset: list[float] | tuple[float, float] | None = None,
    random_obj_xy: bool = False,
    obj_xy_jitter_m: float = 0.05,
    eval_seed: int = 42,
) -> SceneSpec:
    from evaluation.placement import resolve_obj_xy_offset

    dx, dy = resolve_obj_xy_offset(
        random_obj_xy=bool(random_obj_xy),
        obj_xy_jitter_m=float(obj_xy_jitter_m),
        obj_id=obj_id,
        trial=int(seed),
        sim_z_yaw_deg=float(sim_z_yaw_deg),
        obj_xy_offset=obj_xy_offset,
        eval_seed=int(eval_seed),
    )
    placement = resolve_object_placement(
        obj_id,
        object_scale,
        sim_z_yaw_deg,
        obj_xy_offset=(dx, dy),
    )
    return SceneSpec(
        episode_id=episode_id,
        obj_id=obj_id,
        dataset=dataset,
        object_scale=float(object_scale),
        usd_path=placement["usd_path"],
        object_position_world=[float(x) for x in placement["pos"]],
        object_orientation_euler_deg=[float(x) for x in placement["ori"]],
        object_quat_wxyz=_euler_xyz_deg_to_wxyz(placement["ori"]),
        sim_z_yaw_deg=float(sim_z_yaw_deg),
        obj_xy_offset=[float(dx), float(dy)],
        seed=int(seed),
        robot_position=list(ROBOT_POSITION),
        robot_orientation_euler_deg=list(ROBOT_ORIENTATION),
        table_position=list(TABLE_POSITION),
        table_orientation_euler_deg=list(TABLE_ORIENTATION),
        table_scale=list(TABLE_SCALE),
        metadata={
            "candidate_hdf5": os.path.abspath(candidate_hdf5) if candidate_hdf5 else "",
            "z_offset": float(placement["z_offset"]),
            "obj_xy_offset": [float(dx), float(dy)],
            "random_obj_xy": bool(random_obj_xy),
            "obj_xy_jitter_m": float(obj_xy_jitter_m),
            "eval_seed": int(eval_seed),
        },
    )


def _delete_rigid_prims() -> None:
    for i in range(10):
        delete_prim(f"/World/Rigid/rigid_{i}")
    delete_prim("/World/Rigid/rigid")


def _invalidate_rigid_physics_view(rigid) -> None:
    if rigid is None:
        return
    try:
        if hasattr(rigid, "_invalidate_physics_handle_callback"):
            rigid._invalidate_physics_handle_callback(None)
    except Exception:
        pass
    try:
        if hasattr(rigid, "_physics_view"):
            rigid._physics_view = None
    except Exception:
        pass


def _release_scene_rigid_object(scene: SimEvaluationContext) -> None:
    old_obj = getattr(scene, "obj", None)
    if old_obj is None:
        return
    _invalidate_rigid_physics_view(getattr(old_obj, "rigid", None))
    scene.obj = None


def _apply_object_physics_materials(stage, obj) -> None:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    material_path = "/World/PhysicsMaterials/BottleMaterial"
    UsdShade.Material.Define(stage, material_path)
    mat_prim = stage.GetPrimAtPath(material_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
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
            binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")


def _ensure_finger_friction_materials(stage) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    finger_material_path = "/World/PhysicsMaterials/FingerMaterial"
    UsdShade.Material.Define(stage, finger_material_path)
    finger_mat_prim = stage.GetPrimAtPath(finger_material_path)
    finger_physics_mat = UsdPhysics.MaterialAPI.Apply(finger_mat_prim)
    finger_physics_mat.CreateStaticFrictionAttr(1.2)
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
                        "physics",
                    )


def _reset_franka_home(scene: SimEvaluationContext) -> None:
    home_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
    scene.franka.set_joint_positions(home_joints)
    scene.franka.open_gripper()
    for _ in range(150):
        scene.world.step(render=scene.render)


def _spawn_rigid_object(world, spec: SceneSpec, *, render: bool):
    placement = resolve_object_placement(
        spec.obj_id,
        spec.object_scale,
        spec.sim_z_yaw_deg,
        obj_xy_offset=spec.obj_xy_offset,
    )
    obj = RigidObject(
        world,
        usd_path=placement["usd_path"],
        pos=np.array(placement["pos"]),
        ori=np.array(placement["ori"]),
        scale=np.array([spec.object_scale] * 3),
        mass=0.05,
    )
    _apply_object_physics_materials(world.stage, obj)
    for _ in range(100):
        world.step(render=render)
    mesh_info = prepare_curobo_mesh(world.stage, obj.rigid_prim_path)
    return obj, placement, mesh_info


def _update_context_object(
    scene: SimEvaluationContext,
    spec: SceneSpec,
    obj,
    placement: dict,
    mesh_info: dict | None,
) -> None:
    scene.spec = spec
    scene.obj = obj
    scene.object_placement = placement
    scene.metadata["obj_id"] = spec.obj_id
    scene.metadata["sim_z_yaw_deg"] = spec.sim_z_yaw_deg
    if mesh_info is not None:
        scene.curobo_mesh_vertices = mesh_info["vertices"]
        scene.curobo_mesh_faces = mesh_info["faces"]
    else:
        scene.curobo_mesh_vertices = None
        scene.curobo_mesh_faces = None


def reset_scene_pose(scene: SimEvaluationContext, spec: SceneSpec) -> None:
    """Reset same object to a new placement/yaw and home the robot."""
    placement = resolve_object_placement(
        spec.obj_id,
        spec.object_scale,
        spec.sim_z_yaw_deg,
        obj_xy_offset=spec.obj_xy_offset,
    )
    scene.spec = spec
    scene.object_placement = placement
    scene.obj.set_obj_pose(
        np.array(placement["pos"], dtype=np.float64),
        ori=np.array(placement["ori"], dtype=np.float64),
    )
    try:
        scene.obj.rigid.set_linear_velocity(np.zeros(3))
        scene.obj.rigid.set_angular_velocity(np.zeros(3))
    except Exception:
        pass
    _reset_franka_home(scene)


def swap_scene_object(scene: SimEvaluationContext, spec: SceneSpec) -> None:
    """Replace the object USD while keeping World, table, and Franka alive."""
    _release_scene_rigid_object(scene)
    _delete_rigid_prims()
    scene.world.reset()
    for _ in range(20):
        scene.world.step(render=scene.render)
    obj, placement, mesh_info = _spawn_rigid_object(scene.world, spec, render=scene.render)
    _update_context_object(scene, spec, obj, placement, mesh_info)
    _reset_franka_home(scene)


def setup_scene(spec: SceneSpec, *, render: bool) -> SimEvaluationContext:
    """Build an IsaacSim scene for one evaluation episode."""
    world = World(backend="numpy")
    physics = world.get_physics_context()
    physics.enable_ccd(True)
    physics.enable_gpu_dynamics(True)
    physics.set_broadphase_type("gpu")
    physics.enable_stablization(True)
    physics.set_solver_type("TGS")

    set_camera_view(
        eye=[0.0, 4.5, 3.5],
        target=[0.0, 0.0, 0.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    delete_prim("/Replicator/DomeLight_Xform")
    rep.create.light(position=[0, 0, 0], light_type="dome")

    env_usd = SIM_DIR / "assets_scene" / "Collected_default_environment" / "default_environment.usd"
    delete_prim("/World/Environment")
    if env_usd.exists():
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(usd_path=str(env_usd), prim_path="/World/Environment")

    delete_prim("/World/Ground")
    FixedCuboid(
        prim_path="/World/Ground",
        name="ground",
        position=np.array([0.0, 0.0, -0.025]),
        scale=np.array([20.0, 20.0, 0.05]),
        size=1.0,
        visible=False,
    )

    delete_prim("/World/Table")
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=TABLE_POSITION,
        orientation=euler_angles_to_quat(np.array(TABLE_ORIENTATION), degrees=True),
        scale=TABLE_SCALE,
        size=1.0,
        visible=True,
    )

    delete_prim("/World/Franka")
    franka = Franka(world, np.array(ROBOT_POSITION), np.array(ROBOT_ORIENTATION))
    world.reset()
    for _ in range(50):
        world.step(render=render)
    franka.open_gripper()
    for _ in range(10):
        world.step(render=render)
    _ensure_finger_friction_materials(world.stage)

    _delete_rigid_prims()
    obj, placement, mesh_info = _spawn_rigid_object(world, spec, render=render)
    mesh_vertices = mesh_faces = None
    if mesh_info is not None:
        mesh_vertices = mesh_info["vertices"]
        mesh_faces = mesh_info["faces"]
        cprint(
            f"   cuRobo object mesh: {mesh_info['n_faces']} faces"
            f" (raw {mesh_info['n_faces_raw']})",
            "green",
        )
    else:
        cprint("   cuRobo object mesh extraction failed; using table+ground only", "yellow")

    cprint(f"Scene ready for {spec.obj_id}", "green")
    return SimEvaluationContext(
        spec=spec,
        world=world,
        franka=franka,
        obj=obj,
        render=render,
        object_placement=placement,
        curobo_mesh_vertices=mesh_vertices,
        curobo_mesh_faces=mesh_faces,
        metadata={},
    )

