"""
cuRobo world helpers: extract object mesh from Isaac stage and sync for planning.
Used by run_grasp_sim.py and run_grasp_sim_test.py (planning collision; does not affect PhysX).
"""
from __future__ import annotations

import numpy as np

MAX_CUROBO_FACES = 12000


def _triangulate_face_indices(face_counts, face_indices):
    """USD faceVertexCounts/Indices → list of triangle [i,j,k]."""
    faces = []
    idx = 0
    for nc in face_counts:
        verts = face_indices[idx : idx + nc]
        idx += nc
        if nc < 3:
            continue
        if nc == 3:
            faces.append(list(verts))
        elif nc == 4:
            faces.append([verts[0], verts[1], verts[2]])
            faces.append([verts[0], verts[2], verts[3]])
        else:
            for i in range(1, nc - 1):
                faces.append([verts[0], verts[i], verts[i + 1]])
    return faces


def extract_mesh_from_stage(stage, rigid_prim_path):
    """
    Collect all UsdGeom.Mesh under rigid_prim_path into one mesh in rigid-root
    local frame (matches RigidObject.get_obj_pos() body frame).
    Returns (vertices Nx3, faces Mx3) or (None, None) on failure.
    """
    from pxr import Gf, Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(rigid_prim_path)
    if not root_prim.IsValid():
        return None, None

    root_xf = UsdGeom.Xformable(root_prim)
    root_world = root_xf.ComputeLocalToWorldTransform(0)
    root_world_inv = root_world.GetInverse()

    all_verts = []
    all_faces = []
    vert_offset = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue

        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if counts is None or indices is None or len(counts) == 0:
            continue

        mesh_xf = UsdGeom.Xformable(prim)
        mesh_world = mesh_xf.ComputeLocalToWorldTransform(0)

        pts_root = []
        for p in points:
            pw = mesh_world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            pr = root_world_inv.Transform(pw)
            pts_root.append([pr[0], pr[1], pr[2]])

        tris = _triangulate_face_indices(list(counts), list(indices))
        if not tris:
            continue

        all_verts.extend(pts_root)
        for tri in tris:
            all_faces.append([tri[0] + vert_offset, tri[1] + vert_offset, tri[2] + vert_offset])
        vert_offset += len(pts_root)

    if not all_verts:
        return None, None

    vertices = np.asarray(all_verts, dtype=np.float64)
    faces = np.asarray(all_faces, dtype=np.int32)
    return vertices, faces


def coarsen_mesh(vertices, faces, max_faces=MAX_CUROBO_FACES):
    """Reduce face count for cuRobo; returns (vertices, faces)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) <= max_faces:
        return mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32)

    try:
        simplified = mesh.simplify_quadric_decimation(max_faces)
        return simplified.vertices.astype(np.float64), simplified.faces.astype(np.int32)
    except Exception:
        hull = mesh.convex_hull
        return hull.vertices.astype(np.float64), hull.faces.astype(np.int32)


def prepare_curobo_mesh(stage, rigid_prim_path):
    """Extract + coarsen; returns dict with vertices, faces, n_faces_raw or None."""
    verts, faces = extract_mesh_from_stage(stage, rigid_prim_path)
    if verts is None:
        return None
    n_raw = len(faces)
    verts, faces = coarsen_mesh(verts, faces)
    return {
        "vertices": verts,
        "faces": faces,
        "n_faces_raw": n_raw,
        "n_faces": len(faces),
    }


def object_pose_robot_frame(pos_world, quat_wxyz_world, T_robot_world):
    """World pose → robot-base pose as cuRobo [x,y,z,qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation

    pos_r = (T_robot_world @ np.append(pos_world, 1.0))[:3]
    R_w = Rotation.from_quat([
        quat_wxyz_world[1], quat_wxyz_world[2],
        quat_wxyz_world[3], quat_wxyz_world[0],
    ])
    R_rw = Rotation.from_matrix(T_robot_world[:3, :3])
    R_r = R_rw * R_w
    q = R_r.as_quat()
    quat_r = np.array([q[3], q[0], q[1], q[2]])
    return [
        float(pos_r[0]), float(pos_r[1]), float(pos_r[2]),
        float(quat_r[0]), float(quat_r[1]), float(quat_r[2]), float(quat_r[3]),
    ]


def build_world_config_dict(
    table_pos_r,
    ground_pos_r,
    table_dims,
    mesh_vertices=None,
    mesh_faces=None,
    mesh_pose_robot=None,
):
    """World dict for MotionGenConfig.load_from_robot_config / WorldConfig.from_dict."""
    cfg = {
        "cuboid": {
            "table": {
                "dims": list(table_dims),
                "pose": [*table_pos_r.tolist(), 1, 0, 0, 0],
            },
            "ground": {
                "dims": [5.0, 5.0, 0.01],
                "pose": [*ground_pos_r.tolist(), 1, 0, 0, 0],
            },
        },
    }
    if mesh_vertices is not None and mesh_faces is not None and mesh_pose_robot is not None:
        faces_list = mesh_faces.tolist() if hasattr(mesh_faces, "tolist") else mesh_faces
        verts_list = mesh_vertices.tolist() if hasattr(mesh_vertices, "tolist") else mesh_vertices
        cfg["mesh"] = {
            "grasp_object": {
                "vertices": verts_list,
                "faces": faces_list,
                "pose": list(mesh_pose_robot),
            },
        }
    return cfg


def clear_curobo_object_mesh(motion_gen):
    """Remove grasp-object mesh from cuRobo checker.

    cuRobo load_collision_model() does not clear mesh slots when the new world has
    zero meshes; call this before table+ground-only plans (final / lift).
    """
    wc = getattr(motion_gen, "world_coll_checker", None)
    if wc is None:
        return

    if getattr(wc, "_mesh_tensor_list", None) is not None:
        wc._mesh_tensor_list[2][:] = 0
    if getattr(wc, "_env_n_mesh", None) is not None:
        wc._env_n_mesh[:] = 0
    if getattr(wc, "collision_types", None) is not None:
        wc.collision_types["mesh"] = False


def sync_curobo_world(
    motion_gen,
    scene,
    table_pos_r,
    ground_pos_r,
    table_dims,
    T_robot_world,
    include_object_mesh=True,
):
    """Update cuRobo collision world (planning only).

    include_object_mesh=False → table + ground only (e.g. final approach / last mile).
    """
    from curobo.geom.types import WorldConfig

    if not include_object_mesh:
        clear_curobo_object_mesh(motion_gen)

    mesh_verts = mesh_faces = mesh_pose = None
    if include_object_mesh and scene.get("curobo_mesh_vertices") is not None:
        pos_w, quat_wxyz = scene["obj"].get_obj_pos()
        mesh_pose = object_pose_robot_frame(pos_w, quat_wxyz, T_robot_world)
        mesh_verts = scene["curobo_mesh_vertices"]
        mesh_faces = scene["curobo_mesh_faces"]

    world_dict = build_world_config_dict(
        table_pos_r,
        ground_pos_r,
        table_dims,
        mesh_vertices=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_pose_robot=mesh_pose,
    )
    motion_gen.update_world(WorldConfig.from_dict(world_dict))