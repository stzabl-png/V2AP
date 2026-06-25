"""
Mesh Provider — abstract interface for object mesh acquisition.

Provides ARCTIC GT meshes now, with SAM3D placeholder for future video-based reconstruction.
All providers can export meshes to the unified grasp_collection for sim testing.

Usage:
    from data.mesh_provider import get_provider
    provider = get_provider("arctic")
    mesh = provider.get_mesh("box")
    provider.export_to_grasp_collection("box")  # → grasp_collection/box.obj
"""

import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MeshProvider:
    """Abstract interface for object mesh acquisition."""

    def get_mesh(self, object_name):
        """Returns trimesh.Trimesh object, or None if not found."""
        raise NotImplementedError

    def list_objects(self):
        """Returns list of available object names."""
        raise NotImplementedError

    def get_mesh_path(self, object_name):
        """Returns path to the original mesh file, or None."""
        raise NotImplementedError

    def export_to_grasp_collection(self, object_name):
        """Copy mesh to unified grasp_collection dir for sim grasp testing.

        Returns path to the exported file, or None if mesh not found.
        """
        src_path = self.get_mesh_path(object_name)
        if src_path is None:
            return None

        os.makedirs(config.GRASP_MESH_DIR, exist_ok=True)
        ext = os.path.splitext(src_path)[1]
        dst_path = os.path.join(config.GRASP_MESH_DIR, f"{object_name}{ext}")
        shutil.copy2(src_path, dst_path)
        return dst_path

    def export_all(self):
        """Export all available objects to grasp_collection."""
        exported = []
        for obj_name in self.list_objects():
            path = self.export_to_grasp_collection(obj_name)
            if path:
                exported.append(obj_name)
        return exported


class ARCTICMeshProvider(MeshProvider):
    """Uses ARCTIC GT mesh templates from meta/object_vtemplates/."""

    def __init__(self):
        self.mesh_dir = os.path.join(config.ARCTIC_ROOT, "meta", "object_vtemplates")

    def get_mesh(self, object_name):
        import trimesh
        path = self.get_mesh_path(object_name)
        if path:
            return trimesh.load(path, force='mesh')
        return None

    def get_mesh_path(self, object_name):
        obj_dir = os.path.join(self.mesh_dir, object_name)
        if not os.path.isdir(obj_dir):
            return None
        # Prefer 'mesh.obj' (complete mesh) over partial meshes like bottom.obj/top.obj
        candidates = sorted(os.listdir(obj_dir))
        for preferred in ('mesh.obj', 'mesh_tex.obj'):
            if preferred in candidates:
                return os.path.join(obj_dir, preferred)
        for fn in candidates:
            if fn.endswith(('.obj', '.ply', '.stl')):
                return os.path.join(obj_dir, fn)
        return None

    def list_objects(self):
        if not os.path.isdir(self.mesh_dir):
            return []
        return sorted([d for d in os.listdir(self.mesh_dir)
                       if os.path.isdir(os.path.join(self.mesh_dir, d))])


class SAM3DMeshProvider(MeshProvider):
    """Placeholder — will reconstruct mesh from video using SAM3D.

    TODO: Implement when SAM3D is integrated.
    Expects SAM3D to produce meshes in config.SAM3D_CACHE/<object_name>.obj
    """

    def __init__(self):
        self.cache_dir = config.SAM3D_CACHE
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_mesh(self, object_name):
        import trimesh
        path = self.get_mesh_path(object_name)
        if path:
            return trimesh.load(path, force='mesh')
        print(f"⚠️  SAM3D mesh not found for '{object_name}'. "
              f"Place reconstructed mesh at: {os.path.join(self.cache_dir, object_name + '.obj')}")
        return None

    def get_mesh_path(self, object_name):
        mesh_path = os.path.join(self.cache_dir, f"{object_name}.obj")
        if os.path.exists(mesh_path):
            return mesh_path
        return None

    def list_objects(self):
        if not os.path.isdir(self.cache_dir):
            return []
        return sorted([os.path.splitext(f)[0] for f in os.listdir(self.cache_dir)
                       if f.endswith('.obj')])


def get_provider(name="arctic"):
    """Factory function to get a mesh provider by name."""
    providers = {
        "arctic": ARCTICMeshProvider,
        "sam3d": SAM3DMeshProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(providers.keys())}")
    return providers[name]()

