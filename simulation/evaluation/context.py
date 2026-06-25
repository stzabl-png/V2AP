"""Runtime context for IsaacSim evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.specs import SceneSpec


@dataclass
class SimEvaluationContext:
    """IsaacSim runtime handles plus the serializable scene spec."""

    spec: SceneSpec
    world: Any
    franka: Any
    obj: Any
    render: bool
    object_placement: dict[str, Any]
    curobo_mesh_vertices: Any = None
    curobo_mesh_faces: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stage(self):
        return self.world.stage

    def get(self, key: str, default=None):
        """Compatibility helper for code migrated from dict-based scenes."""
        return getattr(self, key, self.metadata.get(key, default))

    def as_legacy_dict(self) -> dict[str, Any]:
        """Dict view for shared helpers that still expect the old scene shape."""
        out = {
            "world": self.world,
            "franka": self.franka,
            "obj": self.obj,
            "obj_id": self.spec.obj_id,
            "object_placement": self.object_placement,
            "sim_z_yaw_deg": self.spec.sim_z_yaw_deg,
        }
        if self.curobo_mesh_vertices is not None:
            out["curobo_mesh_vertices"] = self.curobo_mesh_vertices
            out["curobo_mesh_faces"] = self.curobo_mesh_faces
        out.update(self.metadata)
        return out

    def update_from_legacy_dict(self, scene: dict[str, Any]) -> None:
        """Persist values written by cuRobo helpers to the context metadata."""
        for key, value in scene.items():
            if key in {"world", "franka", "obj"}:
                continue
            if key == "object_placement":
                self.object_placement = value
            elif key == "curobo_mesh_vertices":
                self.curobo_mesh_vertices = value
            elif key == "curobo_mesh_faces":
                self.curobo_mesh_faces = value
            else:
                self.metadata[key] = value

