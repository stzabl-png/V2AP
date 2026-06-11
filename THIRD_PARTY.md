# Third-Party Dependencies

This repository contains our own pipeline code for V2AP.
The following external components are dependencies but are **not included** in this repository.
Users must obtain and install them separately, in compliance with each project's license.

---

## Included as Submodules (optional, see install scripts)

| Component | Used for | Location | License |
|-----------|----------|----------|---------|
| [MegaSAM](https://github.com/mega-sam/mega-sam) | Egocentric monocular depth + camera intrinsics | `third_party/mega-sam/` | [Apache-2.0](https://github.com/mega-sam/mega-sam/blob/main/LICENSE) (code) / CC-BY 4.0 (materials) |
| [Depth Pro](https://github.com/apple/ml-depth-pro) | Third-person monocular depth + camera intrinsics | `third_party/ml-depth-pro/` | [Apple Sample Code License](https://github.com/apple/ml-depth-pro/blob/main/LICENSE) |

---

## External Dependencies (not included — must install separately)

| Component | Used for | License | Install |
|-----------|----------|---------|---------|
| [HaWoR](https://github.com/ThunderVVV/HaWoR) | Egocentric hand reconstruction (MANO mesh) | CC-BY-NC-ND (models) | `bash scripts/install_hawor.sh` |
| [HaPTIC](https://github.com/judyye/haptic) | Third-person hand reconstruction (MANO mesh) | See repo | `bash scripts/install_haptic.sh` |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | Object 6D pose estimation | [NVIDIA non-commercial research](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE) | `bash scripts/install_foundationpose.sh` |
| [SAM3D Objects](https://github.com/facebookresearch/sam-3d-objects) | Object mesh reconstruction | [Meta SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) | `bash scripts/install_sam3d.sh` |
| [Isaac Sim](https://developer.nvidia.com/isaac-sim) | Physics-based grasp validation | NVIDIA EULA | [Install separately](https://docs.isaacsim.omniverse.nvidia.com/) |

---

## MANO (Required by HaWoR and HaPTIC — never redistributed)

> ⚠️ **MANO model files are NOT included and must NEVER be uploaded to any repository.**

MANO is licensed for **non-commercial scientific research** only and requires individual registration:
- Download from: https://mano.is.tue.mpg.de/
- Required files: `MANO_RIGHT.pkl`, `MANO_LEFT.pkl`, `MANO_UV_right.obj`, `MANO_UV_left.obj`
- Placement instructions: see `docs/INSTALL.md`

---

## License Notes

Our code in the `a2g/`, `model/`, `inference/`, `evaluation/`, `sim/`, `data/`, and `tools/` directories
is released under the [MIT License](./LICENSE).

The above third-party components are governed solely by their own respective licenses.
We provide wrapper scripts (`a2g/depth/`, `a2g/hand/`, `a2g/pose/`) that call these external tools,
but we do not include, modify, or redistribute third-party code or model weights.
