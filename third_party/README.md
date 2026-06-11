# Third-Party Dependencies

This directory contains external dependencies managed as git submodules or install scripts.

## Included (git submodules)

| Component | Path | License | Notes |
|-----------|------|---------|-------|
| MegaSAM | `mega-sam/` | Apache-2.0 / CC-BY 4.0 | Egocentric depth + intrinsics |
| Depth Pro | `ml-depth-pro/` | Apple Sample Code License | Third-person depth + intrinsics |

Initialize submodules:
```bash
git submodule update --init --recursive
```

## Not Included (install separately)

Use the provided install scripts:

```bash
bash scripts/install_hawor.sh          # HaWoR: CC-BY-NC-ND
bash scripts/install_haptic.sh         # HaPTIC: see repo license
bash scripts/install_foundationpose.sh # FoundationPose: NVIDIA non-commercial
```

See `THIRD_PARTY.md` in the repo root for complete license information.
