# sim2real/assets_dexmate/README.md

## Dexmate Vega USD Asset

Place `vega_1.usd` here.

### How to obtain

```bash
# Install Python package (includes URDF, download USD from Releases)
pip install dexmate_urdf

# Or clone + grab USD from the GitHub Release (v0.8.3+)
# https://github.com/dexmate-ai/dexmate-urdf/releases
# File: vega_1_usd.zip  → extract vega_1.usd into this folder
```

### Variant to use

`vega_1.usd` — full humanoid, base variant (no built-in hands).
The Sharpa Wave hand will be attached at runtime as a child of `/R_ee`.
