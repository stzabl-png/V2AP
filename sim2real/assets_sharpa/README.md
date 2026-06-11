# sim2real/assets_sharpa/README.md

## Sharpa Wave USD Asset

Place `right_sharpa_wave_with_flange.usda` here.

### How to obtain

```bash
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
cp sharpa-urdf-usd-xml/src/right_sharpa_wave/right_sharpa_wave_with_flange.usda \
   sim2real/assets_sharpa/
```

### Variant explanation

| Variant | Use case |
|---------|----------|
| `right_sharpa_wave_with_flange.usda` | **Use this** – flange-level root, attaches to arm EE directly |
| `right_sharpa_wave_with_wrist.usda` | Includes wrist geometry, for standalone setups |

The `with_flange` variant has its root at the mechanical mounting interface,
which maps to the Dexmate R_ee frame at +5 cm offset (handled in retarget_utils.py).
