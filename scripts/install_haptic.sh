#!/bin/bash
# Install HaPTIC for third-person hand reconstruction.
# License: See https://github.com/judyye/haptic — MANO required (non-commercial).
# Users must download MANO from https://mano.is.tue.mpg.de/ and comply with their license.
set -e

HAPTIC_COMMIT="f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df"
INSTALL_DIR="${1:-third_party/haptic}"

echo "=== Installing HaPTIC ==="
echo "⚠️  HaPTIC requires MANO. Download from https://mano.is.tue.mpg.de/ (non-commercial use only)"
echo ""

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists. Skipping clone."
else
    git clone https://github.com/judyye/haptic.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$HAPTIC_COMMIT"
    cd -
fi

echo ""
echo "=== Applying HaPTIC intrinsics patch ==="
git apply patches/haptic-intrinsics-fix.patch

echo ""
echo "=== HaPTIC installed at $INSTALL_DIR ==="
echo ""
echo "Next steps:"
echo "  1. Download MANO_RIGHT.pkl, MANO_LEFT.pkl, MANO_UV_right.obj, MANO_UV_left.obj"
echo "     from https://mano.is.tue.mpg.de/"
echo "  2. Place at: $INSTALL_DIR/assets/mano/"
echo "  3. Run: cd $INSTALL_DIR && bash scripts/one_click.sh"
echo "  4. Set HAPTIC_DIR=$INSTALL_DIR in your .env file"
