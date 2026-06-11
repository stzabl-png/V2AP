#!/bin/bash
# Install HaWoR for egocentric hand reconstruction.
# License: CC-BY-NC-ND (models). Users must comply with the HaWoR and MANO licenses.
# MANO models must be downloaded separately from https://mano.is.tue.mpg.de/
set -e

HAWOR_COMMIT="66c7d4108d58a716deccd192cb7645170cdc7bd7"
INSTALL_DIR="${1:-third_party/hawor}"

echo "=== Installing HaWoR ==="
echo "⚠️  HaWoR model weights are CC-BY-NC-ND. Non-commercial use only."
echo "⚠️  MANO models must be downloaded separately from https://mano.is.tue.mpg.de/"
echo ""

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists. Skipping clone."
else
    git clone https://github.com/ThunderVVV/HaWoR.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$HAWOR_COMMIT"
    cd -
fi

echo ""
echo "=== HaWoR installed at $INSTALL_DIR (commit: $HAWOR_COMMIT) ==="
echo ""
echo "Next steps:"
echo "  1. Download MANO_RIGHT.pkl and MANO_LEFT.pkl from https://mano.is.tue.mpg.de/"
echo "  2. Place them at: $INSTALL_DIR/_DATA/data/mano/ and _DATA/data_left/mano_left/"
echo "  3. Download HaWoR weights per the official README"
echo "  4. Set HAWOR_DIR=$INSTALL_DIR in your .env file"
