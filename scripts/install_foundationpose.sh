#!/bin/bash
# Install FoundationPose for object 6D pose estimation.
# License: NVIDIA non-commercial research only.
# See: https://github.com/NVlabs/FoundationPose/blob/main/LICENSE
set -e

FP_COMMIT="25e225a"
INSTALL_DIR="${1:-third_party/FoundationPose}"

echo "=== Installing FoundationPose ==="
echo "⚠️  FoundationPose is licensed for NON-COMMERCIAL RESEARCH OR EVALUATION PURPOSES ONLY."
echo "    See: https://github.com/NVlabs/FoundationPose/blob/main/LICENSE"
echo ""
read -p "Do you accept the FoundationPose license terms? (yes/no): " ACCEPT
if [ "$ACCEPT" != "yes" ]; then
    echo "Installation cancelled."
    exit 1
fi

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists. Skipping clone."
else
    git clone https://github.com/NVlabs/FoundationPose.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$FP_COMMIT"
fi

cd "$INSTALL_DIR"

# Install Python dependencies
pip install -r requirements.txt

# Build CUDA extensions (requires cmake, ninja, Eigen3)
echo "Building CUDA extensions (this may take a few minutes)..."
bash build_all_conda.sh

cd -

echo ""
echo "=== FoundationPose installed at $INSTALL_DIR (commit: $FP_COMMIT) ==="
echo ""
echo "Next steps:"
echo "  1. Download weights: python setup_weights.py --tool fp"
echo "  2. Set FP_ROOT=$INSTALL_DIR in your .env file"
