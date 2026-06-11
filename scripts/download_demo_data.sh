#!/bin/bash
# Download demo data for V2AP quick-start.
# Includes 3 example object meshes and pre-computed contact priors.
set -e

echo "=== Downloading V2AP demo data from HuggingFace ==="

pip install -q huggingface_hub

python3 - <<'EOF'
from huggingface_hub import snapshot_download

# Download demo objects and example contact priors
snapshot_download(
    repo_id="UCBProject/ProcessedData",
    repo_type="dataset",
    allow_patterns=["demo/*"],
    local_dir="examples/demo",
)
print("✅ Demo data downloaded to examples/demo/")
EOF

echo ""
echo "=== Demo data ready ==="
echo "Run: python inference/grasp_pose.py --mesh examples/demo/chips_can/mesh.obj"
