#!/bin/bash
# Install Dexmate dexcontrol (AGPL-3.0 / commercial dual license).
# See: https://github.com/dexmate-robotics/dexcontrol
# Users must comply with the AGPL-3.0 license or obtain a commercial license from Dexmate.
set -e

echo "=== Installing Dexmate dexcontrol ==="
echo "⚠️  dexcontrol is dual-licensed: AGPL-3.0 (open source) / commercial."
echo "    Redistribution under AGPL-3.0 requires making your source available."
echo "    Contact Dexmate for commercial license terms."
echo ""

# Clone from official Dexmate repo (adjust URL if private)
# git clone https://github.com/dexmate-robotics/dexcontrol.git third_party/dexcontrol
# pip install -e third_party/dexcontrol

# Or install from your lab's provided package:
# pip install -e /path/to/dexcontrol

echo "Please install dexcontrol from your Dexmate-provided package."
echo "Set DEXCONTROL_DIR in your environment after installation."
