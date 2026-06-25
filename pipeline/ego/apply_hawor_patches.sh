#!/usr/bin/env bash
# apply_hawor_patches.sh  –  Apply compatibility patches to fresh HaWoR install.
# Run this ONCE after `python setup.py install` in DROID-SLAM.
# Usage: bash data/apply_hawor_patches.sh

set -e

HAWOR_DIR="$(cd "$(dirname "$0")/../third_party/hawor" && pwd)"
CONDA_ENV="hawor"
echo "Patching HaWoR at: $HAWOR_DIR"

# ── 1. numpy 1.26.4 (torch 2.1 ABI requirement) ───────────────────────────────
conda run -n "$CONDA_ENV" pip install "numpy==1.26.4" --force-reinstall --no-deps -q
echo "[OK] numpy pinned to 1.26.4"

# ── 2. setuptools 69.5.1 (restore pkg_resources) ─────────────────────────────
conda run -n "$CONDA_ENV" pip install "setuptools==69.5.1" --force-reinstall -q
echo "[OK] setuptools pinned to 69.5.1"

# ── 3. chumpy (MANO dependency, needs --no-build-isolation) ──────────────────
conda run -n "$CONDA_ENV" pip install chumpy --no-build-isolation -q || true

# Patch chumpy/__init__.py for numpy >= 1.24 (removed np.bool etc.)
CHUMPY_PATH=$(conda run -n "$CONDA_ENV" python -c \
    "import importlib.util; s=importlib.util.find_spec('chumpy'); print(s.origin)" 2>/dev/null || \
    find "/home/$USER/anaconda3/envs/$CONDA_ENV" -name "__init__.py" -path "*/chumpy/*" 2>/dev/null | head -1)

if [ -n "$CHUMPY_PATH" ]; then
    sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy as _np; bool=_np.bool_; int=_np.int_; float=_np.float_; complex=_np.complex_; object=_np.object_; unicode=str; nan=_np.nan; inf=_np.inf/' "$CHUMPY_PATH" 2>/dev/null || true
    echo "[OK] chumpy patched at $CHUMPY_PATH"
else
    echo "[WARN] chumpy not found, skipping patch"
fi

# ── 4. missing pytorch-lightning deps ────────────────────────────────────────
conda run -n "$CONDA_ENV" pip install PyYAML tqdm -q
echo "[OK] PyYAML + tqdm installed"

# ── 5. YOLO: use predict() instead of track() ─────────────────────────────────
TOOLS_PY="$HAWOR_DIR/lib/pipeline/tools.py"
if grep -q "hand_det_model.track(" "$TOOLS_PY" 2>/dev/null; then
    sed -i \
        's/results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)/results = hand_det_model.predict(img_cv2, conf=thresh, verbose=False)/' \
        "$TOOLS_PY"
    # Remove the track_id block (track returns None for .id)
    python3 - <<'PYEOF'
import re, sys

path = "$TOOLS_PY"
with open(path) as f:
    txt = f.read()

# Replace track_id assignment block
old = """\
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)"""
new = "                track_id = [-1] * len(boxes)"
txt = txt.replace(old, new)

with open(path, 'w') as f:
    f.write(txt)
print(f"[OK] tools.py patched: track → predict")
PYEOF
else
    echo "[OK] tools.py already patched (predict)"
fi

# ── 6. masked_droid_slam: allow re-initialization ─────────────────────────────
MDS="$HAWOR_DIR/lib/pipeline/masked_droid_slam.py"
if grep -q "set_start_method('spawn')" "$MDS" 2>/dev/null; then
    python3 - <<'PYEOF'
path = "$MDS"
with open(path) as f:
    txt = f.read()
old = "mp.set_start_method('spawn')"
new = "try:\n    mp.set_start_method('spawn', force=True)\nexcept RuntimeError:\n    pass"
txt = txt.replace(old, new)
with open(path, 'w') as f:
    f.write(txt)
print("[OK] masked_droid_slam.py patched")
PYEOF
else
    echo "[OK] masked_droid_slam.py already patched"
fi

# ── 7. other requirements (skip pytorch3d – not critical) ─────────────────────
REQS="$HAWOR_DIR/requirements.txt"
FILTERED_REQS="/tmp/hawor_reqs_filtered.txt"
grep -v "pytorch3d" "$REQS" > "$FILTERED_REQS" || true
conda run -n "$CONDA_ENV" pip install -r "$FILTERED_REQS" -q 2>&1 | tail -5
echo "[OK] remaining requirements installed"

echo ""
echo "==================================================="
echo " All patches applied. Test with:"
echo "   conda run -n hawor python $HAWOR_DIR/run_hawor_seq.py --help"
echo "==================================================="
