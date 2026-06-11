#!/usr/bin/env python3
"""
label_object_category.py — Web-based free-text object labeler.

Shows rendered mesh on the left, free-text input on the right.
User types the object's real-world purpose/name in Chinese.

Usage:
    python tools/label_object_category.py
    python tools/label_object_category.py --resume
"""
from __future__ import annotations
import json, os, sys, io, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

PORT = 8765
OUTPUT_PATH = PROJ / "evaluation" / "configs" / "object_labels.json"

MESH_DIRS = [
    PROJ / "data_hub" / "ProcessedData" / "obj_meshes" / d
    for d in ["oakink", "oakink_scaled", "dexycb", "dexycb_scaled", "ycb", "unseen"]
] + [
    PROJ / "data_hub" / "meshes" / "SAM3DMesh" / sub / ds
    for sub in ["rotated_mesh", "meshes"] for ds in ["oakink", "unseen", "dexycb", "ycb"]
]


def find_mesh(obj_id):
    for d in MESH_DIRS:
        for name in [f"{obj_id}/mesh.ply", f"{obj_id}/mesh.obj"]:
            p = d / name
            if p.exists():
                return str(p)
    return None


def render_mesh_image(mesh_path: str) -> bytes:
    """Render mesh to bright PNG using matplotlib, Z-up."""
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.vertices -= mesh.bounding_box.centroid
    ext = mesh.bounding_box.extents.max()
    if ext > 0:
        mesh.vertices /= ext

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 6), facecolor="#1e1e2f")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1e1e2f")
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) > 10000:
        idx = np.random.choice(len(faces), 10000, replace=False)
        faces = faces[idx]

    # Bright colors
    poly = Poly3DCollection(
        verts[faces], alpha=0.92,
        facecolor="#a8b8ff", edgecolor="#7080cc", linewidth=0.08,
    )
    ax.add_collection3d(poly)
    lim = 0.55
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=20, azim=140)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


class LabelState:
    def __init__(self):
        self.objects: list[str] = []
        self.meta: dict[str, str] = {}
        self.mesh_paths: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        self.current_idx = 0
        self.image_cache: dict[str, bytes] = {}

    def init(self, resume=False):
        eval_path = PROJ / "output" / "evaluation" / "hp_pdm_yaw4x10_random_xy_seed42" / "eval_summary.json"
        with open(eval_path) as f:
            s = json.load(f)
        self.objects = sorted(s["by_object"].keys())

        oakink_path = Path("/home/lyh/Project/OakInk/shape/metaV2/object_id.json")
        oakink = {}
        if oakink_path.exists():
            with open(oakink_path) as f:
                oakink = json.load(f)

        for obj_id in self.objects:
            if obj_id in oakink:
                m = oakink[obj_id]
                attrs = ", ".join(a for a in m.get("attr", []) if a)
                self.meta[obj_id] = f'{m.get("name","?")} [{m.get("class","?")}] ({attrs})'
            elif obj_id.startswith("ycb_dex"):
                self.meta[obj_id] = f"DexYCB #{obj_id.split('_')[-1]}"
            elif obj_id.startswith("unseen"):
                self.meta[obj_id] = f"Unseen novel #{obj_id.split('_')[-1]}"
            else:
                self.meta[obj_id] = ""
            mp = find_mesh(obj_id)
            if mp:
                self.mesh_paths[obj_id] = mp

        if resume and OUTPUT_PATH.exists():
            with open(OUTPUT_PATH) as f:
                self.labels = json.load(f)
            for i, o in enumerate(self.objects):
                if o not in self.labels:
                    self.current_idx = i
                    break
            else:
                self.current_idx = len(self.objects)

        print(f"Objects: {len(self.objects)}, meshes: {len(self.mesh_paths)}, "
              f"labeled: {len(self.labels)}, starting at: {self.current_idx}")

    def save(self):
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

    def get_image(self, obj_id):
        if obj_id not in self.image_cache:
            mp = self.mesh_paths.get(obj_id)
            if mp:
                self.image_cache[obj_id] = render_mesh_image(mp)
        return self.image_cache.get(obj_id)

    def prerender_next(self, n=3):
        for i in range(self.current_idx, min(self.current_idx + n, len(self.objects))):
            oid = self.objects[i]
            if oid not in self.image_cache and oid in self.mesh_paths:
                try:
                    self.image_cache[oid] = render_mesh_image(self.mesh_paths[oid])
                except Exception:
                    pass


STATE = LabelState()

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>Object Labeler</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f0f1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
         display: flex; height: 100vh; overflow: hidden; }
  .left { flex: 1; display: flex; flex-direction: column; align-items: center;
          justify-content: center; padding: 20px; background: #12122a; }
  .left img { max-width: 100%%; max-height: 65vh; border-radius: 12px;
              box-shadow: 0 0 30px rgba(100,120,255,0.15); }
  .obj-title { font-size: 26px; font-weight: 700; color: #8b9cff; margin-bottom: 6px; }
  .obj-meta { font-size: 14px; color: #999; margin-bottom: 14px; }
  .progress { font-size: 13px; color: #555; margin-top: 10px; }
  .right { width: 380px; display: flex; flex-direction: column;
           padding: 24px; background: #161630; border-left: 1px solid #2a2a4a; }
  .right-title { font-size: 18px; font-weight: 700; color: #7b8cef; margin-bottom: 16px; }
  .recent { flex: 1; overflow-y: auto; margin-bottom: 16px; }
  .recent-item { padding: 6px 10px; border-radius: 6px; margin-bottom: 4px;
                 font-size: 13px; background: #1c1c3a; display: flex; justify-content: space-between; }
  .recent-id { color: #7b8cef; font-weight: 600; }
  .recent-label { color: #bbb; }
  .input-area { padding-top: 16px; border-top: 1px solid #2a2a4a; }
  .input-label { font-size: 14px; color: #aaa; margin-bottom: 8px; }
  form { display: flex; flex-direction: column; gap: 10px; }
  input[type=text] { padding: 14px 16px; font-size: 18px; background: #1a1a3a;
                     border: 2px solid #3a3a6a; border-radius: 8px; color: #fff;
                     outline: none; }
  input[type=text]:focus { border-color: #6b7bef; }
  .btn-row { display: flex; gap: 8px; }
  .btn { padding: 10px 18px; border: none; border-radius: 8px; color: #fff;
         font-size: 14px; cursor: pointer; font-weight: 600; flex: 1; text-align: center;
         text-decoration: none; }
  .btn-ok { background: #4a5adf; }
  .btn-ok:hover { background: #5a6aef; }
  .btn-skip { background: #444; }
  .btn-skip:hover { background: #555; }
  .btn-undo { background: #8a4a00; }
  .btn-undo:hover { background: #a05a00; }
  .existing { font-size: 12px; color: #5a5; margin-top: 4px; }
</style></head><body>
<div class="left">
  %%LEFT%%
</div>
<div class="right">
  <div class="right-title">📝 Recent Labels</div>
  <div class="recent">%%RECENT%%</div>
  <div class="input-area">
    %%INPUT%%
  </div>
</div>
<script>
document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('label-input');
  if (inp) inp.focus();
});
</script></body></html>"""


def build_page():
    idx = STATE.current_idx
    total = len(STATE.objects)

    if idx >= total:
        left = '<div class="obj-title">✅ 全部标注完成!</div>'
        inp = f'<div style="color:#6b7bef">已保存 → {OUTPUT_PATH.name}</div>'
        recent = ""
        return PAGE_HTML.replace("%%LEFT%%", left).replace("%%RECENT%%", recent).replace("%%INPUT%%", inp)

    obj_id = STATE.objects[idx]
    meta = STATE.meta.get(obj_id, "")
    existing = STATE.labels.get(obj_id, "")

    left = (f'<div class="obj-title">{obj_id}</div>'
            f'<div class="obj-meta">{meta}</div>'
            f'<img src="/image/{obj_id}" alt="{obj_id}">'
            f'<div class="progress">{idx+1} / {total} · 已标注: {len(STATE.labels)}</div>')

    inp = (f'<div class="input-label">输入 <b>{obj_id}</b> 的真实用途/名称（中文）</div>'
           f'<form action="/label" method="GET">'
           f'<input type="text" id="label-input" name="text" '
           f'placeholder="例: 洗洁精按压瓶" value="{existing}">'
           f'<div class="btn-row">'
           f'<button type="submit" class="btn btn-ok">确认 ✓</button>'
           f'<a href="/skip" class="btn btn-skip">跳过 ⏭</a>'
           f'<a href="/undo" class="btn btn-undo">撤销 ↩</a>'
           f'</div></form>')
    if existing:
        inp += f'<div class="existing">当前标签: {existing}</div>'

    # Recent labels (last 15)
    recent_items = []
    labeled_objs = [o for o in STATE.objects[:idx] if o in STATE.labels]
    for o in labeled_objs[-15:]:
        recent_items.append(
            f'<div class="recent-item">'
            f'<span class="recent-id">{o}</span>'
            f'<span class="recent-label">{STATE.labels[o]}</span></div>'
        )
    recent = "\n".join(reversed(recent_items)) if recent_items else '<div style="color:#555">暂无</div>'

    return PAGE_HTML.replace("%%LEFT%%", left).replace("%%RECENT%%", recent).replace("%%INPUT%%", inp)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/image/"):
            obj_id = path.split("/image/")[1]
            img = STATE.get_image(obj_id)
            if img:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(img)
            else:
                self.send_response(404)
                self.end_headers()
            return

        if path == "/label":
            qs = parse_qs(parsed.query)
            text = qs.get("text", [""])[0].strip()
            if text and STATE.current_idx < len(STATE.objects):
                obj_id = STATE.objects[STATE.current_idx]
                STATE.labels[obj_id] = text
                print(f"  ✓ {obj_id} → {text}")
                STATE.current_idx += 1
                STATE.save()
                threading.Thread(target=STATE.prerender_next, daemon=True).start()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if path == "/skip":
            if STATE.current_idx < len(STATE.objects):
                print(f"  ⏭ Skipped {STATE.objects[STATE.current_idx]}")
                STATE.current_idx += 1
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if path == "/undo":
            if STATE.current_idx > 0:
                STATE.current_idx -= 1
                obj_id = STATE.objects[STATE.current_idx]
                if obj_id in STATE.labels:
                    del STATE.labels[obj_id]
                    STATE.save()
                print(f"  ↩ Undo → {obj_id}")
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        html = build_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    STATE.init(resume=args.resume)

    print("Pre-rendering first objects...")
    STATE.prerender_next(5)
    print("Ready!")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n🌐 Open in browser: {url}\n")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STATE.save()
        print(f"\n✅ Saved {len(STATE.labels)} labels → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
