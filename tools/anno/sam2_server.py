#!/usr/bin/env python3
"""
sam2_server.py — SAM2 推理服务

在 hawor 环境下运行，通过 stdin/stdout 与 GUI 进程通信。
每行输入/输出都是一个 JSON 对象。

消息格式:
  输入: {"cmd": "set_image", "path": "..."}
  输出: {"status": "ok"}

  输入: {"cmd": "predict", "fg": [[x,y],...], "bg": [[x,y],...]}
  输出: {"status": "ok", "mask_path": "/tmp/...", "coverage": 32.5}

  输入: {"cmd": "quit"}
  输出: {"status": "bye"}
"""

import sys, os, json, tempfile, traceback
import numpy as np
import torch

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
SAM2_ROOT   = config.SAM2_DIR
SAM2_CKPT   = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CFGDIR = os.path.join(SAM2_ROOT, "sam2", "configs")
SAM2_CFGNAME = "sam2.1/sam2.1_hiera_t.yaml"

sys.path.insert(0, SAM2_ROOT)

def log(msg):
    print(f"[SAM2-SRV] {msg}", file=sys.stderr, flush=True)

def send(obj):
    print(json.dumps(obj), flush=True)

def load_model():
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=SAM2_CFGDIR, version_base="1.2"):
        cfg = compose(config_name=SAM2_CFGNAME)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    model.eval()
    ckpt = torch.load(SAM2_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return SAM2ImagePredictor(model), device


def main():
    log("Loading SAM2...")
    predictor, device = load_model()
    log(f"✅ SAM2 ready on {device}")
    send({"status": "ready"})

    from PIL import Image

    current_path = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg["cmd"]

            if cmd == "set_image":
                path = msg["path"]
                img = np.array(Image.open(path).convert("RGB"))
                with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                    predictor.set_image(img)
                current_path = path
                send({"status": "ok", "shape": list(img.shape[:2])})

            elif cmd == "predict":
                fg = msg.get("fg", [])
                bg = msg.get("bg", [])
                if not fg:
                    send({"status": "ok", "mask_path": "", "coverage": 0.0})
                    continue
                pts = np.array(fg + bg, dtype=np.float32)
                lbs = np.array([1]*len(fg) + [0]*len(bg), dtype=np.int32)
                with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                    masks, scores, _ = predictor.predict(
                        point_coords=pts, point_labels=lbs, multimask_output=True)
                best = int(np.argmax(scores))
                mask = (masks[best] * 255).astype(np.uint8)
                # Write mask to temp file
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                Image.fromarray(mask).save(tmp.name)
                coverage = float(mask.mean() / 255 * 100)
                send({"status": "ok", "mask_path": tmp.name, "coverage": coverage})

            elif cmd == "quit":
                send({"status": "bye"})
                break

            else:
                send({"status": "error", "msg": f"unknown cmd: {cmd}"})

        except Exception as e:
            send({"status": "error", "msg": str(e)})
            log(traceback.format_exc())


if __name__ == "__main__":
    main()
