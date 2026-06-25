#!/usr/bin/env python3
"""Render frame-0 overlay: project GT-CAD (GT pose, multiple Euler conventions)
and FP-C pose onto the RGB, to validate GT-pose construction and judge FP visually."""
import json, glob
import numpy as np, cv2, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJ = Path(__file__).resolve().parents[2]
GT_ANNO = PROJ/"data/egocentric/hoi4d/HOI4D_annotations"
RAW = PROJ/"data_hub/RawData/EgoRawData/hoi4d/HOI4D_release"
POSE = PROJ/"data_hub/ProcessedData/poses/Egocentric"
SEQS = json.load(open(PROJ/"output/eval/hoi4d_10seqs.json"))
K = np.array([[1376.8,0,967.7],[0,1376.2,526.8],[0,0,1.0]])
OUT = Path("/tmp/claude-1000/-home-lyh/460f2b88-ef44-43d7-baa2-71839cd6794a/scratchpad")

def proj(Xc):
    x=(K@Xc.T).T; return x[:,:2]/x[:,2:3]

def frame0(seq_rel):
    cap=cv2.VideoCapture(str(RAW/seq_rel/"align_rgb/image.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0); _,f=cap.read(); cap.release(); return f

for s in SEQS:
    if not s["gt"]["rigid"]: continue
    seq=s["seq_id"].split("/")[-1]; rel="/".join(seq.split("_")[:7])
    img=frame0(rel);
    d=json.load(open(GT_ANNO.joinpath(*seq.split("_")[:7],"objpose","0.json")))["dataList"][0]
    e=[d["rotation"]["x"],d["rotation"]["y"],d["rotation"]["z"]]
    t=np.array([d["center"][k] for k in "xyz"])
    cad=trimesh.load(s["gt"]["cad_path"],process=False); V=cad.vertices
    c_local=(V.min(0)+V.max(0))/2; Vc=V-c_local
    idx=np.linspace(0,len(Vc)-1,1500).astype(int); Vs=Vc[idx]
    # GT 2dBox (cyan)
    box=np.array(d["2dBox"]["camera1"]);
    cv2.rectangle(img,tuple(box.min(0).astype(int)),tuple(box.max(0).astype(int)),(255,255,0),3)
    # GT pose under several conventions (green shades)
    cols={"XYZ":(0,255,0),"ZYX":(0,180,0),"xyz":(0,255,120),"zyx":(0,120,255),"YXZ":(120,255,0),"ZXY":(0,200,200)}
    for conv,col in cols.items():
        R=Rotation.from_euler(conv,e).as_matrix(); Xc=(R@Vs.T).T+t
        if np.any(Xc[:,2]<=0): continue
        uv=proj(Xc).astype(int)
        for u,v in uv[::8]:
            if 0<=u<img.shape[1] and 0<=v<img.shape[0]: cv2.circle(img,(u,v),1,col,-1)
    # FP-C pose (red)
    Pf=load=None
    fp=POSE/"hoi4d_gtC"/seq/"ob_in_cam"/"000000.txt"
    if fp.exists():
        T=np.loadtxt(fp); Xc=(T[:3,:3]@(Vs+c_local).T).T+T[:3,3]
        uv=proj(Xc).astype(int)
        for u,v in uv[::8]:
            if 0<=u<img.shape[1] and 0<=v<img.shape[0]: cv2.circle(img,(u,v),1,(0,0,255),-1)
    cv2.putText(img,f"{s['obj']} cyan=2dBox green/teal=GT(convs) red=FP-C",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
    p=OUT/f"viz_{s['cat']}_{s['obj']}.png"; cv2.imwrite(str(p),img); print("wrote",p)
