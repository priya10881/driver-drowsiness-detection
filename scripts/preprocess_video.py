import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ddd.geometry import extract_geometry
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input-csv",required=True); a.add_argument("--output-dir",default="data/processed")
    a.add_argument("--sequence-length",type=int,default=30); a.add_argument("--sample-fps",type=float,default=30)
    z=a.parse_args(); import mediapipe as mp
    fm=mp.solutions.face_mesh; out=Path(z.output_dir); crops=out/"faces"; crops.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(z.input_csv); req={"video_path","label","subject_id","domain"}
    if not req.issubset(df.columns): raise SystemExit(f"CSV needs {req}")
    domains={d:i for i,d in enumerate(sorted(df.domain.astype(str).unique()))}; rows=[]
    for _,r in tqdm(df.iterrows(),total=len(df)):
        vp=Path(str(r.video_path)); vp=vp if vp.is_absolute() else ROOT/vp
        cap=cv2.VideoCapture(str(vp))
        if not cap.isOpened(): print("Cannot open",vp); continue
        fps=cap.get(cv2.CAP_PROP_FPS) or z.sample_fps; stride=max(1,round(fps/z.sample_fps))
        fs=[]; gs=[]; idx=0
        with fm.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
            while True:
                ok,im=cap.read()
                if not ok: break
                if idx%stride==0:
                    h,w=im.shape[:2]; rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); res=mesh.process(rgb)
                    if res.multi_face_landmarks:
                        lm=res.multi_face_landmarks[0].landmark; pts=np.array([[p.x,p.y] for p in lm],np.float32)
                        x1=max(0,int(pts[:,0].min()*w)-10); x2=min(w,int(pts[:,0].max()*w)+10)
                        y1=max(0,int(pts[:,1].min()*h)-10); y2=min(h,int(pts[:,1].max()*h)+10)
                        if x2>x1 and y2>y1:
                            fs.append(cv2.cvtColor(im[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)); gs.append(extract_geometry(pts))
                idx+=1
        cap.release(); n=z.sequence_length
        for start in range(0,len(fs)-n+1,n):
            sid=str(r.subject_id).replace("/","_").replace("\\\\","_"); stem=f"{vp.stem}_{sid}_{start:06d}"
            paths=[]
            for j,im in enumerate(fs[start:start+n]):
                p=crops/f"{stem}_{j:02d}.jpg"; cv2.imwrite(str(p),cv2.cvtColor(im,cv2.COLOR_RGB2BGR)); paths.append(str(p.resolve()))
            gp=out/f"{stem}_geometry.npy"; np.save(gp,np.stack(gs[start:start+n]))
            rows.append({"sequence_id":stem,"subject_id":str(r.subject_id),"domain":str(r.domain),
              "domain_id":domains[str(r.domain)],"label":int(r.label),"frames_json":json.dumps(paths),"geometry_path":str(gp.resolve())})
    pd.DataFrame(rows).to_csv(out/"manifest.csv",index=False); print("Sequences:",len(rows),"manifest:",out/"manifest.csv")
if __name__=="__main__": main()
