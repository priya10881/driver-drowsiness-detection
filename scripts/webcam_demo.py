"""EAR/MAR live cue baseline; this is NOT the trained multimodal model."""
import argparse,cv2,sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ddd.geometry import extract_geometry
def main():
    p=argparse.ArgumentParser(); p.add_argument("--camera",type=int,default=0); p.add_argument("--ear-threshold",type=float,default=.20); p.add_argument("--mar-threshold",type=float,default=.65); a=p.parse_args()
    import mediapipe as mp
    cap=cv2.VideoCapture(a.camera)
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True) as mesh:
        while cap.isOpened():
            ok,im=cap.read()
            if not ok: break
            r=mesh.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); label="No face detected"
            if r.multi_face_landmarks:
                pts=np.array([[p.x,p.y] for p in r.multi_face_landmarks[0].landmark],np.float32)
                ear,mar,yaw,pitch,roll=extract_geometry(pts); label=f"EAR {ear:.2f} MAR {mar:.2f} yaw {yaw:.2f} pitch {pitch:.2f} roll {roll:.1f}"
                if ear<a.ear_threshold or mar>a.mar_threshold: label="CUE - verify driver state | "+label
            cv2.putText(im,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2); cv2.imshow("Geometric baseline (ESC to quit)",im)
            if cv2.waitKey(1)==27: break
    cap.release(); cv2.destroyAllWindows()
if __name__=="__main__": main()
