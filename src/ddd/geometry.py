import numpy as np
LEFT=[33,160,158,133,153,144]; RIGHT=[362,385,387,263,373,380]
MOUTH=[78,81,13,311,308,402,14,178]
def d(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))
def ear(p,idx): return (d(p[idx[1]],p[idx[5]])+d(p[idx[2]],p[idx[4]]))/(2*d(p[idx[0]],p[idx[3]])+1e-6)
def extract_geometry(points):
    p=np.asarray(points,dtype=np.float32)
    e=(ear(p,LEFT)+ear(p,RIGHT))/2
    m=np.asarray([p[i] for i in MOUTH])
    mar=(d(m[1],m[7])+d(m[2],m[6])+d(m[3],m[5]))/(3*d(m[0],m[4])+1e-6)
    le,re,nose,chin=p[33],p[263],p[1],p[152]
    mid=(le+re)/2
    roll=np.degrees(np.arctan2(re[1]-le[1],re[0]-le[0]))
    yaw=(nose[0]-mid[0])/(abs(re[0]-le[0])+1e-6)
    pitch=(nose[1]-mid[1])/(abs(chin[1]-mid[1])+1e-6)
    return np.asarray([e,mar,yaw,pitch,roll],dtype=np.float32)
