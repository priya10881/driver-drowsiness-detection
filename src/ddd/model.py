import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .losses import grad_reverse
class CrossModalAttention(nn.Module):
    def __init__(self,d=256,heads=4,drop=.1):
        super().__init__()
        self.vg=nn.MultiheadAttention(d,heads,batch_first=True,dropout=drop)
        self.gv=nn.MultiheadAttention(d,heads,batch_first=True,dropout=drop)
        self.vn=nn.LayerNorm(d); self.gn=nn.LayerNorm(d)
        self.out=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop))
    def forward(self,v,g):
        # Attend across the 30 time steps in the opposite modality.
        # Q=visual, K/V=geometry and vice versa; attention is not a one-token operation.
        vc,_=self.vg(v,g,g,need_weights=False)
        gc,_=self.gv(g,v,v,need_weights=False)
        return self.out(torch.cat([self.vn(v+vc),self.gn(g+gc)],-1))
class DriverDrowsinessNet(nn.Module):
    def __init__(self,num_domains=3,visual_dim=256,hidden_dim=128,lstm_layers=1,dropout=.3,pretrained=False):
        super().__init__()
        weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
        net=mobilenet_v2(weights=weights)
        self.cnn=net.features; self.pool=nn.AdaptiveAvgPool2d(1)
        self.vproj=nn.Sequential(nn.Linear(1280,visual_dim),nn.ReLU(),nn.Dropout(dropout))
        self.gproj=nn.Sequential(nn.Linear(5,64),nn.ReLU(),nn.Linear(64,visual_dim),nn.ReLU())
        self.fusion=CrossModalAttention(visual_dim,4,dropout)
        self.lstm=nn.LSTM(visual_dim,hidden_dim,num_layers=lstm_layers,batch_first=True,
                          bidirectional=True,dropout=dropout if lstm_layers>1 else 0)
        self.drop=nn.Dropout(dropout)
        self.cls=nn.Linear(2*hidden_dim,2)
        self.domain=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_domains))
    def forward(self,frames,geometry,grl_strength=0.0):
        b,t,c,h,w=frames.shape
        x=self.pool(self.cnn(frames.reshape(b*t,c,h,w))).flatten(1)
        v=self.vproj(x).reshape(b,t,-1); g=self.gproj(geometry)
        fused=self.fusion(v,g); seq,_=self.lstm(fused)
        pooled=self.drop(seq).mean(1)
        return {"class_logits":self.cls(pooled),"domain_logits":self.domain(grad_reverse(pooled,grl_strength))}
