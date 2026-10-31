import json
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
class SequenceDataset(Dataset):
    def __init__(self,csv,image_size=224):
        self.df=pd.read_csv(csv)
        req={"frames_json","geometry_path","label","domain_id"}
        if not req.issubset(self.df.columns): raise ValueError(f"Manifest missing {req-set(self.df.columns)}")
        self.tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; paths=json.loads(r.frames_json)
        x=torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths])
        g=torch.from_numpy(np.load(r.geometry_path).astype("float32"))
        if len(x)!=len(g): raise ValueError(f"Frame/geometry mismatch row {i}")
        return x,g,torch.tensor(int(r.label)),torch.tensor(int(r.domain_id))
