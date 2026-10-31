import argparse,sys,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ddd.data import SequenceDataset
from src.ddd.model import DriverDrowsinessNet
from src.ddd.metrics import calculate
from src.ddd.utils import device
def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--output",default="results/eval.json"); p.add_argument("--device",default="auto"); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); c=ck["config"]; dev=device(a.device)
    ds=DataLoader(SequenceDataset(a.csv,c["image_size"]),batch_size=c["batch_size"])
    m=DriverDrowsinessNet(c["num_domains"],c["visual_dim"],c["hidden_dim"],c["lstm_layers"],c["dropout"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for f,g,y,d in ds: ps.extend(torch.softmax(m(f.to(dev),g.to(dev))["class_logits"],1).cpu().numpy()); ys.extend(y.numpy())
    result=calculate(ys,ps); result["n_sequences"]=len(ys)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
