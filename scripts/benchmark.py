import argparse,sys,time,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ddd.model import DriverDrowsinessNet
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="cpu"); p.add_argument("--iterations",type=int,default=60); p.add_argument("--warmup",type=int,default=8); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); c=ck["config"]; dev=torch.device(a.device)
    m=DriverDrowsinessNet(c["num_domains"],c["visual_dim"],c["hidden_dim"],c["lstm_layers"],c["dropout"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
    x=torch.randn(1,c["sequence_length"],3,c["image_size"],c["image_size"],device=dev); g=torch.randn(1,c["sequence_length"],5,device=dev)
    with torch.no_grad():
        for _ in range(a.warmup): m(x,g)
        if dev.type=="cuda": torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(a.iterations): m(x,g)
        if dev.type=="cuda": torch.cuda.synchronize()
    ms=(time.perf_counter()-t)*1000/a.iterations
    print(json.dumps({"device":str(dev),"latency_ms_per_sequence":ms,"sequences_per_second":1000/ms,"approx_input_fps":c["sequence_length"]*1000/ms},indent=2))
if __name__=="__main__": main()
