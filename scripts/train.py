import argparse,sys,math
from pathlib import Path
import torch,torch.nn as nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ddd.data import SequenceDataset
from src.ddd.model import DriverDrowsinessNet
from src.ddd.losses import DomainLoss
from src.ddd.utils import config,seed,device
def main():
    p=argparse.ArgumentParser()
    for x in ["train-csv","val-csv"]: p.add_argument("--"+x,required=True)
    p.add_argument("--target-csv"); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--output-dir",default="checkpoints"); p.add_argument("--device",default="auto")
    a=p.parse_args(); c=config(a.config); seed(c["seed"]); dev=device(a.device)
    tr=DataLoader(SequenceDataset(a.train_csv,c["image_size"]),batch_size=c["batch_size"],shuffle=True,num_workers=c["num_workers"])
    va=DataLoader(SequenceDataset(a.val_csv,c["image_size"]),batch_size=c["batch_size"])
    tg=DataLoader(SequenceDataset(a.target_csv,c["image_size"]),batch_size=c["batch_size"],shuffle=True) if a.target_csv else None
    m=DriverDrowsinessNet(c["num_domains"],c["visual_dim"],c["hidden_dim"],c["lstm_layers"],c["dropout"]).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    ce=nn.CrossEntropyLoss(); de=DomainLoss(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); stale=0; ti=iter(tg) if tg else None
    for ep in range(c["epochs"]):
        m.train(); losses=[]
        for f,g,y,d in tr:
            f,g,y,d=f.to(dev),g.to(dev),y.to(dev),d.to(dev); opt.zero_grad()
            lam=c["grl_max_lambda"]*(2/(1+math.exp(-10*(ep+1)/c["epochs"]))-1)
            o=m(f,g,lam); loss=ce(o["class_logits"],y)
            if ti:
                try: tf,tg0,_,td=next(ti)
                except StopIteration: ti=iter(tg); tf,tg0,_,td=next(ti)
                ot=m(tf.to(dev),tg0.to(dev),lam)
                loss=loss+c["domain_loss_weight"]*de(torch.cat([o["domain_logits"],ot["domain_logits"]]),torch.cat([d,td.to(dev)]))
            loss.backward(); opt.step(); losses.append(loss.item())
        m.eval(); vl=[]
        with torch.no_grad():
            for f,g,y,d in va: vl.append(ce(m(f.to(dev),g.to(dev))["class_logits"],y.to(dev)).item())
        v=sum(vl)/max(1,len(vl)); print(f"epoch {ep+1}: train={sum(losses)/max(1,len(losses)):.4f} val={v:.4f}")
        if v<best:
            best=v; stale=0; torch.save({"model":m.state_dict(),"config":c},out/"best.pt")
        else:
            stale+=1
            if stale>=c["early_stopping_patience"]: print("early stop"); break
    print("Saved",out/"best.pt")
if __name__=="__main__": main()
