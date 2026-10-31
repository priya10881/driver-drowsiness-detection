import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
def split(df,frac,seed):
    i,j=next(GroupShuffleSplit(n_splits=1,test_size=frac,random_state=seed).split(df,groups=df.subject_id.astype(str)))
    return df.iloc[i],df.iloc[j]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--val-fraction",type=float,default=.15); p.add_argument("--test-fraction",type=float,default=.15); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); df=pd.read_csv(a.manifest); o=Path(a.output_dir); o.mkdir(parents=True,exist_ok=True)
    tr=[]; va=[]; te=[]
    for _,d in df.groupby("domain"):
        tv,t=split(d,a.test_fraction,a.seed); x,v=split(tv,a.val_fraction/(1-a.test_fraction),a.seed)
        tr.append(x); va.append(v); te.append(t)
    for n,parts in [("train",tr),("val",va),("test",te)]: pd.concat(parts).to_csv(o/f"{n}.csv",index=False)
    print("Wrote train/val/test; inspect subject IDs and dataset protocols.")
if __name__=="__main__": main()
