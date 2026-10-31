import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix
def calculate(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p); pred=p.argmax(1)
    pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=[1],average="macro",zero_division=0)
    try: auc=float(roc_auc_score(y,p[:,1]))
    except ValueError: auc=None
    return {"accuracy":float(accuracy_score(y,pred)),"precision_drowsy":float(pr),
      "recall_drowsy":float(re),"f1_drowsy":float(f1),"roc_auc":auc,
      "confusion_matrix_labels_0_alert_1_drowsy":confusion_matrix(y,pred,labels=[0,1]).tolist()}
