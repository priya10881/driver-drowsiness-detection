import random,numpy as np,torch,yaml
def config(path):
    with open(path,encoding="utf8") as f: return yaml.safe_load(f)
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def device(d="auto"):
    return torch.device("cuda" if d=="auto" and torch.cuda.is_available() else ("cpu" if d=="auto" else d))
