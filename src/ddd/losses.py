import torch
import torch.nn as nn
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l): ctx.l=l; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.l*g,None
def grad_reverse(x,l=1.0): return GRL.apply(x,l)
class DomainLoss(nn.Module):
    def __init__(self): super().__init__(); self.loss=nn.CrossEntropyLoss()
    def forward(self,logits,labels): return self.loss(logits,labels)
