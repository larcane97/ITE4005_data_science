import torch.nn as nn
import torch

class RecommendLoss(nn.Module):
    def __init__(self):
        super(RecommendLoss,self).__init__()
        
    def forward(self,target,pred):
        return torch.sqrt(((target-pred)**2).mean())