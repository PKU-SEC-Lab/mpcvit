import torch.nn as nn 
import torch
import torch.nn.functional as F

class Learnable_Relu(nn.Module):
    def __init__(self, slope_init=0.):
        super(Learnable_Relu, self).__init__()
        self.slope = nn.Parameter(torch.tensor(slope_init))
        self.slope_lr_scale = 1
        self.flag = 1
    
    def set_flag(self, flag):
        self.flag = flag

    def forward(self, x):
        slope = (self.slope - self.slope * self.slope_lr_scale).detach() + self.slope * self.slope_lr_scale
        x = F.relu(x) + (x - F.relu(x)) * torch.clamp(slope, 0, 1)

        return x