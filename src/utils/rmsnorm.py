import math
import torch
import torch.nn as nn


class GatedRMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-8
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        normed_x = x / norm.clamp(min = self.eps) * self.g
        return normed_x * (x * self.w).sigmoid()