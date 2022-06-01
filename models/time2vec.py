import torch
from torch import nn
from torch.nn import functional as F

class SinWithScaleActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.scale = nn.parameter.Parameter( 
            torch.randn(1), requires_grad=True
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(1), requires_grad=True
        )
    
    def forward(self, cur_activations):
        return torch.sin(cur_activations * self.scale + self.bias) 

class Time2Vec(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k
        self.activations = nn.ModuleList(
            [
                SinWithScaleActivation() for _ in range(k) 
            ]
        )

    def forward(self, times):
        # (times has shape *) 
        # (output has shape *, k)
        if self.k == 0:
            return times.unsqueeze(-1)
        outputs = [times]
        for i in range(self.k):
            current_time = self.activations[i](outputs[-1])
            outputs.append(current_time)
        return torch.stack(outputs, dim=-1)