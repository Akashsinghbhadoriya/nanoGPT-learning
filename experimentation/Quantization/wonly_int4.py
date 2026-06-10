"This is a weight only quantization approach"
import torch
import torch.nn as nn
from torch.nn import functional as F

def quantize_int4(weight):

    w_max = weight.abs().max(dim=1, keepdim=True)[0]

    scales = torch.clamp(w_max / 7.0, min=1e-8)
    
    w_q = torch.round(weight / scales)
    
    w_q = torch.clamp(w_q, -8, 7)
    
    w_q = w_q.to(torch.int8)
    
    return w_q, scales

def dequantize_int4(w_q, scales):

    return w_q.float() * scales

class WeightonlyINT4Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()

        weight_q, scales = quantize_int4(linear.weight.data)

        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scales", scales)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None

    def forward(self,x):

        W = dequantize_int4(self.weight_q, self.scales)
        
        return F.linear(x, W, self.bias)
