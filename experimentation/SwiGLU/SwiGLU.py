import torch
import torch.nn as nn
from torch.nn import functional as F

class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.w1 = nn.Linear(config.n_embd, config.hidden_dim, bias = config.bias)
        self.w2 = nn.Linear(config.n_embd, config.hidden_dim, bias = config.bias)
        self.w3 = nn.Linear(config.hidden_dim, config.n_embd, bias = config.bias)

    def forward(self, x):
        gate = F.silu(self.w1(x)) # (B, T, C) * (C, HD) -> (B, T, HD)
        value = self.w2(x) # (B, T, C) * (C, HD) -> (B, T, HD)

        x = self.w3(gate * value) # (B, T, HD) * (HD, C) -> (B, T, C)
        return x
    
class GPTConfig:
    n_embd: int = 6
    hidden_dim: int = 16
    bias: bool = True

if __name__ == "__main__":

    x = torch.randn(1,1,6)
    config = GPTConfig()
    swiglu = SwiGLU(config)

    output = swiglu(x)
    print(f"Output: {output}")
