import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Squared normalization """
    def __init__(self, ndim, p = -1., eps = 1e-8, bias=False):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(ndim))
        self.ndim = ndim
        self.p = p
        self.eps = eps
        self.bias = bias
        self.offset = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        if self.p < 0. or self.p > 1. :
            # 2 means L2 normalization
            norm_input = input.norm(2, dim=-1, keepdim=True) # provides a normalized output sqrt(x1^2 + x2^2 + .... + xn^2) (B, T, C)-> (B, T, 1)
            n = self.ndim
        else:
            partial_size = int(self.ndim * self.p)
            partial_input, _ = torch.split(input, [partial_size, self.ndim - partial_size], dim=-1)

            norm_input = partial_input.norm(2, dim=-1, keepdim=True)

            n = partial_size
        
        
        rms_norm = norm_input / (n ** (0.5))
        
        input_rms_norm = input / (rms_norm + self.eps)

        if self.bias :
            return self.scale * input_rms_norm + self.offset
        
        return self.scale * input_rms_norm
    
if __name__ == "__main__":

    x = torch.tensor([[1.,2.], [3.,4.]])
    rms = RMSNorm(2)

    output = rms(x)
    print(output)