import torch
import time

# =========================================================
# INT8 Quantization Linear Layer (Weight only quantization)
# =========================================================

def quantize_int8(tensor):
    max_val = tensor.abs().max()

    scale = max_val / 127

    q_tensor = torch.round( tensor / scale).to(torch.int8)

    return q_tensor, scale

def de_quantize_int8(q_tensor, scale):
    
    return q_tensor.float() * scale

class Linear:
    def __init__(self, d_in, d_out):
        self.weight = torch.randn(d_in, d_out)

    def forward(self, x):
        return x @ self.weight
    
class QuantizedLinear:
    def __init__(self, weight):
        self.weight_q, self.scale = quantize_int8(weight)
    
    def forward(self, x):
        W = de_quantize_int8(self.weight_q, self.scale)

        return x @ W

if __name__ == "__main__":

    W = torch.randn(128, 256)
    x = torch.randn(32, 128)
    y = x @ W
    layer = QuantizedLinear(W)
    y_int8 = layer.forward(x)
    print(torch.mean(y-y_int8) ** 2)
    print("FP32 memory:", (W.numel() * 4)/(1024 * 1024))
    print("INT8 memory:", (layer.weight_q.numel() * 1)/(1024*1024))

    start = time.time()
    for _ in range(100):
        y = x @ W

    print("time for FP32:", (time.time() - start)*1000)

    start = time.time()
    for _ in range(100):
        y_int8 = layer.forward(x)

    print("time for quantized INT8:", (time.time()- start)*1000)