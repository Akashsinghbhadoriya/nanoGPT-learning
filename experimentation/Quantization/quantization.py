import torch
import time


class Linear:
    def __init__(self, d_in, d_out):
        self.weight = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return x @ self.weight
# =========================================================
# INT8 Quantization Linear Layer (Weight only quantization)
# =========================================================    
class QuantizedLinear_INT8:
    def __init__(self, weight):
        self.weight_q, self.scale = self.quantize_int8(weight)
    
    def forward(self, x):
        W = self.de_quantize_int8(self.weight_q, self.scale)

        return x @ W
    
    def quantize_int8(self, tensor):
        max_val = tensor.abs().max()

        scale = max_val / 127

        q_tensor = torch.round( tensor / scale).to(torch.int8)

        return q_tensor, scale

    def de_quantize_int8(self, q_tensor, scale):
        
        return q_tensor.float() * scale
    
    
# =========================================================
# INT8 Quantization Linear Layer (Weight only quantization)
# =========================================================
class QuantizedChannel_INT8:
    def __init__(self, weight):
        self.weight_q, self.scale = self.quantize_channel_int8(weight)
    
    def forward(self, x):
        W = self.de_quantize_channel_int8(self.weight_q, self.scale)

        return x @ W.T
    
    def quantize_channel_int8(self, tensor):
        max_val = tensor.abs().max(dim=1,keepdim=True)[0]

        scales = max_val / 127
        print(scales.flatten())
        q_weight = torch.round( tensor / scales).to(torch.int8)

        return q_weight, scales

    def de_quantize_channel_int8(self, q_weight, scales):
        
        return q_weight.float() * scales

def calculate_quantization(x, W, layer):
    y = x @ W
    y_int8 = layer.forward(x)
    print(torch.mean(y-y_int8) ** 2)
    print("FP32 memory:", (W.numel() * 4)/(1024 * 1024))
    print("INT8 memory:", (layer.weight_q.numel() * 1)/(1024*1024))

    start = time.time()
    for _ in range(100):
        a = x @ W

    print("time for FP32:", (time.time() - start)*1000)

    start = time.time()
    for _ in range(100):
        a_int8 = layer.forward(x)

    print("time for quantized INT8:", (time.time()- start)*1000)
    return torch.mean(y-y_int8) ** 2

def calculate_quantization_channel(x, W, layer):
    y = x @ W.T
    y_int8 = layer.forward(x)
    print(torch.mean(y-y_int8) ** 2)
    print("FP32 memory:", (W.numel() * 4)/(1024 * 1024))
    print("INT8 memory:", (layer.weight_q.numel() * 1)/(1024*1024))

    start = time.time()
    for _ in range(100):
        a = x @ W.T

    print("time for FP32:", (time.time() - start)*1000)

    start = time.time()
    for _ in range(100):
        a_int8 = layer.forward(x)

    print("time for quantized INT8:", (time.time()- start)*1000)
    return torch.mean(y-y_int8) ** 2

if __name__ == "__main__":

    count = 0
    for i in range(100):
        W = torch.randn(4, 4)
        W[0] *= 10000
        W[1] *= 5000
        W[2] *= 2000
        W[3] *= 1000
        x = torch.randn(4, 4)
        layer = QuantizedLinear_INT8(W)
        tmse = calculate_quantization(x, W, layer)
        W = W.T
        print(W)
        layer_channel = QuantizedChannel_INT8(W)
        cmse = calculate_quantization_channel(x, W, layer_channel)
        if(tmse < cmse):
            count+=1

    print("count:",count)