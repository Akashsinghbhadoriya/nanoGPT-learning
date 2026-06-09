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
        
        q_weight = torch.round( tensor / scales).to(torch.int8)

        return q_weight, scales

    def de_quantize_channel_int8(self, q_weight, scales):
        
        return q_weight.float() * scales
    
# =========================================================
# LLM.int8() Mixed-Precision quantization
# =========================================================
class MixedPrecisionQuantization:
    def __init__(self, w, th, x):
        self.w = w
        self.th = th
        self.x = x

    def forward(self):
        max_per_channel = self.x.abs().max(dim = 0)[0]

        outlier_mask = (max_per_channel > self.th)
        
        norm_idx = (~outlier_mask).nonzero().squeeze()
        out_idx = (outlier_mask).nonzero().squeeze()
        
        x_norm =  self.x[:, norm_idx]
        x_out = self.x[:, out_idx]
        w_norm = self.w[:, norm_idx]
        w_out = self.w[:, out_idx]
        if x_norm.shape[1] > 0:
            q = QuantizedChannel_INT8(w_norm)
            y_norm = q.forward(x_norm)
            y_out = x_out @ w_out.T
            return y_norm + y_out
        
        return x_out @ w_out.T

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
    # print(torch.mean(y-y_int8) ** 2)
    # print("FP32 memory:", (W.numel() * 4)/(1024 * 1024))
    # print("INT8 memory:", (layer.weight_q.numel() * 1)/(1024*1024))

    start = time.time()
    for _ in range(100):
        a = x @ W.T

    # print("time for FP32:", (time.time() - start)*1000)

    start = time.time()
    for _ in range(100):
        a_int8 = layer.forward(x)

    # print("time for quantized INT8:", (time.time()- start)*1000)
    return torch.mean(y-y_int8) ** 2

def calculate_mixed_precision(x, W, layer):
    y = x @ W.T
    y_int8 = layer.forward()
    
    return torch.mean(y-y_int8) ** 2

if __name__ == "__main__":

    hidden_dim = 768
    x = torch.randn(128, hidden_dim)
    w = torch.randn(3072, hidden_dim)
    x[0][0] = 20
    x[0][10] = 100
    th = 10.0
    layer = MixedPrecisionQuantization(w, th, x)
    mse = calculate_mixed_precision(x, w, layer)
    print("int8.() mse:",mse)

    layer1 = QuantizedChannel_INT8(w)
    mse1 = calculate_quantization_channel(x,w,layer1)
    print("Per-channel Quantization:", mse1)
