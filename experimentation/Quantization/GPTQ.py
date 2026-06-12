"""
Complete GPTQ Implementation
"""
import sys
from pathlib import Path

# Navigates up 4 levels from this file to reach '/Users/akashsingh/nanoGPT'
repo_root = str(Path(__file__).resolve().parents[3])

if repo_root not in sys.path:
    sys.path.append(repo_root)

from nanoGPT.model import GPT, GPTConfig
from transformers import GPT2Tokenizer
import torch
from torch.nn import functional as F
import torch.nn as nn
import os

# Step 1: Collecting activations

def collect_activations(model, layer, idx):
    activations = []

    def hook(module, inp, out):
        activations.append(inp[0].detach())

    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        model(idx)

    handle.remove()

    X = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
    print(X.shape)
    return X

# Step 2: Compute Hessian Approximation
# Step 3: Invert Hessian
def compute_hessian(X):
    H = X.T @ X
    damp = 0.01 * H.diag().mean()

    H += damp * torch.eye(H.shape[0], device=H.device) # damping along the diagonal all other values are not impacted

    return torch.linalg.inv(H)


# Step 4: GPTQ quantizer
def quantize_column_int4(w):

    scale = w.abs().max() / 7

    scale = torch.clamp(
        scale,
        min=1e-8
    )

    q = torch.round(
        w / scale
    )

    q = torch.clamp(
        q,
        -8,
        7
    )
    q = q.to(torch.int8)

    return q, scale

# Step 5: GPTQ Core Loop
def gptq_quantize(W, H_inv):

    W = W.clone()
    out, inp = W.shape
    W_int = torch.empty((out,inp),dtype=torch.int8, device=W.device)
    scales = torch.empty(inp, device=W.device)

    for i in range(inp):

        w = W[:, i]
        q, scale = quantize_column_int4(w)

        w_deq = q.float() * scale
        W_int[: , i] = q
        scales[i] = scale
        error = w - w_deq

        if i < inp - 1:

            correction = error/H_inv[i,i]
            W[:, i+1:] -= (correction.unsqueeze(1) * H_inv[i,i+1:])
    
    return W_int, scale

# packing weights
def pack_int4(q_weight):
    q = (q_weight + 8).to(torch.uint8)
    
    if q.numel() % 2:
        q = F.pad(q.flatten(), (0,1))
    else:
        q = q.flatten()
    
    low = q[0::2]
    high = q[1::2]

    packed = low | (high << 4)
    return packed

# unpacking the weights
def unpack_int4(packed):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    q = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
    q[0::2] = low.to(torch.int8) - 8
    q[1::2] = high.to(torch.int8) - 8

    return q

class GPTQQuantize(nn.Module):
    def __init__(self, layer, model, idx):
        super().__init__()

        X = collect_activations(model, layer, idx)
        H_inv = compute_hessian(X)
        w_q, scale = gptq_quantize(layer.weight.data, H_inv)
        packed = pack_int4(w_q)
        self.weight_shape = w_q.shape
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scale)

        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        weight_q = unpack_int4(self.packed)
        weight_q = weight_q.view(self.weight_shape)
        W = weight_q.float() * self.scales.unsqueeze(0)
        return F.linear(x, W, self.bias)

def quantize_model(module, model, idx):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):

            if name == "lm_head":
                continue
            print(f"Quantizing layer:{name}")
            setattr(module, name, GPTQQuantize(child, model, idx))
            print(f"Quantization complete layer:{name}")

        else:
            quantize_model(child, model, idx)

def evaluate_quantized_model():
    config = GPTConfig
    model = GPT.from_pretrained("gpt2")
    # torch.save(model.state_dict(), "gpt2.pt")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input = "Hi how are you? this"
    input_ids = tokenizer.encode(input, return_tensors="pt")
    with torch.no_grad():
        gpt_logits, gpt_loss = model(input_ids)

    quantize_model(model, model, input_ids)
    quant_model = model
    
    with torch.no_grad():
        quant_gpt_logits, quant_loss = quant_model(input_ids)

    mse = (
        (gpt_logits - quant_gpt_logits)
        .pow(2)
        .mean()
    )
    rmse = torch.sqrt(
        ((gpt_logits - quant_gpt_logits) ** 2).mean()
    )
    fp_pred = gpt_logits.argmax(dim=-1)
    q_pred = quant_gpt_logits.argmax(dim=-1)
    agreement = (
        fp_pred == q_pred
    ).float().mean()

    print("aggrement:",agreement)
    max_new_tokens = 100
    # print(tokenizer.decode(quant_model.generate(input_ids, max_new_tokens = max_new_tokens, temperature=0.8)))
    print(gpt_logits.shape)
    print(quant_gpt_logits.shape)
    print("mse:",mse)
    print("rmse:",rmse)

    cos = F.cosine_similarity(
        gpt_logits.flatten(),
        quant_gpt_logits.flatten(),
        dim=0
    )

    relative_error = (
        (gpt_logits - quant_gpt_logits).abs().mean()
        /
        gpt_logits.abs().mean()
    )
    print("cosine:", cos)
    print("relative error:", relative_error)

    torch.save(quant_model.state_dict(), "gpt2_int8.pt")
    if os.path.isfile("gpt2.pt"):
        print(
            os.path.getsize(
                "gpt2.pt"
            ) / 1024**2
        )

    if os.path.isfile("gpt2_int8.pt"):
        print(
            os.path.getsize(
                "gpt2_int8.pt"
            ) / 1024**2
        )

if __name__ == "__main__":
    
    evaluate_quantized_model()