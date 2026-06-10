"""
Implementing QuantizedLinear and W8A8 on the gpt2 small model
The mse for Quantize linear < W8A8 as in W8A8 outliers are afftecting the mse
"""

import sys
from pathlib import Path

# Navigates up 4 levels from this file to reach '/Users/akashsingh/nanoGPT'
repo_root = str(Path(__file__).resolve().parents[3])

if repo_root not in sys.path:
    sys.path.append(repo_root)

from nanoGPT.model import GPT, GPTConfig
import os
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from wonly_int4 import WeightonlyINT4Linear
 
class QuantizedLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()

        W = linear.weight.data
        
        max_vals = W.abs().max(dim=1, keepdim=True)[0]

        scales = torch.clamp(max_vals / 127, min=1e-8)

        W_q = torch.round(W / scales).to(torch.int8)

        self.register_buffer("weight_q", W_q)
        self.register_buffer("scales", scales)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None

    def forward(self, x):

        W = (self.weight_q.float() * self.scales)

        return F.linear(x, W, self.bias)
    
class W8A8(nn.Module):
    def __init__(self,linear):
        super().__init__()

        W = linear.weight.data

        max_vals = W.abs().max(dim=1, keepdim=True)[0]

        scales = torch.clamp(max_vals / 127, min=1e-8)

        W_q = torch.round(W / scales).to(torch.int8)

        self.register_buffer("weight_q", W_q)
        self.register_buffer("scales", scales)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        x_q , x_scale = self.quantize_x(x);
    
        x_d = x_q.float() * x_scale
        W = (self.weight_q.float() * self.scales)

        return F.linear(x_d, W, self.bias)
        

    def quantize_x(self, x):
        x_max = x.abs().max()

        scale = torch.clamp( x_max / 127, 1e-8)

        x_q = torch.round(x / scale)

        return x_q, scale
    
def quantize_model(module):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):

            setattr(module, name, WeightonlyINT4Linear(child))

        else:
            quantize_model(child)

    
    return module


if __name__=="__main__":
    
    config = GPTConfig
    model = GPT.from_pretrained("gpt2")
    # torch.save(model.state_dict(), "gpt2.pt"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input = "Hi how are you?"
    input_ids = tokenizer.encode(input, return_tensors="pt")
    with torch.no_grad():
        gpt_logits, gpt_loss = model(input_ids)

    quant_model = quantize_model(model)
    
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


    # torch.save(quant_model.state_dict(), "gpt2_int8.pt")
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

