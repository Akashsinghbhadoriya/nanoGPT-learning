"This is a weight only quantization approach"
import sys
from pathlib import Path

# Navigates up 4 levels from this file to reach '/Users/akashsingh/nanoGPT'
repo_root = str(Path(__file__).resolve().parents[3])

if repo_root not in sys.path:
    sys.path.append(repo_root)

from nanoGPT.model import GPT, GPTConfig
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import json

def quantize_int4(weight):

    w_max = weight.abs().max(dim=1, keepdim=True)[0]

    scales = torch.clamp(w_max / 7, min=1e-8)
    
    w_q = torch.round(weight / scales)
    
    w_q = torch.clamp(w_q, -7.0, 7.0)
    
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
    
def quantize_int4_groupwise(weight, group_size=128):
    
    out_features, in_features = weight.shape

    q_weight = torch.empty_like(weight, dtype=torch.int8)
    scales = []

    for start in range(0, in_features, group_size):

        end = min(start + group_size, in_features)

        group = weight[: , start:end]

        max_vals = group.abs().max(dim=1, keepdim=True)[0]

        scale = torch.clamp(max_vals / 7, 1e-8)

        group_q = torch.round(group / scale)

        group_q = torch.clamp(group_q, -8.0, 7.0)

        group_q = group_q.to(torch.int8)

        q_weight[:, start:end] = group_q
        scales.append(scale)
    
    scales = torch.cat(scales, dim=1)

    return q_weight, scales

def dequantize_int4_groupwise(q_weight, scales, group_size=128):

    out_features, in_features = q_weight.shape
    weight = torch.empty(q_weight.shape, dtype= torch.float32, device = q_weight.device)
    num_groups = scales.shape[1]

    for group_idx in range(num_groups):

        start = group_idx * group_size
        end = min( start + group_size, in_features)
        scale = scales[: , group_idx].unsqueeze(1)

        weight[:, start:end] = (q_weight[:, start:end].float() * scale)

    return weight

class GroupwiseINT4Linear(nn.Module):

    def __init__(self, linear, group_size=128):

        super().__init__()
        weight_q, scales = quantize_int4_groupwise(linear.weight.data, group_size)
        self.group_size = group_size
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scales", scales)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None

    def forward(self,x):

        W = dequantize_int4_groupwise(self.weight_q, self.scales, self.group_size)
        
        return F.linear(x, W, self.bias)
    
def quantize_single_layer(module, target_name, current_prefix=""):

    for name, child in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if isinstance(child, nn.Linear):

            if full_name == target_name:
                setattr(module, name, GroupwiseINT4Linear(child))
                return module
            

        else:
            quantize_single_layer(child, target_name, current_prefix=full_name)

    
    return module

def get_linear_layer_names(module, prefix=""):
    names = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            names.append(full_name)
        else:
            names.extend(get_linear_layer_names(child, full_name))
    return names


if __name__=="__main__":

    config = GPTConfig
    model = GPT.from_pretrained("gpt2")
    # torch.save(model.state_dict(), "gpt2.pt"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input = "Hi how are you?"
    input_ids = tokenizer.encode(input, return_tensors="pt")
    with torch.no_grad():
        gpt_logits, gpt_loss = model(input_ids)
    
    linear_layers = get_linear_layer_names(model)
    quantization_results = {}

    for layer_name in linear_layers:
        model_copy = copy.deepcopy(model)
        quant_model = quantize_single_layer(model_copy, target_name=layer_name)
        
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
        quantization_results[layer_name] = {
            "mse":mse.item(),
            "rmse":rmse.item(),
            "aggrement":agreement.item()
        }
    
    output_filename = "quantization_multiple_metrics.json"
    with open(output_filename, "w") as f:
        json.dump(quantization_results, f, indent=4)