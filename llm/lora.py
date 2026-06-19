from model import GPTConfig, GPT
import torch
import torch.nn as nn
from torch.nn import functional as F

class ReplaceLora(nn.Module):

    def __init__(self, layer, rank = 8, alpha = 16):
        super().__init__()
        self.original_layer = layer
        out, inp = layer.weight.shape
        self.scaling = alpha / rank

        self.a_lora = nn.Linear(inp, rank, bias=False)
        self.b_lora = nn.Linear(rank, out, bias=False)
        
        nn.init.kaiming_uniform_(self.a_lora.weight, a=5**0.5)
        nn.init.zeros_(self.b_lora.weight) 

    def forward(self, x):

        original_output = self.original_layer(x)
        lora_output = self.b_lora(self.a_lora(x))

        return original_output + self.scaling * lora_output


def lora_adapters(model, target_modules, lora_r, alpha):

    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules):
            parts = name.split('.')
            sub_name = parts[-1]
            parent_path = parts[:-1]
            
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
                
            # PASS THE OLD MODULE IN: This keeps the original weights!
            setattr(parent, sub_name, ReplaceLora(module, rank=lora_r, alpha=alpha))

    return model
        

def get_linear_layer_names(module, prefix=""):
    names = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            names.append(full_name)
        else:
            names.extend(get_linear_layer_names(child, full_name))
    return names

def freeze_parameters(model):
    for name, param in model.named_parameters():
        
        if "_lora" not in name:
            param.requires_grad = False

def LoraModel(model, lora_r, target_substrings, alpha):

    all_layers = get_linear_layer_names(model)
    target_names = [target for target in all_layers if any(sub in target for sub in target_substrings)]
    
    lora_model = lora_adapters(model, target_names, lora_r, alpha)
    freeze_parameters(lora_model)
    return lora_model
