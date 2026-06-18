from model import GPTConfig, GPT
import torch
import torch.nn as nn
from torch.nn import functional as F

class ReplaceLora(nn.Module):

    def __init__(self, layer, rank = 4):
        super().__init__()
        self.original_layer = layer

        self.original_layer.weight.requires_grad = False #freezing the original weights
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        out, inp = layer.weight.shape

        self.a_lora = nn.Linear(inp, rank, bias=False)
        self.b_lora = nn.Linear(rank, out, bias=False)
        
        nn.init.kaiming_uniform_(self.a_lora.weight, a=5**0.5)
        nn.init.zeros_(self.b_lora.weight) #initializing b to zero so that the pretrained model is not impacted

    def forward(self, x):

        original_output = self.original_layer(x)
        lora_output = self.b_lora(self.a_lora(x))

        return original_output + lora_output


def lora_adapters(model, target_modules):

    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules):
            parts = name.split('.')
            sub_name = parts[-1]
            parent_path = parts[:-1]
            
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
                
            # PASS THE OLD MODULE IN: This keeps the original weights!
            setattr(parent, sub_name, ReplaceLora(module, rank=4))

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

def LoraModel():

    config = GPTConfig()
    model = GPT.from_pretrained("gpt2")
    all_layers = get_linear_layer_names(model)
    target_substrings = ["c_attn"]
    target_names = [target for target in all_layers if any(sub in target for sub in target_substrings)]
    print(target_names)
    loral_model = lora_adapters(model, target_names)
    print(loral_model)

if __name__=="__main__":

    LoraModel()