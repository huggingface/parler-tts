import torch.nn as nn
import torch
from tqdm import tqdm 
import math 

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.linear = linear_layer
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.lora_A = nn.Linear(linear_layer.in_features, lora_r, bias=False)
        self.lora_B = nn.Linear(lora_r, linear_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) # following microsoft/LoRA
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = self.lora_alpha / self.lora_r
        self.linear.requires_grad_(False)

    def forward(self, x):
        out = self.linear(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out

def replace_linear_with_lora(model, lora_r, lora_alpha, lora_dropout):
    """
    Given a model, replaces all linear layers with a Linear LORA layer in-place. Returns model
    """
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    for total_len, _ in enumerate(model.named_modules()):
        pass

    i = 0
    for name, module in tqdm(model.named_modules(), total=total_len, desc='Replacing Linear with Low-Rank Layers', mininterval=5):
        if any(item in name for item in ['embed_prompts', 'lm_heads']):
        	print('Ignored adding peft to ', name)

        elif module in linear_info:
            info = linear_info[module]
            new_module = LoRALinear(module, lora_r, lora_alpha, lora_dropout)
            setattr(info["father"], info["name"], new_module)

            del linear_info[module]
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    print('Replaced linear layers with low-rank layers.')
    return model

