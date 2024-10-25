import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.linear = linear_layer
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.lora_A = nn.Linear(linear_layer.in_features, lora_r, bias=False)
        self.lora_B = nn.Linear(lora_r, linear_layer.out_features, bias=False)
        self.scaling = self.lora_alpha / self.lora_r
        
    def forward(self, x):
        x = self.lora_dropout(x)
        x = self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling
        return x

def replace_linear_with_lora(model, lora_r, lora_alpha, lora_dropout):
    for name, module in model.named_modules():
        if any(item in name for item in ['embed_prompts', 'lm_heads']):
            print('Ignored adding peft to ', name)
            continue

        if isinstance(module, nn.Linear):
            lora_linear = LoRALinear(module, lora_r, lora_alpha, lora_dropout)
            setattr(model, name, lora_linear)
    return model

def set_non_lora_gradients_to_false(model):
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

        if 'lm_heads' in name or 'embed_prompts' in name:
            param.requires_grad = True 
            print("Using gradients for lm_heads or embed_prompts", name)
    return model