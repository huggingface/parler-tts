import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class LoRALinear(nn.Module):
    """
    LoRA adapted Linear layer using only PyTorch primitives.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        r: int = 8, 
        lora_alpha: float = 16, 
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        # Store original layer parameters
        if weight is None:
            # Initialize with zeros to make it obvious if weights aren't set properly
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        else:
            self.weight = nn.Parameter(weight.clone())
            
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
        
        # LoRA specific parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA low-rank matrices
        # We use kaiming_uniform initialization per original LoRA paper
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.reset_lora_parameters()
        
        # Optional dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # For tracking active status
        self.active = True
    
    def reset_lora_parameters(self):
        """Reset LoRA parameters using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main linear operation
        result = F.linear(x, self.weight, self.bias)
        
        # Add the LoRA contribution when active
        if self.active:
            # Apply dropout to the input
            lora_x = self.lora_dropout(x)
            
            # Low-rank adaptation contribution: B·(A·x)·scaling
            lora_result = (lora_x @ self.lora_A.T) @ self.lora_B.T
            result += lora_result * self.scaling
            
        return result
    
    def set_active(self, active: bool):
        """Set whether LoRA adaptation is active."""
        self.active = active


class LoRAModuleMixin:
    """
    Mixin to add LoRA functionality to a model.
    """
    def mark_only_lora_as_trainable(self):
        """Freeze all parameters except LoRA parameters."""
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA parameters."""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_state_dict[name] = param.data.clone()
        return lora_state_dict
    
    def save_lora_weights(self, save_path: Union[str, Path]):
        """Save only LoRA weights to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        lora_state_dict = self.get_lora_state_dict()
        torch.save(lora_state_dict, save_path)
    
    def load_lora_weights(self, load_path: Union[str, Path]):
        """Load LoRA weights from disk."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"LoRA weights file {load_path} does not exist.")
        
        # map_location ensure that the LoRA weights are on the same device as the model
        lora_state_dict = torch.load(load_path, map_location=next(self.parameters()).device)
        
        # Load LoRA weights into model
        for name, param in self.named_parameters():
            if name in lora_state_dict:
                param.data.copy_(lora_state_dict[name])
    
    def set_lora_active(self, active: bool):
        """Enable or disable LoRA adaptation in the model."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.set_active(active)


def apply_lora_to_linear_layer(
    linear_layer: nn.Linear, 
    r: int = 8, 
    lora_alpha: float = 16, 
    lora_dropout: float = 0.0
) -> LoRALinear:
    """Replace a linear layer with a LoRA-adapted version."""
    in_features, out_features = linear_layer.in_features, linear_layer.out_features
    
    # Create new LoRA linear layer with the original weights and biases
    lora_layer = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        weight=linear_layer.weight.data,  # Pass the actual weights
        bias=linear_layer.bias.data if linear_layer.bias is not None else None,  # Pass the actual bias
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    return lora_layer


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0
) -> nn.Module:
    """
    Apply LoRA to specific modules in a model.
    
    Args:
        model: The model to modify
        target_modules: List of module names to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
    
    Returns:
        Modified model with LoRA layers
    """
    # Apply LoRA mixin to the model
    model.__class__ = type(
        f"{model.__class__.__name__}WithLoRA",
        (model.__class__, LoRAModuleMixin),
        {}
    )
    
    # Replace target modules with LoRA versions
    # the list is important to ensure there are no issues when replacing the modules
    for name, module in list(model.named_modules()):
        if any(target_name in name for target_name in target_modules):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else _get_submodule(model, parent_name)
            
            if isinstance(module, nn.Linear):
                lora_layer = apply_lora_to_linear_layer(
                    linear_layer=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(parent, child_name, lora_layer)
    
    # Set only LoRA parameters as trainable
    model.mark_only_lora_as_trainable()
    
    return model


def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    """Get a submodule from a model given its path."""
    if target == "":
        return model
    
    atoms = target.split(".")
    module = model
    
    for atom in atoms:
        if not hasattr(module, atom):
            raise AttributeError(f"Module {module} has no attribute {atom}")
        module = getattr(module, atom)
        
    return module 
