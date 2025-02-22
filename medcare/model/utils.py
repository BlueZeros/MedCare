import torch 
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft.utils import _get_submodules
# from peft.tuners.lora import mark_only_lora_as_trainable
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
from safetensors import safe_open
import warnings

local_rank = None

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def mark_only_lora_as_trainable(model: nn.Module, bias, freeze_base_experts=False) -> None:

    if freeze_base_experts:
        print("Freeze Share Expert!")

    for n, p in model.named_parameters():
        if "lora" not in n and "switch" not in n:
            p.requires_grad = False
        if freeze_base_experts and "lora" in n and ".base." in n:
            p.requires_grad = False

    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

def mark_only_orthlora_as_trainable(model: nn.Module, bias) -> None:
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False
        elif "lora" in n and "base" in n:
            p.requires_grad = False
        elif "lora" in n and "orth" in n:
            p.requires_grad = True

    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

def check_target_module_exists(lora_config, key, target_modules):
    target_module_found = any(key.endswith(module_name) for module_name in target_modules)
    return target_module_found

def create_mixoflora_module(lora_config, target, model_args, add_bias=True):
    in_features, out_features = target.in_features, target.out_features
    # print(lora_config)
    new_module = MoLoRALinear(in_features, out_features, model_args.num_experts, model_args.num_experts_per_token,
                              r=lora_config.r,
                              lora_alpha=lora_config.lora_alpha,
                              lora_dropout=lora_config.lora_dropout,
                              use_rslora=lora_config.use_rslora,
                              share_expert=model_args.share_expert,
                              num_share_experts=model_args.num_share_experts,
                              bias=add_bias)
    return new_module

def create_orthlora_module(lora_config, target, model_args, attn_orth=False, add_bias=True):
    in_features, out_features = target.in_features, target.out_features
    new_module = OrthLoRALinear(in_features, out_features, r=lora_config.r if not attn_orth else lora_config.r//2,
                                lora_alpha=lora_config.lora_alpha,
                                lora_dropout=lora_config.lora_dropout,
                                use_rslora=lora_config.use_rslora,
                                bias=add_bias)
    return new_module

def get_orthlora_model(model, model_args, lora_config, decoder_type=Qwen2DecoderLayer, lora_name_or_path=None, inference_mode=False):
    
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type):
                if getattr(model_args, "wrap_ffn_orth", True) and "mlp" in name:
                    names = name.split(".")
                    target_module_names.add(names[0] if len(names) == 1 else names[-1])
                
                if getattr(model_args, "wrap_attn_orth", False) and "self_attn" in name:
                    names = name.split(".")
                    target_module_names.add(names[0] if len(names) == 1 else names[-1])

    target_module_names = list(target_module_names)
    
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False

        new_module = create_orthlora_module(lora_config, target, model_args, attn_orth=True if "mlp" not in key else False, add_bias=add_bias)
        setattr(parent, target_name, new_module)
        new_module.weight = target.weight 
        if hasattr(target, "bias"):
            if target.bias is not None:
                new_module.bias = target.bias

        new_module.to(target.weight.device)
        
        if getattr(target, "state", None) is not None:
            new_module.state = target.state
            new_module.to(target.weight.device)
        
        del target
    
    if lora_name_or_path is not None and os.path.exists(lora_name_or_path):
        shared_lora_params = {}

        if getattr(model_args, "wrap_attn_orth", False):
            shared_attn_lora_params = {}
            print(f"Initializing attention base LoRA weights from attention LoRA in {os.path.join(lora_name_or_path, 'adapter_model.safetensors')}")
            with safe_open(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), framework="pt", device=0) as f:
                for k in f.keys():
                    shared_attn_lora_params[k] = f.get_tensor(k)
            shared_attn_lora_params = {(k[11:] if k.startswith('base_model.') else k): v for k, v in shared_attn_lora_params.items()}
            if any(k.startswith('model.model.') for k in shared_attn_lora_params):
                shared_attn_lora_params = {(k[6:] if k.startswith('model.') else k): v for k, v in shared_attn_lora_params.items()}
            shared_lora_params.update(shared_attn_lora_params)

        if getattr(model_args, "wrap_ffn_orth", True):
            print(f"Initializing mlp base LoRA weights from share_experts in {os.path.join(lora_name_or_path, 'non_lora_trainables.bin')}")
            shared_ffn_lora_params = torch.load(os.path.join(lora_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')
            shared_ffn_lora_params = {(k[11:] if k.startswith('base_model.') else k): v for k, v in shared_ffn_lora_params.items()}
            if any(k.startswith('model.model.') for k in shared_ffn_lora_params):
                shared_ffn_lora_params = {(k[6:] if k.startswith('model.') else k): v for k, v in shared_ffn_lora_params.items()}
            shared_lora_params.update(shared_ffn_lora_params)

        initial_base_params = {}
        for k in shared_lora_params:
            if "mlp" in k and "lora" in k and "share_experts" in k:
                initial_base_params[k.replace("share_experts.lora_A", "base.lora_A").replace(
                    "share_experts.lora_B", "base.lora_B"
                )] = shared_lora_params[k]
            elif "self_attn" in k and "lora" in k:
                initial_base_params[k.replace("lora_A", "base.lora_A").replace("lora_B", "base.lora_B")] = shared_lora_params[k]
            elif "mlp" in k and ".experts." in k:
                continue 
            elif "switch" in k:
                continue
            else:
                rank0_print(k)
                exit(-1)

        incompatible_keys = model.load_state_dict(initial_base_params, strict=False)
        print(incompatible_keys)
        # print(incompatible_keys)
    else:
        print("No initialization for Base!")

    mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"), getattr(model_args, "freeze_base_experts", False))
    if inference_mode:
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model


def merge_and_unload(model):
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if isinstance(target, LoRALayer):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            target.merge()
            replace_module(parent, target_name, new_module, target)



    return model

def replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state

    new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)
            
def get_mixoflora_model(model, model_args, lora_config, decoder_type=Qwen2DecoderLayer, lora_name_or_path=None, lora_loading_type=None, inference_mode=False):
    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and "mlp" in name:
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False

        new_module = create_mixoflora_module(lora_config, target, model_args, add_bias=add_bias)
        replace_module(parent, target_name, new_module, target)
        
        del target
    
    print(lora_name_or_path)
    loading_params = {}
    if lora_name_or_path is not None and os.path.exists(lora_name_or_path):
        if lora_loading_type == "all":
            print(f"Initializing attn LoRA from {os.path.join(lora_name_or_path, 'adapter_model.safetensors')}")
            # non_lora_trainables = torch.load(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), map_location='cpu')
            with safe_open(os.path.join(lora_name_or_path, 'adapter_model.safetensors'), framework="pt", device=0) as f:
                for k in f.keys():
                    loading_params[k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")] = f.get_tensor(k)

            print(f"Initializing MoLoRA from all experts of {os.path.join(lora_name_or_path, 'non_lora_trainables.bin')}")
            non_lora_trainables = torch.load(os.path.join(lora_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')
            # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            loading_params.update(non_lora_trainables)
    else:
        print("No initialization for MoLoRA & LoRA!")
    
    incompatible_keys = model.load_state_dict(loading_params, strict=False)
    print(incompatible_keys)
    if len(incompatible_keys.unexpected_keys) != 0:
        print(incompatible_keys)

    mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"))
    if inference_mode:
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRAModule, self).__init__()
        self.lora_a = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_b = nn.Parameter(torch.zeros((out_features, r)))
        self.reset_parameters()

    def forward(self):
        return self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

class OrthLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        use_rslora: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.fan_in_fan_out = fan_in_fan_out

        self.use_rslora = use_rslora
        nn.Linear.reset_parameters(self)
        
        if r > 0:
            # print(r)
            self.base = nn.ModuleDict({"lora_A": nn.Linear(in_features, 2 * r, False, dtype=torch.float32), "lora_B": nn.Linear(2 * r, out_features, False, dtype=torch.float32)})
            self.orth = nn.ModuleDict({"lora_A": nn.Linear(in_features, 2 * r, False, dtype=torch.float32), "lora_B": nn.Linear(2 * r, out_features, False, dtype=torch.float32)})
            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)
            self.weight.requires_grad = False
            
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
        if init_lora_weights:
            self.reset_lora_parameters()
        self.to(self.weight.device)
            
    def reset_lora_parameters(self):
        # nn.Linear.reset_parameters(self)
     
        if hasattr(self, 'base'):
            nn.init.kaiming_uniform_(self.base[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.base['lora_B'].weight)
        
        if hasattr(self, 'orth'):
            nn.init.kaiming_uniform_(self.orth[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.orth[f'lora_B'].weight)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            x = x.to(self.base['lora_A'].weight.dtype)
            x = self.lora_dropout(x)
            result += (
                self.base['lora_B'](
                    self.base['lora_A'](x)
                )
                * self.scaling
            )
            
            # modified 
            result += (
                self.orth['lora_B'](
                    self.orth['lora_A'](x)
                )
                * self.scaling
            )

        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
        result = result.to(previous_dtype)
        
        return result
    
    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return 
        if self.r > 0:
            self.weight.data += (
                transpose(
                    self.base['lora_B'].weight @ self.base['lora_A'].weight,
                    self.fan_in_fan_out
                ) * self.scaling + \
                    transpose(
                        self.orth['lora_B'].weight @ self.orth['lora_A'].weight,
                        self.fan_in_fan_out
                    ) * self.scaling 
            )
            self.merged = True
        

class MoLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        share_expert: bool = False,
        num_share_experts: int = 1, 
        use_rslora: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # moe parameters
        self.num_experts = num_experts 
        self.num_experts_per_token = num_experts_per_token
        self.share_expert = share_expert
        self.use_rslora = use_rslora

        if num_experts > 1:
            self.switch = nn.Linear(in_features, num_experts)
        
        # Actual trainable parameters
        if r > 0:
            self.experts = nn.ModuleList([
                nn.ModuleDict({"lora_A_{}".format(i): nn.Linear(in_features, r, False, dtype=torch.float32),
                               "lora_B_{}".format(i): nn.Linear(r, out_features, False, dtype=torch.float32)})
            for i in range(num_experts)])

            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)
            self.weight.requires_grad = False

            if self.share_expert:
                self.share_experts = nn.ModuleDict({"lora_A": nn.Linear(in_features, r*num_share_experts, False, dtype=torch.float32),
                        "lora_B": nn.Linear(r*num_share_experts, out_features, False, dtype=torch.float32)})
        
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
    
        if hasattr(self, 'experts'):
            for idx, expert in enumerate(self.experts):
                nn.init.kaiming_uniform_(expert[f'lora_A_{idx}'].weight, a=math.sqrt(5))
                nn.init.zeros_(expert[f'lora_B_{idx}'].weight)
        
        if hasattr(self, 'share_experts'):
            nn.init.kaiming_uniform_(self.share_experts[f'lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(self.share_experts[f'lora_B'].weight)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)


    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            moe_result = self.molora_helper2(x) if self.training else self.molora_helper(x)
            result += moe_result
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    
    def molora_helper2(self, x: torch.Tensor):
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
            
        previous_dtype = x.dtype 
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        if self.share_expert:
            share_result = self.share_experts[f'lora_B'](self.share_experts[f'lora_A'](x)) * self.scaling # [bs * N, out_features]

        gate_logits = self.switch(x)  # [bs * N, expert]

        temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts)], dim=0)  # [expert, bs * N, out_features]
        temp_results = temp_results.transpose(0, 1)  # [bs * N, expert, out_features]

        gate_probs = gate_logits.softmax(-1)
        # print(gate_lprobs)
        weights, selected_experts = torch.topk(gate_probs, self.num_experts_per_token)
        weights /= weights.sum(dim=-1, keepdim=True)

        selected_results = temp_results.gather(1, selected_experts.unsqueeze(-1).expand(-1, -1, self.out_features))  # [bs * N, select_expert, out_features]
        assert selected_results.shape == (batch_size * N, self.num_experts_per_token, self.out_features)
        if self.share_expert:
            weights = torch.cat([weights, torch.ones(weights.shape[0], 1).to(weights)], dim=-1)
            selected_results = torch.cat([
                selected_results,
                share_result.unsqueeze(1)
            ], dim=1)

        # weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.einsum("be, bef -> bf", weights, selected_results)
        results = results.contiguous().view(batch_size, N, -1)
        results = results.to(previous_dtype)
        return results
    
    def molora_helper(self, x: torch.Tensor):
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
            
        previous_dtype = x.dtype 
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        if self.share_expert:
            share_result = self.share_experts[f'lora_B'](self.share_experts[f'lora_A'](x)) * self.scaling # [bs * N, out_features]

        gate_logits = self.switch(x)  # [bs * N, expert]

        temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts)], dim=0)  # [expert, bs * N, out_features]
        temp_results = temp_results.transpose(0, 1)  # [bs * N, expert, out_features]

        gate_probs = gate_logits.softmax(-1)
        weights, selected_experts = torch.topk(gate_probs, self.num_experts_per_token)

        weights /= weights.sum(dim=-1, keepdim=True)

        selected_results = temp_results.gather(1, selected_experts.unsqueeze(-1).expand(-1, -1, self.out_features))  # [bs * N, select_expert, out_features]
        assert selected_results.shape == (batch_size * N, self.num_experts_per_token, self.out_features)
        if self.share_expert:
            weights = torch.cat([weights, torch.ones(weights.shape[0], 1).to(weights)], dim=-1)
            selected_results = torch.cat([
                selected_results,
                share_result.unsqueeze(1)
            ], dim=1)

        # weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.einsum("be, bef -> bf", weights, selected_results)
        results = results.contiguous().view(batch_size, N, -1)
        results = results.to(previous_dtype)
        return results