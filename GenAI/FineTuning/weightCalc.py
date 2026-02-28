import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# load gpt2
model = AutoModelForCausalLM.from_pretrained("gpt2")

# We define loRA Config
config = LoraConfig(
    r=8, # rank of the low-rank matrices
    lora_alpha=16, # scaling factor
    target_modules=["c_attn"], # gpt2 uses c_attn for attention rather than qkv
    lora_dropout=0.05, # dropout probability
    bias="none", # bias type
)

# Apply LoRA to the model
lora_model = get_peft_model(model, config)

# Original weight (W)
w_original = lora_model.transformer.h[0].attn.c_attn.weight
print("Original weight shape:", w_original.shape)

# loRA Matrices (A and B)
A = lora_model.transformer.h[0].attn.c_attn.lora_A["default"].weight
B = lora_model.transformer.h[0].attn.c_attn.lora_B["default"].weight

print("LoRA A weight shape:", A.shape)
print("LoRA B weight shape:", B.shape)

# We wll calculate Delta W = W + BA
alpha = config.lora_alpha
r = config.r
Delta_W = (alpha/r) * (B @ A)

print("Delta W shape:", Delta_W.shape)

# Compute Final Effecitive weight
w_effective = w_original + Delta_W.T

print("Effective weight shape:", w_effective.shape)

# Compare original and effective weights
print("Original weight:", w_original)
print("Effective weight:", w_effective)