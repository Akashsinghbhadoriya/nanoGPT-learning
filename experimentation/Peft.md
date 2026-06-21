# PEFT

PEFT is an engineering framework that injects parameter-efficient methods (Lora, IA3, AdaLora, Prefix Tuning), etc. into existing transformer models.

```
Transformer
      ↓
Find target modules
      ↓
Replace with LoRALayer
      ↓
Freeze base model
      ↓
Train adapters only

sample example usage of peft

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05
)
```

Saving model in Peft is easy it only stores Lora A, Lora B, Config