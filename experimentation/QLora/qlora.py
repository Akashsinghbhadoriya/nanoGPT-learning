"""
QLora Implementation using PEFT Library 
Run in Google Colab will fail in MAC because MPS has issue in supporting bitsandbytes
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,  
    AutoTokenizer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer

MODEL_NAME="openai-community/gpt2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
print(model)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"]
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

dataset = load_dataset("yahma/alpaca-cleaned")

def formatting_func(example):

    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['output']}"
    )

training_args = TrainingArguments(
    output_dir="./qlora_gpt2",
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 8,
    learning_rate=2e-4,
    num_train_epochs= 1,
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit", # using paged optimizers for QLora
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,
    args=training_args
)

trainer.train()

model.save_pretrained("./qlora_gpt2_adapter")
tokenizer.save_pretrained("./qlora_gpt2_adapter")
