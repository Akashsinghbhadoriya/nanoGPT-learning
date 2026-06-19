from model import GPT
import torch
from pathlib import Path
from config import ModelConfig
from transformers import GPT2Tokenizer
from model import GPT
from lora import LoraModel
import os

def process_prompt(prompt, model, config, tokenizer, max_new_tokens, temperature, top_k):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens, temperature, top_k)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)

    print(f">> Reply: {output_text}")

def interact(model, config, tokenizer, max_new_tokens, temperature, top_k):
    while True:
        try:
            prompt = input(">> Prompt: ")
        
        except KeyboardInterrupt:
            break

        prompt = prompt.strip()
        if not prompt or prompt.lower() in ("!quit", "!exit"):
            break

        process_prompt(prompt, model, config, tokenizer, max_new_tokens, temperature, top_k)

@torch.inference_mode()
def main(config_path: Path, checkpoint_dir:Path = None, max_new_tokens: int = 50, temperature: float = 0.9, top_k: int | None = 50):
    print(config_path)

    config = ModelConfig.load_config(config_path)

    print(f"config: {config}")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if "gpt" in config.name.lower():
        model = GPT.from_pretrained(config.name, config)
        checkpoint_dir = Path("out/finetune/gpt2/ckpt.pt")
        if checkpoint_dir is not None and 'finetune' in checkpoint_dir.parts and os.path.exists(checkpoint_dir):
            print(f"Found checkpoint at: {checkpoint_dir}")
            checkpoint = torch.load(checkpoint_dir, map_location='cpu', weights_only=False)
            saved_config = checkpoint['config']
            saved_model_state = checkpoint['model']
            target_substrings = checkpoint['target_substrings']
            lora_rank = checkpoint['lora_rank']
            alpha = checkpoint['alpha']
            model = LoraModel(model, lora_rank, target_substrings, alpha)
            model.load_state_dict(saved_model_state, strict=False)


    else:
        out_dir = os.path.join("out",config.name)
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            print(f"Found checkpoint at: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            saved_config = checkpoint['config']
            saved_model_state = checkpoint['model']

        config = saved_config
        model = GPT(config)
        model.load_state_dict(saved_model_state)


    interact(model, config, tokenizer, max_new_tokens, temperature, top_k)


