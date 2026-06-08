from model import GPT
import torch
from pathlib import Path
from config import ModelConfig
from transformers import GPT2Tokenizer
from model import GPT

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
def main(config_path: Path, max_new_tokens: int = 50, temperature: float = 0.9, top_k: int | None = 50):
    print(config_path)

    config = ModelConfig.load_config(config_path)

    print(f"config: {config}")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT(config)

    interact(model, config, tokenizer, max_new_tokens, temperature, top_k)


