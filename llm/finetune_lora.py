from pathlib import Path
from config import ModelConfig, FinetuneArgs
from model import GPT
from dataloader import build_dataloader
from trainer import Trainer
import os
from lora import LoraModel

def build_optimizer(model, train_args, device):

    optimizer = model.configure_optimizers(
        train_args.weight_decay, 
        train_args.learning_rate, 
        (train_args.beta1, train_args.beta2),
        device
    )

    return optimizer

def finetune(
        config_path: Path, 
        train_config_path: Path,
        lora_r: int = 16,
        target_substrings: list = ["c_attn", "c_proj"]
    ):

    config = ModelConfig.load_config(config_path)
    train_args = FinetuneArgs.load_config(train_config_path)

    device = train_args.device

    model = LoraModel(GPT.from_pretrained("gpt2", config), lora_r, target_substrings)

    data_dir = os.path.join('../data/', train_args.dataset)
    train_loader = build_dataloader(config, data_dir, train_args, 'train')
    val_loader = build_dataloader(config, data_dir, train_args, 'val')

    optimizer = build_optimizer(model, train_args, device)

    trainer = Trainer(model,config , optimizer, train_loader, val_loader, train_args, device, is_lora=True)

    trainer.fit(lora_r, target_substrings)

    