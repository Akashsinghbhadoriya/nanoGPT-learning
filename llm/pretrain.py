from pathlib import Path
from config import ModelConfig, TrainArgs
from model import GPT


def pretrain(config_path: Path, train_config_path: Path):

    config = ModelConfig.load_config(config_path)
    train_args = TrainArgs.load_config(train_config_path)

    model = GPT(config)

    train_loader = build_dataloader(config)

    