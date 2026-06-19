import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, index):
        x = torch.tensor(self.tokens[index: index + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[index + 1 : index + 1 + self.block_size], dtype=torch.long)

        return x, y

class InstructionDataset(Dataset):
    def __init__(self, input_ids, labels, max_length, block_size):
        assert max_length <= block_size
        self.input_ids = input_ids
        self.labels = labels
        self.block_size = block_size
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids) // self.max_length
    
    def __getitem__(self, index):
        start_idx = index * self.max_length
        end_idx = start_idx + self.max_length
        x = torch.tensor(self.input_ids[start_idx: end_idx], dtype=torch.long)
        y = torch.tensor(self.labels[start_idx : end_idx], dtype=torch.long)

        return x, y

def build_dataloader(config, data_dir, train_args, type):

    if type == "train":
        tokens = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif type == "val":
        tokens = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    dataset = TokenDataset(
        tokens = tokens,
        block_size= config.block_size
        )
    
    return DataLoader(
        dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=train_args.num_workers,
        drop_last=True,
    )

def build_instruction_dataloader(config, data_dir, train_args, type):

    if type == "train":
        input_ids = np.memmap(os.path.join(data_dir, 'input_train.bin'), dtype=np.int32, mode='r')
        labels = np.memmap(os.path.join(data_dir, 'label_train.bin'), dtype=np.int32, mode='r')

        with open(os.path.join(data_dir, 'train_dataset_config.json'), "r") as f:
            train_config = json.load(f)
    elif type == "val":
        input_ids = np.memmap(os.path.join(data_dir, 'input_val.bin'), dtype=np.int32, mode='r')
        labels = np.memmap(os.path.join(data_dir, 'label_val.bin'), dtype=np.int32, mode='r')

        with open(os.path.join(data_dir, 'val_dataset_config.json'), "r") as f:
            train_config = json.load(f)

    dataset = InstructionDataset(
        input_ids= input_ids,
        labels= labels,
        max_length = train_config["max_length"],
        block_size= config.block_size
        )
    
    return DataLoader(
        dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=train_args.num_workers,
        drop_last=True,
    )