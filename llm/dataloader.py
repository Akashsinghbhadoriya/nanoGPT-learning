import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

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

def build_dataloader(config, data_dir, train_args):

    tokens = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

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