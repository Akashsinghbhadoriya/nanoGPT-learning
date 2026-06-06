"""
This is the code implemented by modifying the Attention mechanism to use GQA(Grouped Query Attention)
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # embedding dimension should be divisible by number of heads for evenly splitting the features
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % config.n_kv_head == 0
        assert config.n_head % config.n_kv_head == 0

        # key, query, value projections for all heads, but in a batch
        # good practice to calculate the matrix multiplication in 1 go instead of 3 different matrix
        self.q_attn = nn.Linear(config.n_embd, config.n_embd , bias=config.bias) 
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.kv_dim = config.n_embd // self.n_rep
        self.kv_attn = nn.Linear(config.n_embd, 2 * self.kv_dim, bias = config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            #shape  (1, 1, blocksize, blocksize)
            #[[1, 1, 1], [1, 1, 1], [1, 1 ,1]] => [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, T, C) @ (C , C) => (B, T, C)
        q  = self.q_attn(x)
        k, v  = self.kv_attn(x).split(self.kv_dim, dim=2) 
        print(f"q.shape: {q.shape}, v.shape: {v.shape}, k.shape: {k.shape}")

        # view to look at data in new format rather than copying to new memory
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        print(f"q.shape: {q.shape}")
        k = k.view(B, T, self.n_kv_head, self.kv_dim // self.n_kv_head) # (B, T, KVH, kvhs)
        print(f"k.shape: {k.shape}")
        k = repeat_kv(k, self.n_head // self.n_kv_head).transpose(1,2)
        print(f"k.shape after repeat: {k.shape}")
        v = v.view(B, T, self.n_kv_head, self.kv_dim // self.n_kv_head) # (B, T, KVH, kvhs)
        print(f"v.shape: {v.shape}")
        v = repeat_kv(v, self.n_head // self.n_kv_head).transpose(1,2)
        print(f"v.shape after repeat: {v.shape}")

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            #dividing by sqrt because to scale down the variance to 1
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # att*(self.bias) == 0 we set the value as -inf 
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y)) #blends the independent insights from individual heads into single unified representation
        return y

def repeat_kv(x, n_rep):
    B, T, n_kv, d = x.shape

    x = x[:,:,:,None,:]

    x = x.expand(B,T,n_kv,n_rep,d) # we use expand because it does not create a copy but uses the same vector location
    return x.reshape(B,T,n_kv * n_rep, d)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_kv_head: int = 4

if __name__ == "__main__" :

    config = GPTConfig()

    x = torch.randn(1,4,config.n_embd)
    print("input:", x)

    attn = CausalSelfAttention(config)

    output = attn(x)
