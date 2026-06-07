"""
Based on NanoGPT implementation with improved funtionality of Rope, RMSNorm, KVCache, GQA, config driven approach
"""

import math
import inspect
from dataclasses import dataclass
from transformers import GPT2Tokenizer
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# =====================
# LayerNorm
# =====================
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim)) # this acts as the scale parameter gamma defaults to float.32
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None #this acts a shift parameter or beta and bias if true then we initiallize with zero

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) #uses pytorch function for faster calculation when training models 

# =====================
# RMSNorm
# ===================== 
class RMSNorm(nn.Module):
    """Root Mean Squared normalization """
    def __init__(self, ndim, p = -1., eps = 1e-8, bias=False):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(ndim))
        self.ndim = ndim
        self.p = p
        self.eps = eps
        self.bias = bias
        self.offset = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        if self.p < 0. or self.p > 1. :
            # 2 means L2 normalization
            norm_input = input.norm(2, dim=-1, keepdim=True) # provides a normalized output sqrt(x1^2 + x2^2 + .... + xn^2) (B, T, C)-> (B, T, 1)
            n = self.ndim
        else:
            partial_size = int(self.ndim * self.p)
            partial_input, _ = torch.split(input, [partial_size, self.ndim - partial_size], dim=-1)

            norm_input = partial_input.norm(2, dim=-1, keepdim=True)

            n = partial_size
        
        
        rms_norm = norm_input / (n ** (0.5))
        
        input_rms_norm = input / (rms_norm + self.eps)

        if self.bias :
            return self.scale * input_rms_norm + self.offset
        
        return self.scale * input_rms_norm
    
# =====================
# Rope
# =====================
class RotaryEmbedding(nn.Module):
    def __init__ (self, dim, max_position_embedding = 4096, base = 10000):
        super().__init__()
        self.dim = dim

        # inverse frequency for all the pairs of columns along the dimension
        inv_freq = 1 / (torch.pow( base , torch.arange(0, dim, 2).float() / dim)) # (dim/2) 1D
        # persistent prevents the buffer from being saved in the model state dict
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        #precompute the positional angles upto max context length
        t = torch.arange(max_position_embedding, dtype = torch.float32) #(pos_emb) 1D

        # takes 2 1D matrix and create a 2D matrix where every value of 1 vector is multiplied with every value of the 2nd vector
        freqs = torch.outer(t, self.inv_freq) #(pos_emb, dim/2)

        #duplicate frequencies to match the dimensions head
        emb = torch.cat((freqs, freqs), dim=-1) #(pos_emb, dim)

        # Cache Cosine and Sine values (unsqueezed for batch/head broadcasting)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent = False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent = False)

    def forward(self, q, k, seq_len):
        cos, sin = self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]
        q, k = self.apply_rotary_pos_emb(q, k, sin, cos)
        return q, k
    
    def rotate_half(self, x):
        # split the tensor along the last dimension and rotates the half.
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, sin, cos):
        # rotary formula q = q * cos + rotate_half(q) * sin
        q_rotated = q * cos + self.rotate_half(q) * sin # dimensions (b, nh, T, hs) * (:,:,T, hs) + (b, nh, T, hs) * (:,:,T, hs)
        k_rotated = k * cos + self.rotate_half(k) * sin

        return q_rotated, k_rotated

# =====================
# Grouped Query Attention
# =====================   
class GQA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

    def repeat_kv(self,x, n_rep):
        B, n_kv, T, d = x.shape

        x = x[:,:,None,:,:]

        x = x.expand(B,n_kv,n_rep,T,d) # we use expand because it does not create a copy but uses the same vector location
        return x.reshape(B,n_kv * n_rep,T, d)

# =====================
# KV Cache
# =====================   
class KVCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('cache_k', None, persistent = False)
        self.register_buffer('cache_v', None, persistent = False)

    def reset_cache(self) :
        self.cache_v = None
        self.cache_k = None

    def forward(self, k, v):
        if self.cache_k is None:
            self.cache_k = k
            self.cache_v = v
        else :
            self.cache_k = torch.cat([self.cache_k, k], dim = 2)
            self.cache_v = torch.cat([self.cache_v, v], dim = 2)
        k , v = self.cache_k, self.cache_v
        return k , v

# =====================
# Multi head Causal Attention
# =====================
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.use_rope = config.use_rope
        self.use_kvcache = config.use_kvcache
        self.use_gqa = config.use_gqa

        self.config = config

        self.head_dim = config.n_embd // config.n_head
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, config.block_size, config.base)
        if self.use_gqa:
            self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.kv_attn = nn.Linear(config.n_embd, 2 * self.head_dim * config.n_kv_head, bias = config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        if self.use_kvcache:
            self.kv_cache = KVCache(self.config)

        if self.use_gqa:
            self.gqa = GQA()

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def reset_cache(self) :
        self.kv_cache.reset_cache()

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if self.use_gqa:
            q  = self.c_attn(x)
            k, v  = self.kv_attn(x).split((self.head_dim * self.n_kv_head), dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)
            v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)

            if self.use_rope: q, k = self.rope(q, k, T)
            if self.use_kvcache : k, v = self.kv_cache(k, v)
            n_rep = self.n_head // self.n_kv_head
            k = self.gqa.repeat_kv(k, n_rep) 
            v = self.gqa.repeat_kv(v, n_rep)
        else:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) 
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            if self.use_rope: q, k = self.rope(q, k, T)
            if self.use_kvcache : k, v = self.kv_cache(k, v)
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# =====================
# MLP Feed Forward Network
# =====================
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# =====================
# SwiGLU
# =====================    
class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.w1 = nn.Linear(config.n_embd, config.hidden_dim, bias = config.bias)
        self.w2 = nn.Linear(config.n_embd, config.hidden_dim, bias = config.bias)
        self.w3 = nn.Linear(config.hidden_dim, config.n_embd, bias = config.bias)

    def forward(self, x):
        gate = F.silu(self.w1(x)) # (B, T, C) * (C, HD) -> (B, T, HD)
        value = self.w2(x) # (B, T, C) * (C, HD) -> (B, T, HD)

        x = self.w3(gate * value) # (B, T, HD) * (HD, C) -> (B, T, C)
        return x

# =====================
# Transformer block
# =====================
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd, config.p, config.eps, config.bias)
            self.ln_2 = RMSNorm(config.n_embd, config.p, config.eps, config.bias)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)
        if config.use_swiglu:
            self.mlp = SwiGLU(config)
        else:
            self.mlp = MLP(config)

    def reset_cache(self):
        self.attn.reset_cache()

    def forward(self, x):
        #shortcut connections or residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# =====================
# GPT Config
# =====================
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# =====================
# Model Config
# =====================
@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int 
    n_embd: int
    dropout: float
    bias: bool
    use_rope: bool
    use_kvcache: bool
    use_rmsnorm: bool
    use_gqa: bool
    use_swiglu: bool
    base: int
    p: float #used for partial rms norm
    eps: float
    hidden_dim: int

# =====================
# GPT Model
# =====================
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.current_pos = 0

        # complete transformer block of gpt
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        if not self.config.use_rope:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        
        if self.config.use_rmsnorm :
            self.transformer['ln_f'] = RMSNorm(config.n_embd, p = config.p, eps= config.eps, bias=config.bias)
        else:
            self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # custom weight initialization method used before training.
    # It sets a stable starting variance for linear and embedding layers like token and position
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # bias to zero to avoid initial deviation
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_kv_cache(self):
        self.current_pos = 0
        for block in self.transformer.h :
            block.reset_cache()

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # sequence length t should be less than the block size which is our context length of the model
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        if not self.config.use_rope:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) #layer normalization after the transformer block

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # cross entropy loss is the negative loss probability calculation comparing the predicted values with target values
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) # updating the positional embedding layer
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size] # reduce the bias size for mask filling to fit the new context length

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # we discard it since it is not learnable parameter during fine-tuning and can break weight optimizers
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        # loads a pretrained gpt model from huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape #sd_hf[k].shape[::-1] reverses the array dimension using slicing
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def memory_usage_kv_cache(self):
        total_bytes = 0
        for block in self.transformer.h :
            total_bytes += block.attn.kv_cache.cache_k.numel() * block.attn.kv_cache.cache_k.element_size()
            total_bytes += block.attn.kv_cache.cache_v.numel() * block.attn.kv_cache.cache_v.element_size()

        return total_bytes / 1024**2

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # weight decay reduces model overfitting by penalising large weights
        # we weight decay only learnable parameters which are embedding and matmuls
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused is used for speeding up the optimization
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        times = []
        memory = 0

        for i in range(max_new_tokens):

            if i == 0 and self.config.use_kvcache:
                self.reset_kv_cache()
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            elif i > 0 and self.config.use_kvcache:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            start = time.time()
            # forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits, _ = self(idx_cond)
            end = time.time()
            times.append((end - start))
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        if self.config.use_kvcache: 
            memory = self.memory_usage_kv_cache()

        return idx, memory, sum(times) / len(times)

    
if __name__=="__main__":

    config = ModelConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_kv_head=1,
        n_embd=768,
        dropout=0.0,
        bias=True,
        use_rope=True,
        use_kvcache=True,
        use_rmsnorm=True,
        use_gqa=True,
        use_swiglu=False,
        hidden_dim=2048,
        base=10000,
        p=-1,
        eps=1e-8
    )
    input_text = "capital of india is"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_token_ids = tokenizer.encode(input_text, return_tensors="pt")
    print("input tokens:", input_token_ids)

    model = GPT(config)

    benchmark_kv = []
    for x in range(10, 511, 50):
        print(f"Generating for max new tokens:{x}")
        max_new_tokens = x
        start_time = time.time()
        output_tokens, memory, latency = model.generate(input_token_ids, max_new_tokens = max_new_tokens, temperature=1)
        end_time = time.time()
        latency = latency * 1000 #converting to ms
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens = True)
        total_time = end_time - start_time
        print(f"total_time for {x} token is {total_time*1000}ms and latency is {latency}ms with memory usage {memory}")
        total_tokens_sec = max_new_tokens / total_time
        print(f"tokens per sec is : {total_tokens_sec}")
        benchmark_kv.append({"token_generated":x, "tokens_per_sec": total_tokens_sec, "total_time": total_time, "memory": memory, "latency": latency})
    
    with open("MQA_benchmark_results_kv.json", "w") as file:
        json.dump(benchmark_kv, file, indent=4)
