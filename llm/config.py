from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
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

    @classmethod
    def load_config(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(**config)

@dataclass  
class TrainArgs:
    out_dir: str
    eval_interval: int
    log_interval: int
    eval_iters: int
    dataset: str
    learning_rate: float
    max_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float
    device: str
    num_workers: int
    batch_size: int

    @classmethod
    def load_config(cls, train_config_path: Path):
        with open(train_config_path, "r") as f:
            train = yaml.safe_load(f)

        return cls(**train)