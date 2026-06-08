from dataclasses import dataclass
import yaml
from pathlib import Path

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

    @classmethod
    def load_config(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        return cls(**config)