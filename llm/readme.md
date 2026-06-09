# nanoGPT Learning

A modular transformer implementation inspired by nanoGPT, built to understand and experiment with modern LLM architectures and inference optimizations.

## Features

### Core Transformer

* GPT-style decoder-only architecture
* Causal self-attention
* Multi-head attention
* Feed-forward networks
* Residual connections

### Modern LLM Components

* Rotary Position Embeddings (RoPE)
* RMSNorm
* Grouped Query Attention (GQA)
* KV Cache for efficient autoregressive inference
* SwiGLU activation
* Config-driven model construction

### Quantization

* INT8 quantization experiments
* Per-tensor quantization
* Per-channel quantization
* Quantization benchmarking and analysis

### Configuration System

* YAML-based model configuration
* CLI-driven execution
* Easy experimentation with different architectures
* Feature toggles through configuration files

### Inference

* Text generation
* Interactive chat interface
* Streaming token generation
* KV cache optimized decoding

## Project Structure

```text
.
├── cli.py
├── model.py
├── chat.py
├── configs/
│   ├── small.yaml
│   └── ...
├── checkpoints/
├── experiments/
└── README.md
```

## Running

### Chat

```bash
python cli.py chat configs/small.yaml
```

### Configuration Example

```yaml
block_size: 1024
vocab_size: 50304

n_layer: 12
n_head: 12
n_embd: 768

rope: true
rmsnorm: true
gqa: true
```

## Purpose

The goal of this repository is not only to reproduce transformer architectures from scratch, but also to progressively evolve the codebase toward a modular, configurable, and production-inspired LLM framework while maintaining educational clarity.
# Future Scope

1. Adding train.py for training model based on different config and datasets.
2. Adding top-p in the generate function.
3. Streaming output tokens one by one.
4. Adding a UI based chat to chat with models.
5. Building the Visualizer for the Transformer architecture.