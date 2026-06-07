# nanoGPT Learning Repository

## Overview

This repository is my structured study of Andrej Karpathy's nanoGPT implementation.

The objective is not only to run the code but to understand:

* Transformer architecture implementation
* Attention mechanisms
* Layer normalization
* Residual connections
* Training pipelines
* GPT-2 architecture decisions
* Inference and sampling
* Modern improvements to transformer models

The repository serves as:

* A personal knowledge base
* A code-annotated version of nanoGPT
* A collection of implementation experiments
* A foundation for future open-source contributions

---

# Learning Goals

## Phase 1: Understanding nanoGPT

* [x] Read entire model.py
* [x] Document LayerNorm implementation
* [x] Document Causal Self Attention
* [x] Document MLP block
* [x] Document GPT architecture
* [x] Trace complete forward pass
* [x] Understand train.py end-to-end
* [x] Understand sampling pipeline

---

## Phase 2: Modern Transformer Improvements

Planned implementations:

* [x] Rotary Position Embeddings (RoPE)
* [x] RMSNorm
* [x] KV Cache
* [x] Grouped Query Attention (GQA)
* [x] SwiGLU
* [x] Multi Query Attention (MQA)
* [ ] Quantization Experiments
* [ ] Flash Attention Integration

---

## Phase 3: Inference Optimization

Planned studies:

* [ ] Efficient autoregressive decoding
* [ ] KV Cache benchmarking
* [ ] Memory optimization
* [ ] Throughput analysis
* [ ] Speculative Decoding

---

# Repository Structure

```text
nanogpt-learning/

├── README.md

├── notes/
│   ├── layernorm.md
│   ├── causal-self-attention.md
│   ├── mlp.md
│   ├── residual-connections.md
│   ├── embeddings.md
│   └── forward-pass.md

├── diagrams/
│   ├── transformer-architecture.png
│   ├── attention-flow.png
│   └── tensor-shapes.png

├── experiments/
│   ├── rope/
│   ├── rmsnorm/
│   ├── kv-cache/
│   ├── flash-attention/
│   └── gqa/

├── benchmarks/

└── llm/
│   ├── model.py
│   └── readme.md
```
# Modern GPT Implementation

A progressively enhanced GPT implementation built from scratch in PyTorch, incorporating several techniques used in modern Large Language Models. Check the file llm/model.py

## Implemented Features

### Core Transformer

* Token Embeddings
* Positional Information
* Multi-Head Self-Attention
* Feed Forward Network (MLP)
* Residual Connections
* Causal Masking

### Modern LLM Improvements

* Rotary Positional Embeddings (RoPE)
* RMSNorm
* KV Cache for autoregressive inference
* Grouped Query Attention (GQA)

## Architecture

The implementation follows a modular design within a single `model.py` file, allowing features to be enabled or disabled through configuration.

```text
Input
  ↓
Token Embeddings
  ↓
Transformer Block
    ├── RMSNorm
    ├── GQA / MHA
    ├── RoPE
    ├── KV Cache
    ├── Attention
    └── MLP
  ↓
Output Projection
```

## Attention Pipeline

```text
QKV Projection
      ↓
RoPE
      ↓
KV Cache Update
      ↓
GQA KV Expansion
      ↓
Scaled Dot Product Attention
      ↓
Output Projection
```
```
