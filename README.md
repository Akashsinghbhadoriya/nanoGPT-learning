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
* [ ] RMSNorm
* [ ] SwiGLU
* [ ] KV Cache
* [ ] Grouped Query Attention (GQA)
* [ ] Multi Query Attention (MQA)
* [ ] Flash Attention Integration
* [ ] Quantization Experiments

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

└── nanogpt-source/
```

---

# Model.py Study Progress

## Components Completed

### LayerNorm

Topics covered:

* Purpose of normalization
* Mean and variance computation
* Learnable scale and bias parameters
* Pre-LayerNorm architecture

Notes:

`notes/layernorm.md`

---

### Causal Self Attention

Topics covered:

* Query / Key / Value generation
* Multi-head attention
* Causal masking
* Scaled dot-product attention
* Tensor transformations

Notes:

`notes/causal-self-attention.md`

---

### MLP Block

Topics covered:

* Feed-forward networks
* Expansion ratio
* GELU activation
* Projection layers

Notes:

`notes/mlp.md`

---

# Future Papers

## Completed

* Attention Is All You Need
* Build a Large Language Model From Scratch

## Planned

### Transformer Architecture

* RoFormer
* FlashAttention
* GQA
* MQA

### Fine-Tuning

* LoRA
* QLoRA

### Alignment

* DPO
* InstructGPT

---

# Key Learning Principle

For every topic:

1. Read the paper
2. Read the implementation
3. Reimplement the idea
4. Benchmark results
5. Document learnings
6. Publish findings

The goal is to build systems-level understanding rather than passive familiarity.
