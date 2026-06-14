# Flash Attention

1. Why was FlashAttention needed? -> Transformer attention becomes extremely expensive as context length increased. Standard attention creates an attention matrix of size N*N as the N grows the memory usage becomes extremely enormous. The attention matrix is created and used once and then discarded and yet huge amount of gpu memory is spend storing it.\
Flashattention was created to:
- avoid storing the full attention matrix.
- reduce memory transfers.
- Increase throughputs.
- Enable longer context lengths.

2. Why is standard attention slow? -> Moving data is slower than doing computation on modern GPUs.\
step 1: Compute $QK^T$ store to HBM.\
step 2: Read from HBM and apply softmax and then store results.\
step 3: Read again multiply by V and store results.\
This Compute -> Write -> Read -> Compute -> Write -> Read -> Compute -> Write (A lot of memory traffic occurs GPU spend significant time waiting for data.)

3. What is HBM? -> (High Bandwidth Memory) It is a large memory attached with GPU(it acts as GPU RAM).\
A100-> 40GB/80GB\
H100-> 80GB\
pros:- Huge Capacity, Very High Bandwidth.\
cons:- still much slower than on-chip memory.\
Accessing HBM still takes longer than SRAM.

4. What is SRAM?-> (Static Random Access Memory) This memory is directly located inside the GPU chip. very small very fast.\
pros-> Extremely low latency, extremely high bandwidth.\
cons-> very limited.\
Flashattention keeps the Blocks on the SRAM.

5. Why does memory movement matter?-> Modern GPUs are incredibly powerful. An H100 can perform trillions of operations per second. The issue is compute is faster than the memory. GPU has to wait for data.\
Worker = GPU cores\
Warehouse = HBM\
Desk = SRAM\
Flash attention minimizes warehouse trips. The GPU stays busy computing.

6. What is Online Softmax? -> Normal Softmax requires all scores at once. Flash attention only sees a block at a time. Therefore it keeps running maximum, denominator for each row this allows exact softmax calculation without storing the entire matrix.

#### More questions to answer?

1. What problem FlashAttention solves\
Problem 1: Excessive Memory Usage\
standard attention stores -> $QK^T$, $SOFTMAX(QK^T)$ and Intermediate buffers. the memory grows $O(N^2)$\
Flash Attention avoids storing these matrices its memory grows $O(Nd)$ where d is head dimension.

Problem 2: Memory bandwidth bottleneck\
GPU can compute very fast. The bottleneck is reading and writing memory and reading and writing again flash attention dramatically reduces these operations.

2. Why it is faster?-> Flash attention keeps computations inside the fast on-chip memory.\
instead of Compute -> Write -> Read -> Compute -> Write -> Read -> Compute -> Write\
It performs Load Block -> Compute Everything -> Write the final result.\
Flash attention does not reduce computational complexity it is $O(N^2d)$ it does less memory movement.

3. Why it uses less memory? -> Standard attention stores: $QK^T$ shape is $N*N$ but flash attention process Q block, K block, V block one chunk at a time after processing discard the block and go to the next block so the memory usage becomes linear rather than quadratic.

4. How it differs from KV Cache -> KV Cache and flash attention solve different problems the KV Cache prevents the recomputation of already calculated k and v values for previous tokens by storing them in the KV cache memory but flash attention increases the speed of computation by managinging GPU memory during runtime computation of attention. flashattention changes the attention algorithm to reduce memory traffic.

5. How it differs from Quantization -> again quantization solves a different problem reduces memory size increases throughput and reduces mdel size.


#### How does FlashAttention work?
```
step 1:- Split Q into blocks Q1, Q2, Q3, Q4, ....
step 2:- Split K and V into blocks K1, K2, K3, K4, ... and V1, V2, V3, V4, .....
step 3:- load a Q block into SRAM.
step 4:- Iterate through K/V blocks
step 5:- maintain a running max, sum, output using online softmax.
step 6:- Write only the final output. Never write the attention matrix.
```
#### GPU Memory Hierarchy

Memory levels:-\
```
Registers      (Fastest, Smallest)
    ↓
Shared Memory / SRAM
    ↓
L1 Cache
    ↓
L2 Cache
    ↓
HBM (GPU RAM)
```

- Register -> Fastest memory on the GPU. Each thread owns its registers. Fastest access, very limited capacity, private to thread.
- SRAM -> Shared among threads in a block. Extremely fast, much smaller than HBM and user controlled.
- L1 Cache -> Automatic cache. GPU places frequently used data here. Programmer usually doesn't control it.
- L2 Cache -> Shared across entire GPU. Larger than L1. slower than L1.
- HBM -> This is GPU RAM. Every tensor initially lives here.
#### Online Softmax

This is the heart of the Softmax.

For normal softmax: for a row : [2, 5, 1]\
softmax : $$softmax(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
This needs all values simultaneously.

Flash attention processes blocks:\
Block 1 : [2, 5, 1]\
Block 2 : [7, 4, 3]\
We never have all scores together.

We solve this by running maximum and running sum so that we can compute each softmax incrementally and do not need to store all the scores. This makes flash attention possible.
# FlashAttention Forward Pass

## Goal

Compute attention:

\[
O = \text{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
\]

without materializing the full attention matrix.

---

## Why?

Standard attention computes:

```text
QKᵀ  →  (N,N)
```

For long sequences this matrix becomes extremely large.

Example:

```text
N = 8192

Attention Matrix:
8192 × 8192
≈ 67 Million Elements
```

FlashAttention avoids storing this matrix.

---

## Core Idea

Split tensors into blocks:

```text
Q → Q1,Q2,Q3,...

K → K1,K2,K3,...

V → V1,V2,V3,...
```

Process one block at a time inside SRAM.

---

## Algorithm

### Step 1

Load a Q block into SRAM.

```text
Q_block
```

---

### Step 2

Initialize running statistics.

```python
m = -inf      # running row max
l = 0         # running denominator
O = 0         # running output
```

---

### Step 3

Iterate through KV blocks.

```python
for K_block, V_block:
```

Compute scores:

\[
S = Q_{block}K_{block}^{T}
\]

---

### Step 4

Update row maximum.

\[
m_{new}
=
\max(m_{old}, rowmax(S))
\]

---

### Step 5

Compute exponentials using the new maximum.

\[
P
=
e^{S-m_{new}}
\]

---

### Step 6

Update denominator.

\[
l_{new}
=
l_{old}e^{m_{old}-m_{new}}
+
\sum P
\]

---

### Step 7

Update output accumulator.

\[
O_{new}
=
\frac{
l_{old}e^{m_{old}-m_{new}}O_{old}
+
PV_{block}
}
{l_{new}}
\]

---

### Step 8

Continue until all KV blocks are processed.

---

### Step 9

Write final output.

```text
Output → HBM
```

---

## Pseudocode

```python
for q_block in Q:

    m = -inf
    l = 0
    O = 0

    for k_block, v_block in KV:

        S = q_block @ k_block.T

        m_new = max(m, rowmax(S))

        P = exp(S - m_new)

        l_new = (
            l * exp(m - m_new)
            + rowsum(P)
        )

        O = (
            O * l * exp(m - m_new)
            + P @ v_block
        ) / l_new

        m = m_new
        l = l_new

    write(O)
```

---

## Memory Complexity

Standard Attention:

\[
O(N^2)
\]

FlashAttention:

\[
O(Nd)
\]

---

## Key Insight

The attention matrix never exists in memory.

Only small blocks are processed.

This dramatically reduces memory traffic.



# FlashAttention Backward Pass

## Problem

Training requires gradients.

Standard attention stores:

```text
QKᵀ
Softmax(QKᵀ)
```

and reuses them during backward.

FlashAttention never stored them.

---

## Challenge

Need gradients:

\[
\frac{\partial L}{\partial Q}
\]

\[
\frac{\partial L}{\partial K}
\]

\[
\frac{\partial L}{\partial V}
\]

but attention probabilities were never saved.

---

## Solution

Recompute them.

---

## Main Idea

During backward:

```text
Reload Q
Reload K
Reload V
```

and recompute the same attention blocks.

---

## Why This Works

Computing is cheap.

Memory bandwidth is expensive.

Recomputing attention often costs less than storing huge matrices.

---

## Backward Steps

### Step 1

Load:

```text
Q_block
K_block
V_block
```

---

### Step 2

Recompute:

\[
S = QK^T
\]

---

### Step 3

Recompute Online Softmax statistics.

```text
row max
row sum
probabilities
```

---

### Step 4

Recover attention probabilities.

\[
P
=
\text{softmax}(S)
\]

---

### Step 5

Compute gradient wrt V.

\[
dV
=
P^T dO
\]

---

### Step 6

Compute gradient wrt probabilities.

\[
dP
=
dO V^T
\]

---

### Step 7

Backprop through softmax.

\[
dS
=
\text{SoftmaxBackward}(P,dP)
\]

---

### Step 8

Compute gradient wrt Q.

\[
dQ
=
dS K
\]

---

### Step 9

Compute gradient wrt K.

\[
dK
=
dS^T Q
\]

---

## Tradeoff

Standard Attention:

```text
Store activations
Use more memory
Less recomputation
```

FlashAttention:

```text
Store less
Recompute during backward
Use much less memory
```

---

## Key Insight

FlashAttention backward is essentially:

```text
Forward Again
+
Gradient Computation
```

using block-wise recomputation.



# PyTorch SDPA (Scaled Dot Product Attention)

## What is SDPA?

SDPA is PyTorch's optimized attention API.

Formula:

\[
O
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V
\]

---

## API

```python
import torch
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True
)
```

---

## Why SDPA Exists

Before SDPA:

```python
scores = q @ k.transpose(-2,-1)
scores = scores / math.sqrt(d)

attn = torch.softmax(scores, dim=-1)

output = attn @ v
```

This creates large intermediate tensors.

SDPA hides these details and dispatches optimized kernels.

---

## Available Backends

### 1. Math Attention

Reference implementation.

```text
Slowest
Most compatible
```

---

### 2. Memory Efficient Attention

Intermediate optimization.

```text
Less memory
Faster than math kernel
```

---

### 3. FlashAttention

Fastest backend.

```text
Block-wise execution
Online Softmax
SRAM optimized
```

---

## Backend Selection

PyTorch automatically chooses based on:

```text
GPU Architecture
Tensor Shape
Data Type
Mask Type
```

---

## Example

```python
import torch
import torch.nn.functional as F

B = 2
H = 8
N = 1024
D = 64

q = torch.randn(B,H,N,D, device="cuda")
k = torch.randn(B,H,N,D, device="cuda")
v = torch.randn(B,H,N,D, device="cuda")

out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True
)
```

---

## Why Modern LLMs Use SDPA

Used in:

```text
Llama
Mistral
Gemma
GPT-style models
Qwen
DeepSeek
```

because PyTorch automatically dispatches highly optimized attention kernels.

---

## Relationship to FlashAttention

```text
Attention Formula
        ↓
SDPA API
        ↓
PyTorch Backend Selection
        ↓
FlashAttention Kernel
```

Most users never call FlashAttention directly.

They call:

```python
F.scaled_dot_product_attention(...)
```

and PyTorch executes FlashAttention internally whenever possible.

---

## Summary

SDPA is the production interface.

```text
User
 ↓
SDPA
 ↓
FlashAttention
 ↓
GPU
```

This allows developers to benefit from FlashAttention without implementing the algorithm themselves.