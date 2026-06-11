"""
Complete GPTQ Implementation
"""

import torch
from wonly_int4 import quantize_int4, dequantize_int4

# Step 1: Collecting activations

activations = []

def hook(module, inp, out):
    activations.append(inp[0].detach())

handle = layer.register_forward_hook(hook)

with torch.no_grad():
    model(idx)

handle.remove()

X = torch.cat(activations, dim=0)
print(X.shape)

# Step 2: Compute Hessian Approximation
def compute_hessian(X):
    H = X.T @ X
    damp = 0.01 * H.diag().mean()

    H += damp * torch.eye(H.shape[0], device=H.device) # damping along the diagonal all other values are not impacted

    return H

# Step 3: Invert Hessian
H = compute_hessian(X)
H_inv = torch.linalg.inv(H)

# Step 4: GPTQ quantizer
def quantize_column_int4(w):

    scale = w.abs().max() / 7

    scale = torch.clamp(
        scale,
        min=1e-8
    )

    q = torch.round(
        w / scale
    )

    q = torch.clamp(
        q,
        -8,
        7
    )

    return q * scale

# Step 5: GPTQ Core Loop
def gptq_quantize(W, H_inv):

    W = W.clone()
    W_q = torch.zeros_like(W)

    n_cols = W.shape[1]

    for i in range(n_cols):

        w = W[:, i]
        w_q = quantize_column_int4(w)

        W_q[: , i] = w_q
        error = w - w_q

        if i < n_cols - 1:

            correction = error/H_inv[i,i]
            W[:, i+1] -= (correction.unsqueeze(1) * H_inv[i,i+1:])
    
    return W_q