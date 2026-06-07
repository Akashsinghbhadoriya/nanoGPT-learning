# SwiGLU

SwiGLU is a activation function used in the MLP block of transformer. It improves model quality with relatively small computational overhead.

### Standard MLP

$$FFN(x)=W2​(GELU(W1​x))$$

```
self.c_fc = nn.Linear(n_embd, 4 * n_embd)
self.gelu = nn.GELU()
self.c_proj = nn.Linear(4 * n_embd, n_embd)
```

### Gated Linear Unit

Researcher found that adding a gate improves information flow.

```
hidden = GELU(Wx)
value = Wv(x)
gate  = Wg(x)

output = value * gate
```

The gate decides:

0 -> suppress information

1 -> pass information

Think of this as an attention but inside MLP

### SwiGLU

SwiGLU replace the gate with Swish (SiLU) activation:

$$SwiGLU(x)=(Wv​x)⊙SiLU(Wg​x)$$
$$SiLU(z)=zσ(z)$$
$$SiLU(z) = \frac{z}{1 + e^{-z}}$$



             x
           /   \
          /     \
        Wv      Wg
         |       |
         |     SiLU
         |       |
          \     /
           \   /
      Element-wise *
               |
              Wo
               |
            Output


#### Why is it better?

Standard MLP
x → GELU → projection

Every neuron contributes independently.

SwiGLU

x → value branch

x → gate branch

The gate can dynamically control which features are important.

This gives:

- Better expressiveness
- Better gradient flow
- Improved training stability
- Better perplexity for similar parameter counts

To keep compute similar, models use:

hidden_dim ≈ (8/3) * d_model