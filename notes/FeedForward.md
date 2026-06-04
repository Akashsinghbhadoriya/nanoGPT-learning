# FeedForward

feedforward takes the input from the self attention layer and gives the final output it is typically consist of 3 layers. 2 Linear layers and 1 activation layer in our case GELU it improves the model performance. Its is also called MLP (Multilayer Perceptron )

### Practical Significance

1. MLP acts as the thinking and memory layer of the model if we remove it the model capacity is limited and complex reasoning would not happen. It is the computational layer after attention it does computation of the information to learn. Attention is communication and MLP is computation.
2. The expansion of dimension creates a larger space for nonLinear computation most of the transformer's compute and prameters reside in MLP layer.
3. GELU smooth activation function which improves optimization and model quality compared to simpler activations. GELU is smooth then RELU and differentiable everywhere and gives a more stable gradient flow.

```raw
layer1 = nn.Linear(emb_dim, 4 * emb_dim)
GELU = nn.GELU()
layer2 = nn.Linear(4 * emb_dim, emb_dim)

x = x @ layer1
x = GELU(x)
x = x @ layer2
```

Linear layer increase the dimension of the input and then we apply non-linear activation function 

gelu formula approximation:

$$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left[ \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right] \right)$$
