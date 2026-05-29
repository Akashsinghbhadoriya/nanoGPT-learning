# FeedForward

feedforward takes the input from the self attention layer and gives the final output it is typically consist of 3 layers. 2 Linear layers and 1 activation layer in our case GELU it improves the model performance.

```
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
