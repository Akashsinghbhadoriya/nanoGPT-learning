## Layer Normalization

Layer Normalization is done to solve the problem of vanishing gradients when training a Neural Networks.
Instead of batch normalization which computes mean and variance along the batch in layer normalization we do it for each training example

The Layer Normalization is done to make mean = 0 and variance = 1 along the embedding dimension

for making the mean = 0 and variance = 1 we need to use the below formula for each input element
$x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

$\mu$ = mean of the data along row\
$\sigma$ = variance along the row\
$\epsilon$ = avoid dividing by zero\

Also we use learning parameters to avoid normalizing all the hidden layers as a linear function we need to make it complex so we use scale($\gamma$ gamma) and shift($\beta$ beta)

The final normalized layer is calculated as:
$x_n = \gamma * x + \beta$

$\gamma$ = `nn.Parameter(torch.ones(emb_dim))`\
$\beta$ = `nn.Parameter(torch.zeros(emb_dim))`

so when we are training using backpropogation scale and shift are adjusted accordingly to fit the training.
