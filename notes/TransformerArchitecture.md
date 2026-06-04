# Transformer Architechture

Transformer architecture is the one block of transformer which will be repeated in the gpt model mmultiple time for example in gpt 2 small 124M the transformer block is repeated 12 times below is the proper transformer architecture:
Transformer architecture is the building block of modern large language models. Also we add shortcut or residual connections in the transformer as well to prevent the issue of vanishing gradients.

### Flow of input

1. Input -> Input * token_embedding_layer
2. x -> token_embedding + pos_embedding
3. Input -> LayerNormalization(x)
4. LayerNormalization(x) -> Multi-head Causal Self Attention(x) (residual connection added after this operation)
5. Multi-head Causal Self Attention(x) -> LayerNormalization(x)
6. LayerNormalization(x) -> FFN(x) (residual connection added after this operation)

### Practical Significance

1. token embedding table converts the discrete token ids into continous vectors.
2. Future tokens are hidden using causal masking this enables autoregressive next-token prediction.
3. Weight tying means the token embedding and the output projection matrix share the parameters this reduces parameter count and improves language modeling performance.
4. Weight tying improves performance because it encorages consistency between how tokens are represented and how they are predicted.

#Tensor Tracing

| Step | Input Dimensions | Output Dimensions | Notes |
| ---- | ---------------- | ----------------- | ----- |
| Input| (B, C) | (B, C) |
|Token Embedding Layer | (B, C) , (C, emb_dim) | (B, C, emb_dim) | we generate token embeddings using the token ids and a lookup on the embedding layer|
| Positional Embedding Layer | (C), (block_size, emb_dim) | (C, emb_dim) | positional embedding gives the information about the absolute positions|
| final embedding | (B, C, emb_dim) + (C, emb_dim) | (B, C, emb_dim)| x = token_embedding + positional_embedding|
| Layer Normalization| (B, C, emb_dim) * (emb_dim) + (emb_dim) | (B, C, emb_dim) | multiplication and addition happens on the last dimension on each row of each batch |
| Causal Attention | (B, C, emb_dim) | (B, C, emb_dim) | for individual calculations check model.py this output contains information about multiple heads|
| Residual Connection | (B, C, emb_dim) + (B, C, emb_dim) | (B, Cm emb_dim) | This is the step where we add the input back to the transformed x value |
| Layer Normalization| (B, C, emb_dim) * (emb_dim) + (emb_dim) | (B, C, emb_dim) | multiplication and addition happens on the last dimension on each row of each batch |
| Feed - Forward Network (Linear Layer) | (B, C, emb_dim) | (B, C, 4 * emb_dim) | we expand the last embedding dimension. |
| Feed - Forward Network (GELU) | (B, C, 4 * emb_dim) | (B, C, 4 * emb_dim) | Activation function to add non - linearity in Neural Network | 
| Feed - Forward Network (Linear Layer) | (B, C, 4 * emb_dim) | (B, C, emb_dim) | we compress the last embedding dimension. |
| Residual Connection | (B, C, emb_dim) + (B, C, emb_dim) | (B, Cm emb_dim) | This is the step where we add the input back to the transformed x value |