# Transformer Architechture

Transformer architecture is the one block of transformer which will be repeated in the gpt model mmultiple time for example in gpt 2 small 124M the transformer block is repeated 12 times below is the proper transformer architecture:

### Flow of input

1. Input -> LayerNormalization(x)
2. LayerNormalization(x) -> Multi-head Causal Self Attention(x)
3. Multi-head Causal Self Attention(x) -> LayerNormalization(x)
4. LayerNormalization(x) -> FFN(x)

#Tensor Tracing

