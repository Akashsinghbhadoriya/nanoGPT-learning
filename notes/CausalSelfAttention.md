# Causal Self Attention (Masking)

Causal self attention is done to mask the embeddings after the current position to avoid the large language model from learning from the future data. We do this by making the values above the diagonal as zero. Causal self attention is implemented using multi-head which means multiple causal self attention layers are combined together and we do the matrix calculation of heads in one go i will explain it below.

### Why use multi-head?

- Different heads attend to different aspects of the input.
- By operating in parallel multiple heads allow for better learning of dependencies.
- The model doesn't rely on single attention avoiding overfitting.

### Simple Self Attention

In simple self attention every input layer is weighted against every other input token that is the reason we are calling it self attention and we use weights for calculating the values for each layer. The approach is the q, k, v approach where we use the query to search the key and finally return the value corresponding to that key. The calculation is as follows:
```
q = torch.nn.Linear(emb_dim, emb_dim)
k = torch.nn.Linear(emb_dim. emb_dim)
v = torch.nn.Linear(emb_dim, emb_dim)

queries = x @ q
keys = x @ k
values = x @ v

self_attn = queries @ keys.T
context_length = keys.shape[0]
context_vector = attn_weights @ values

```

### Causal Self Attention
It is a modified version of simple self attention with masking and dropout added at the last and multihead self attention is just joining multiple heads together we will show the calculations below

```
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
self.dropout = nn.Dropout(dropout)
self.register_buffer(
    'mask',
    torch.triu(torch.ones(context_length,context_length), diagonal=1)
)

def forward(self, x) :
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vect = attn_weights @ values

        return context_vect
```

### Dropout

Dropout is done to randomly set some of the inputs in the self-attention to zero to reduce learning rates these are usually small