# Causal Self Attention (Masking)

Causal self attention is done to mask the embeddings after the current position to avoid the large language model from learning from the future data. We do this by making the values above the diagonal as zero. Causal self attention is implemented using multi-head which means multiple causal self attention layers are combined together and we do the matrix calculation of heads in one go i will explain it below.

### Why use multi-head?

- Different heads attend to different aspects of the input.
- By operating in parallel multiple heads allow for better learning of dependencies.
- The model doesn't rely on single attention avoiding overfitting.

### Simple Self Attention

In simple self attention every input layer is weighted against every other input token that is the reason we are calling it self attention and we use weights for calculating the values for each layer. The approach is the q, k, v approach where we use the query to search the key and finally return the value corresponding to that key. The calculation is as follows:

### Practical Significance

1. We compute Qk because it gives us the similarity score between tokens, which token should attend to each other. after getting these attention weights we use them to get the corresponding values.
2. Attention is also called soft lookup table because we are not taking a hard value like querying a value from a table using a key instead we are using the weighted average of the values by multiplying each one of them by their probabilites which is calculated by passing the attention to the softmax function which creates probabilistic distribution from random values the sum of all is 1.
3. we apply dropout to randomly turn off some of the values in the attention to prevent overfitting and the model should learn better.
4. We scale down by $\sqrt(d_k)$ because the dot product of attention becomes very large for larger embedding dimensions and to prevent that we scale down the attention. If we do not scale down then the softmax function will assign higher probability to higher gradients and for others it will be almost zero so gradients becomes unstable the model becomes divergent and take longer time to train.
5. Softmax helps in converting the similarity scores in to probability distribution allowing model to determine how much attention to be paid to each token. If we remove softmax we will get the gradients as negative and arbitrarily large and model will lose the probabilistic interpretation.
6. attention matrix represent the strength of relationship between very pair of tokens each row shows where a particular token is focusing.
7. The projection matrices Q, K, V are learned and not the attention is learned. Model learns how to project the token representation into q, k, v spaces.
8. Multiple - Heads helps the model to learn different type of relationships simultaneously. One head learn syntax other might learn long range dependencies. This is computionally efficient because everything is done in one step. Multiple heads encourage specialization.
9. Attention visualization helps researchers learn what each head has learn subject-verb, punctuation handling, name references, long range context retreival.
10. We concatenate different heads to retain the information which is learned and out projection mixes information from all head and conver it to emb_dim this allows information available from all heads to interact and become useful for downstream layers.


```raw
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

```raw
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