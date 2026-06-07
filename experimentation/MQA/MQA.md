# MQA

Multi Query attention is technique used to reduce the memory utilization of KV Cache by grouping all the queries with multiple heads with a single head for key and value. This gives much faster calculation and much effiecient memory.

#### Researcher observation:

- Different query heads learn different attention patterns.
- key and value heads are often redundant.
- So do we really need separate key and value.
- MQA says no.

```
Q1 ─┐
Q2 ─┤
Q3 ─┤
Q4 ─┤
Q5 ─┤────► Shared K,V
Q6 ─┤
Q7 ─┤
Q8 ─┘
```

#### Drawbacks in larger models

- Researchers observed quality degradation in Larger models so they used GQA which gives much quality closed to MHA

### Comparison of MHA, GQA, MQA

MHA  → Best quality, expensive KV cache

MQA  → Smallest KV cache, some quality loss

GQA  → Nearly MHA quality, much smaller cache