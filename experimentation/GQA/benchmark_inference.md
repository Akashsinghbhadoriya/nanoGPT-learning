# KV Cache + GQA 

n_heads = 12
n_kv_heads = 4

Implementing GQA reduced KV cache memory consumption from 36.14 MB to 12.05 MB (≈66.7% reduction) while maintaining nearly identical decoding throughput (108.16 vs 105.36 tokens/sec) and latency (8.69 vs 8.92 ms/token). These results demonstrate the primary benefit of GQA: substantially lower inference memory requirements with minimal impact on generation speed.

| Tokens Generated | Tokens/sec | Total Time (s) | KV Cache Memory (MB) | Latency (ms/token) |
| ---------------- | ---------: | -------------: | -------------------: | -----------------: |
| 10               |     103.74 |          0.096 |                 0.33 |               9.05 |
| 60               |     114.52 |          0.524 |                 1.50 |               8.17 |
| 110              |     115.32 |          0.954 |                 2.67 |               8.12 |
| 160              |     113.80 |          1.406 |                 3.84 |               8.24 |
| 210              |     112.30 |          1.870 |                 5.02 |               8.36 |
| 260              |     110.66 |          2.350 |                 6.19 |               8.49 |
| 310              |     110.54 |          2.805 |                 7.36 |               8.50 |
| 360              |     106.25 |          3.388 |                 8.53 |               8.85 |
| 410              |     103.35 |          3.967 |                 9.70 |               9.10 |
| 460              |     100.86 |          4.561 |                10.88 |               9.31 |
| 510              |     105.36 |          4.841 |                12.05 |               8.92 |

### Comparison: KV Cache vs KV Cache + GQA

| Metric (≈510 Tokens) | KV Cache | KV Cache + GQA |     Change |
| -------------------- | -------: | -------------: | ---------: |
| Tokens/sec           |   108.16 |         105.36 |      -2.6% |
| Latency (ms/token)   |     8.69 |           8.92 |      +2.6% |
| KV Memory (MB)       |    36.14 |          12.05 | **-66.7%** |
| Total Time (s)       |     4.72 |           4.84 |      +2.5% |

