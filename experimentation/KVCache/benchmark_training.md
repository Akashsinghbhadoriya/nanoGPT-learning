# Benchmarking performance of KV cache

We implemented KV Cache on the model.py and used it for text generation we generated same number of tokens using both with KV Cache and without KV Cache. The results were surprising.

#### With KV Cache

1. Tokens generated = 100, token speed per sec = 111.69771686068545 tokens per sec, memory usage = KV Cache memory:7.31 MB
2. Tokens generated = 1000, token speed per sec = 101.0287371480117 tokens per sec, memory usage = KV Cache memory:70.59 MB

| Tokens Generated | Tokens/sec | Total Time (s) | KV Cache Memory (MB) | Latency (ms/token) |
| ---------------- | ---------: | -------------: | -------------------: | -----------------: |
| 10               |      96.97 |          0.103 |                 0.98 |               9.71 |
| 60               |     113.03 |          0.531 |                 4.50 |               8.30 |
| 110              |     109.87 |          1.001 |                 8.02 |               8.53 |
| 160              |     107.60 |          1.487 |                11.53 |               8.72 |
| 210              |     110.86 |          1.894 |                15.05 |               8.47 |
| 260              |     110.95 |          2.343 |                18.56 |               8.46 |
| 310              |     110.71 |          2.800 |                22.08 |               8.49 |
| 360              |     110.41 |          3.261 |                25.59 |               8.51 |
| 410              |     110.00 |          3.727 |                29.11 |               8.54 |
| 460              |     108.65 |          4.234 |                32.63 |               8.64 |
| 510              |     108.16 |          4.715 |                36.14 |               8.69 |


#### Without KV Cache

1. Tokens generated = 100, token speed per sec = 51.97621708969151 tokens per sec
2. Tokens generated = 1000, token speed per sec = 11.08879195955563 tokens per sec

| Tokens Generated | Tokens/sec | Total Time (s) | KV Cache Memory (MB) | Latency (ms/token) |
| ---------------- | ---------: | -------------: | -------------------: | -----------------: |
| 10               |      57.20 |          0.175 |                 0.00 |              16.42 |
| 60               |      60.24 |          0.996 |                 0.00 |              16.03 |
| 110              |      50.04 |          2.198 |                 0.00 |              19.41 |
| 160              |      43.22 |          3.702 |                 0.00 |              22.56 |
| 210              |      36.31 |          5.784 |                 0.00 |              26.95 |
| 260              |      31.01 |          8.385 |                 0.00 |              31.64 |
| 310              |      28.37 |         10.927 |                 0.00 |              34.64 |
| 360              |      25.60 |         14.065 |                 0.00 |              38.47 |
| 410              |      23.88 |         17.170 |                 0.00 |              41.28 |
| 460              |      21.69 |         21.205 |                 0.00 |              45.49 |
| 510              |      19.70 |         25.891 |                 0.00 |              50.15 |


#### Implications

1. By Using KV Cache we have significantly improved the tokens generated per second but the memory usage also increases significantly the details are mentioned above.
2. But the speed improvement is worth the tradeoff of the increase in memory overhead.

| Metric (≈510 Tokens) | With KV Cache | Without KV Cache |            Improvement |
| -------------------- | ------------: | ---------------: | ---------------------: |
| Tokens/sec           |        108.16 |            19.70 |       **5.49× faster** |
| Latency (ms/token)   |          8.69 |            50.15 |        **5.77× lower** |
| Total Time (s)       |          4.72 |            25.89 |       **5.49× faster** |
| Memory (MB)          |         36.14 |             0.00 | Uses additional memory |
