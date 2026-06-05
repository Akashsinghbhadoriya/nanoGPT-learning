# Benchmarking performance of KV cache

We implemented KV Cache on the model.py and used it for text generation we generated same number of tokens using both with KV Cache and without KV Cache. The results were surprising.

#### With KV Cache

1. Tokens generated = 100, token speed per sec = 111.69771686068545 tokens per sec
2. Tokens generated = 1000, token speed per sec = 101.0287371480117 tokens per sec

#### Without KV Cache

1. Tokens generated = 100, token speed per sec = 51.97621708969151 tokens per sec
2. Tokens generated = 1000, token speed per sec = 11.08879195955563 tokens per sec