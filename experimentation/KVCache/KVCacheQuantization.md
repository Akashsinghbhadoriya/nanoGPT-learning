# KV Cache Quantization

we do kv cache quantization to reduce the memory requirement of the KV cache which grows and can become larger than that of the actual model size as the number fof token generated increases.

1. INT8 KV Quantization -> In INT8 KV Cache quantization we use convert the KV Cache Float32 into INT8 representation and storing the scales as well for the quantization. The memory storage of the kv cache got reduced from 36.140625 MB to 0.7568511962890625 MB after the quantization for 510 generated tokens.

2. KIVI Quantization -> IN KIVI Quantization we quantize the keys per channel and the values per token so that the outlier in the key channels are restricted to a particular channel itself and it does not spill over to other channels.

Since for key quantizing along channels is difficult because these channels spread along various tokens and the number of tokens generated is arbitrary we divide the key in to 2 parts
$$X_k -> X_{k_{g}} \space X_{k_{r}}$$
$$X_{k_{g}} = X_k[:l-r]$$
$$X_{k_{r}} = X_k[l-r:]$$
where (l-r) is divisible by G where keys are grouped by G token at a time. Now $X_{k_{g}}$ is quantized and $X_{k_{r}}$ is kept in full precision and any new token will be appended to the $X_{k_{r}}$ and as soon as the size becomes R where R is divisible by G we quantize it and append to $X_{k_{g}}$ and $X_{k_{r}}$ tensor is emptied. Similarly for the value vector we split it into 2 parts.