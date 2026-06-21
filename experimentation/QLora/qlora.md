# QLora (Quantized Low rank adapters)

So Lora solved the problem of finetuning the complete model to only finetuning the adapters or the $\Delta W$, but the model is still in FP16 which is of large size QLora solves this by Quantization we Quantize the frozen weights to 4-bit quantization their by reducing the size of the frozen weights of the model. The Lora adapters are still trained in high precision.

#### QLora Components

1. 4-bit Quantization
2. NF4 Data Type
3. Double Quantization
4. Paged Optimizers

##### 4-bit Quantization

Normally each weight is stored as FP16 so after quantization we convert each of the weights into 4-bit quantization weight which reduces the memory required to storage by 75%. During Forward Pass we use the below approach.
```
4-bit
  ↓
dequantize
  ↓
BF16
  ↓
matmul
```

##### NF4 Data Type

Naive Quantization hurts performance. Neural networks weights generally follow a normal distribution most values are near zero so uniform quantization wastes precision. So NF4(NormalFLoat4) uses quanitzation bins optimized for gaussian weights more precision near zero less precision near outliers this results in 4-bit accuracy equivalent to 16-bit accuracy. This is the major innovation on QLora.

##### Double Quantization

Normal Quantization stores the quantized weights and the scale but scale is itself requires memory so QLora quantizes the scale as well therefore we get double quantization this saves ~0.37 bits/parameter.

##### Paged Optimizers

During training memory spikes occur loss.backward() suddenly GPU OOM because optimizers states grow. QLora uses paged AdamW which leverages unified memory so that Inactive pages move to CPU. GPU <-> CPU Preventing memory spikes.