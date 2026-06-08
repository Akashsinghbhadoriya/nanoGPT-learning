# Quantization

Quantization is the process of converting high precision numbers into lower precision numbers.

| Format | Bits |
| ------ | ---- |
| FP32   | 32   |
| FP16   | 16   |
| BF16   | 16   |
| INT8   | 8    |
| INT4   | 4    |
| INT2   | 2    |


Suppose a weight matrix in FP16 occupies 16 bits -> [0.234, -1.756, 2.845, -0.928]

After INT8 Quantization -> [12, -91, 127, -48]

scale = 0.0224

real_value = scale * quantized_value

small loss in precision gives large reduction in memory and compute.

### Why Quantization matters:

For a 7B model\
FP16 -> 7B * 2 byte = 14 GB\
INT8 -> 7B * 1 byte = 7 GB\
INT4 -> 7B * 0.5 byte = 3.5 GB\
This is how people run larger models on laptop:
- Lower VRAM
- Faster inference
- Lower bandwidth
- Cheaper Deployment

### Where Quantization happens:
```
Input
 ↓
Embedding
 ↓
Attention
 ↓
MLP
 ↓
Output
```

Each layer is -> x * W\
W -> Weights\
x -> Activations

Weight-only quantization: Weights -> INT4, Activations -> FP16\
Weight + Activation Quantization: Weights -> INT8, Activations -> INT8 (harder because activation have large outliers)\
KV Cache quantize: Modern inference system quantize K-cache and V-cache which reduces long context memory drastically.

### Quantization Mathematics

x -> floating point value is mapped to\
$q = round(x / scale)$\
$x_hat = q * scale (reconstructed)$\
$scale = max(abs(x)) / 127$ (symmetric quantization because INT8 ranges from -128 to 127)

### Symmetric vs Asymmetric Quantization

Symmetric Quantization -> range[-127, 127] (fast and used heavily for llms)\
Formula: $q = round(x / scale)$

Asymmetric Quantization -> range[0, 255] (More accurate, more expensive often used in mobile deployment)\
Formula: $q = round(x/scale) + zeropoint$\
for calculating the zeropoint we need to calculate the zero point because the values might not be symmetric about zero\
scale is calculated by calculating the range for the real-world data to the integer range.
$$scale = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}$$
$$q_{x=0} = \frac{0.0}{scale} + Z$$
$$q_{x=0} = Z$$

So Z is the integer representation of the real value 0.0
$$q_{x_{min}} = \frac{x_{min}}{scale} + Z$$
$$Z = q_{x_{min}} - \frac{x_{min}}{scale}$$

### Quantization Error

Suppose 0.87 become 0.85 after quantization the remaining 0.02 is the error.\
Every weight introduces error. Millions of weight create accuracy degradation and the entire research focus on minimizing this error.

#### Types of Quantization

1. INT8 Quantization -> It is a post training quantization it reduces the memory requirements by a factor of 4 by using 8 bits only but the inference time increases because we have an additional dequantized step. So Quantization alone does not make inference faster we need INT8 kernels which make inference faster.
```
INT8 Weights
      ↓
Dequantize
      ↓
FP32 MatMul
      ↓
Output
```

Production Systems do:
```
INT8 Weights
INT8 Activations
      ↓
INT8 GEMM
      ↓
Output
```
