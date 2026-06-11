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
```raw
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

1. INT8 Quantization -> It is a post training quantization it reduces the memory requirements by a factor of 4 by using 8 bits only but the inference time increases because we have an additional dequantized step. So Quantization alone does not make inference faster we need INT8 kernels which make inference faster. This is also called per tensor quantization where a single scale must fit every neuron.
```raw
INT8 Weights
      ↓
Dequantize
      ↓
FP32 MatMul
      ↓
Output
```

Production Systems do:
```raw
INT8 Weights
INT8 Activations
      ↓
INT8 GEMM
      ↓
Output
```

Limitations:
- suppose the weight matrix is then scale = 15.7/127 = 0.1236 so for 1st row 0.1/0.1236 = 0.8 ~ 1 fpr 2nd row 15.7/0.1236 = 127 so the 2nd row scales properly but not the 2nd ones very poor precision for the 1st row.
```raw
W = [[0.1, 0.2, -0.3],
    [12.5, -8.4, 15.7]]
```
- different neurons have very different weight distribution so we need to use different scales for different neurons

2. Per-Channel Quantization-> This solves the problem of using a single scale for all the neurons instead use different scale for different neurons. So every neuron gets maximum precision.

3. LLM.int8() -> Before this paper the INT8 quantization of weights and activations worked well for CNNs and smaller transformer models below 6.7 B parameters but failed for larger GPT-3, BLOOM, OPT styled models which gave lost of accuracy when quantized. The issue observed that for larger transformer models weight quantized properly but activations did not after analysing the activation values researchers found that tiny number dimensions had huge values these were called emergent features or outlier dimensions. These few dimensions were causing a problem few extreme channels were destroying quantization quality as models become larger the outliers became stronger. The solution to this problem was Quantize Most dimensions with INT8 but the outlier ones with FP16
`99.9% INT8 0.1% FP16` This worked because the expensive computation remains INT8 and the dangerous outlier dimensions stay FP16 which preserves the accuracy. The central idea of the paper is "Mixed-Precision Decomposition"
channels above the threshold use FP16
`abs(x) > threshold, threshold = 6 or 8`
Only activations were FP16 because weights behave nicely but activations create the real problem

Channel for an activation x is the column or the hidden dimensions(128, 768) so their are 768 channels in the activations for the weight the channel means each output neuron which is W(3072, 768)(out, in) here a(hidden dim)== w(in) so we take out the outlier activation channel and the similar column from the weight for multiplication spearately.

Both are mathematically identical
```
Y = X_normal @ W_normal.T + X_outlier @ W_outlier.T
Y = X @ W.T

Example:

X = [x0,x1,x2,x3]
W = [w0,w1,w2,w3]

X @ W.T = x0*w0 + x1*w1 + x2*w2 + x3*w3

x_norm = [x0,x1,x2]
w_norm = [w0,w1,w2]
x_out = [x3]
w_out = [w3]

y_norm + y_out = x0*w0 + x1*w1 + x2*w2 + x3*w3
```

4. SmoothQuant(W8A8) -> It is a Post training quantization technique used to quantize the weights as well as activations with INT8 which becomes W8A8 quantization this increases the inference on larger models by 1.5x and reduces the memory requirements by 2x. Since quanitzation becomes difficult for models larger than 6.7B parameters where a small percentage of outliers can decrease the accuracy of the model during inference. These outliers make activation quantization difficult. The outliers in activation are 100x the size of the other activation values. We can use per channel quantization which handles outliers much better in the channels which contain these outliers also the accuracy is much better than per-tensor quantization we could use per-token quantization but it helps little over per-channel tokenization. But the per-channel tokenization has a limitation that it does not map well with hardware accelerated GEMM kernels that rely on sequence of operations executed at high throughput. In those kernels scaling can only be performed along the outer dimensions of the matrix multiplication. So instead of per-channel activation quantization we "smooth" the input activation by dividing it by per-channel smoothing factor to keep the mathematical equivalence of linear layer we scale the weights accordingly in the reversed direction.

Migrating the quantization difficulty from activations to weights:

We aim to use a per-channel smoothing factor for reducing the quantization error we should increase the effective quantization bits for all the channels this will happen if all the channels have the same maximum magnitude. This will happen if
$$s_{j} = max(|X_{j}|) , j = 1,2,3,...$$
This formula insures that after division each channel have the same maximum value which is easy for quantization. However this formula pushes all the quantization diffculty of the activation to weights matrix. In this case quantization errors will be larger for weights. We can also push the quantization difficulty of weights to the activation by using the below s:
$$s_{j} = \frac{1}{max(|W_{j}|)}$$

Therefore we need to split the quantization difficulty between weights and activations so that both are easy to quantize. So we use hyperparameter migration strength $\alpha$ to control how much difficulty we migrate from activations to weights:

$$s_{j} = \frac{max(|X_{j}|)^\alpha}{max(|W_{j}|)^(1-\alpha)}$$
Most of the models opt $\alpha = 0.5$

5. INT4 quantization :- In this quanitzation we do scaling from the range [-8,7] which represent only 4 bits and gives a 4x reduction than FP16 also we do packing of 2 INT4 numbers into one INT8 for calculation by bit shifting. While doing int4 quantization we got an issue where the agreement was 0 and mse was around 191 then we ran a diagnostic on all the layers of the gpt2 one by one layer and the issue was cause by lm_head so we did not quantize this linear layer and after this that mse went down and the agreement was 1