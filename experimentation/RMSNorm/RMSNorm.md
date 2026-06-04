# RMSNorm (Root Mean Squared Normalization)

1. Layer Normalization while being applied to many neural networks for training helps stabilize training and improve model convergence but the layer normalization is a computationally expensive step for models with deep neural network and large number of parameters the efficiency gain by layer normalization is replaced by increased computational cost per training example.
2. In Transformers the main danger to training stability is the gradient explosion or vanishing gradients which are a factor of the multiplication (scale) and not result of addition(shift).
3. Shifting the inputs with a constant or a non-zero mean changes the location of the distribution but does not inherently compress the spread or variance of the gradients.
4. So according to RMSNorm mean normalization does not reduce the variance of hidden states or model gradients and it has little impact over the success of the LayerNorm. RMSNorm paper says that re-scaling invariance is the reason for the success of LayerNorm rather than re-centering invariance and RMSNorm only focuses on re-scaling invariance and regularizes the summed inputs.
5. RMSNorm regularizes the summed input to a neuron in one layer with the root mean square(RMS) statistic alone.
6. RMSNorm reduces the amount of computation and increases efficiency over LayerNorm.
7. We can also use pRMSNorm (partial RMSNorm) where we calculated the RMS on the partial summed inputs maintaining the invariance property.
8. RMSNorm is equal to LayerNorm when the mean of the summed inputs is zero

$$\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i$$

$$\text{RMS}(a) = \sqrt{\frac{1}{n} \sum_{i=1}^n a_i^2}$$

### pRMSNorm

1. The rescaling - invariance property of RMSNorm ascribes to the linearity property of RMS.
2. Consider neurons in one layer often have independent identically distributed structure so RMS can be estimated on the subset of these neurons rather than all of them.
3. The linearity property of $\overline{\text{RMS}}$ which indicates pRMSNorm shares the same invariance properties as RMSNorm

$$\overline{\text{RMS}}(a) = \sqrt{\frac{1}{k} \sum_{i=1}^k a_i^2}$$

$$\text{where } k = \lceil n \cdot p \rceil$$


### Impact of RMSNorm

1. RMSNorm gives accelerated training and inference for larger models which are trained on billions of parameters.
2. Reduced memory footprint since we removed the shift parameter.
3. Maintains the model quality same as that of LayerNorm.

### Limitations

1. Shifted mean representation since it does not re-center the mean.
2. Sensitivity to Activation Outliers since the RMS is sqrt(value).
3. Limited benefits in smaller models.
