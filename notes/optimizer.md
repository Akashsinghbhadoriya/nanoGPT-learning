# Optimizers

An optimizer is an algorithm that adjusts model weights based on gradients to minimize the loss function. It is the engine which drives learning in a LLM model. The most famous optimizer being used in LLM training is the AdamW.

### AdamW

1. It is a successor to famous Adam optimizer
2. Adam combined first moment with adaptive per-parameter learning rates (second moment) but their was a flaw in implementation of L2 regularization.
3. In standard Adam weight decay was incorrectly coupled with the adaptive learning rate making the regularization weaker for parameters with large gradient variance.
4. AdamW decoupled the weight decay from the adaptive learning rate.
5. Key hyperparameters are $\beta1$, $\beta2$, learning rate, weight decay.
6. weight decay regularization which penalizes larger weights to prevent model overfitting.
7. we only optimize the learnable parameters such as matmuls and embeddings the biases and layer normalizations are not weight decayed
8. $\beta1$ is the first moment and $\beta2$ is 2nd moment.
9. learning rate is the hyperparameter that determines the size of the steps an optimizer takes to adjust a machine learning model's weight.