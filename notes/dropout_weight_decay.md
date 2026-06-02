# Dropout vs Weight Decay


| Aspect | Dropout | Weight Decay |
| :--- | :--- | :--- |
| **Objective** | Prevent overfitting | Penalize large weights |
| **Implementation** | Randomly set neurons to zero | Add a regularization term |
| **Effect on Neurons** | Temporarily deactivate some | Penalize large weights |
| **Ensemble Learning** | Yes | No |
| **Computation Overhead** | Adds computational cost during training | Adds computational cost during training |
| **Hyperparameter** | Dropout rate | Regularization parameter (lambda) |
| **Interpretability** | Introduces randomness, making interpretation challenging | Encourages smoother weight distributions |
| **Common Use Case** | Deep learning architectures | Linear regression, neural networks, etc. |
