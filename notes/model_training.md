# Model Training

The train.py contains the code for model training their are 3 different ways we can use train.py.
1. train a model from scratch.
2. finetune a pretrained gpt model
3. resume a model training from the last checkpoint

### Steps involved in model training
Note:- In the current train.py we are not using dataset and dataloader we are implementing it from first principle

1. split the data into training and validation data.
2. load the training and validation data in a batch using Dataset
3. then create a DataLoader using the Dataset the dataloader creates an iterable for the Dataset
4. iterate over the datalaoder and calculate the logits and the loss.
5. pass the loss in the backward pass for calculating the gradients.
6. using the optimizer to step and update the weights of the parameters.
7. after training we save the model weights and the optimizer state dict.

### Practical Questions

1. If we do not save the optimizer state training resumes with incorrect momentum estimates. Training loss fluctuates more because is measured across individual batches and validation loss is the average.
2. val loss can be smaller than training loss because during training dropout and other regularization are active.
3. AdamW is used instead of SGD because it adapts learning rate per parameter and model converges faster.
4. Momentum intuitvely means it accumulates past gradient information to smooth updates.
5. Weight decay is decoupled in AdamW to produce more predictable generalization.
6. we decay the learning rate to help fine-tune model near convergence.
7. If learning rate is too high the training becomes unstable and may diverge. If learning rate is low the training becomes too slow and may get stuck.

### Training Analysis & Research

1. If the training and validation loss decreases continously we know the model is learning.
2. We detect overfitting if the training loss decreases and the validation loss is stuck or the gap between the two increases.
3. Perplexity is useful because it converts loss into more interpretable prediction - quality metric.
4. Additional experiments to run to verify results more seeds, longer training and long - context evaluation.
5. We can validated that rope works on long context by training in short context and evaluating the model on longer context.
6. we will measure loss, perplexity , throughput, memory usage and long - context performance.
7. potential weakness of rope is performance can still degrade for really long contexts.