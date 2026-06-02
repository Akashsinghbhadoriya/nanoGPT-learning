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