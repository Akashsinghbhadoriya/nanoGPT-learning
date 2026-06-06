### Latest changes from litgpt to implement in our repo

1. use torch.inference_mode() instead of torch.auto_grad() it is designed to completely disable auto_grad and stripping away internal tracking overhead.
2. when generating text or during inference we can also use top_p also known as nucleus sampling along with top_k and temperature. top_p is the sum of probablities of the top tokens to be considered 0<=top_p<=1. In practice it is very useful to generate contextually relevant output.
