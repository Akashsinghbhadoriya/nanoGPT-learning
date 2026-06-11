# Quantization 

The details about this folder is as follows:
1. model_quantization.py -> Contains the INT8.LLm and W8A8 quantization for all the transformer layer and the code for quantizing all the layers in a model.
2. quantization.md -> theory about all the methods of quantization.
3. quantization.py -> Smmothquant, LinearInt8 and mixed precision quantization implementation.
4. wonly_int4.py -> groupwise int 4 quantization

For testing any quantization method on the gpt2 model we can use the `evaluate_quantized_model()` function from the `model_quantization.py` also for testing different techniques of quantization make changes to the `quantize_model()` function in the above python file.