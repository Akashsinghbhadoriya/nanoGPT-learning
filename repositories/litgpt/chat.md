### Litgpt chat model-name

#### Steps executed when running the above command 

command: `litgpt chat meta-llama/Llama-3.2-3B-Instruct`

Hitting the above command check the pyproject.toml and look for `scripts.litgpt = "litgpt.__main__:main"`.
This calls the `__main__.py` inside the litgpt folder

1. CLI(PARSER_DATA) -> `litgpt/__main__.py`

Purpose: Convert the `chat` into proper python function to call
Input: `CLI(PARSER_DATA)`
```
PARSER_DATA = {
    "download": download_fn,
    "chat": chat_fn,
    "finetune": finetune_lora_fn,
    "finetune_lora": finetune_lora_fn,
    "finetune_full": finetune_full_fn,
    "finetune_adapter": finetune_adapter_fn,
    "finetune_adapter_v2": finetune_adapter_v2_fn,
    "pretrain": pretrain_fn,
    "generate": generate_base_fn,
    "generate_full": generate_full_fn,
    "generate_adapter": generate_adapter_fn,
    "generate_adapter_v2": generate_adapter_v2_fn,
    "generate_sequentially": generate_sequentially_fn,
    "generate_speculatively": generate_speculatively_fn,
    "generate_tp": generate_tp_fn,
    "convert_to_litgpt": convert_hf_checkpoint_fn,
    "convert_from_litgpt": convert_lit_checkpoint_fn,
    "convert_pretrained_checkpoint": convert_pretrained_checkpoint_fn,
    "merge_lora": merge_lora_fn,
    "evaluate": evaluate_fn,
    "serve": serve_fn,
    "validate": validate_fn,
}
```
Output: `from litgpt.chat.base import main as chat_fn` calls the main function of from the `litgpt/chat/base.py`

2. chat_fn()

Purpose: The is the main function which intialize the model loads the checkpoint based on the model config encodes the user given prompt and generates an interactive CLI for the user to interact with the loaded model.

Input: Model config is given as the input.
Output: CLI Chat interface to use the loaded model and chat with it.

3. Model Download

Purpose: We need to check wether the model being downloaded is already their in the `checkpoints` folder or not if not we will download the model from hugging face. The function used is `auto_download_checkpoint` from `litgpt/utils.py`

Input: `model_name, access_token, ignore_tokenizer_files` we use `download_from_hub` function from `litgpt/scripts/download.py`
Output: We download the model inside the folder `checkpoints/provider/modelname`

4. model loading

Purpose: Initialize the model with the `config.yaml` from the checkpoints folder. The model definition is in the `litgpt/model.py` the `GPT` class is the one which initialize the repo. in model.forward we are using clamp head for the output logits
Input: We give the config as the input this config is generated using the `Config` class from the `litgpt/config.py`
Output: Loads the model with proper transformer architecture by using the details from the config

5. checkpoint loading

Purpose: For loading the checkpoint we are using the `litgpt/utils.py`. The checkpoint can be loaded in 3 different ways Fully Sharded Data Parallel (FSDPStrategy), Tensor and Pipeline Parallelism (ModelParallelStrategy), Single-GPU, CPU, or Standard Data Parallel 
Input: The input is passed (fabric, model, checkpoint_path) 
Output: Loads the state_dict into the model from the checkpoint which was downloaded

6. prompt encoding

Purpose: For encoding we use the tokenizer from the `litgpt/tokenizer.py` We load the tokenizer using the tokenizer.json config from the checkpoints folder also we reconstruct the tokenizer pipeline in the above file. This tokenizer also has a decoder stream for generating output for one token at a time instead of generating it once.
Input: We pass the string to be encoded
Output: We get the list of tokenids which will be passed to the model for generation.

7. generation loop

Purpose: We use the `interact()` function from the `litgpt/chat/base.py` for creating an interactive cli chat where user enters the query it groups the multiline query into single line as well and pass to the `process_prompt()` function in the same file to generate the output and the output is streamed one work at a time so that user does not have to wait before generating the complete output.
Input: takes user query
Output: generates an output reponse based on the user query.

