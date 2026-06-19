from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
import json

ds = load_dataset("tatsu-lab/alpaca")
print(ds)

sub_dataset = ds['train'].select(range(5000))

split_data = sub_dataset.train_test_split(test_size=0.1, train_size=0.9, seed=42)
train_data = split_data['train']
val_data = split_data['test']


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def build_prompt(instruction, input, output):

    prompt_response = f"""
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    Instruction:
    {instruction}

    Input:
    {input}

    Response:
    {output}
"""
    
    prompt = f"""
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    Instruction:
    {instruction}

    Input:
    {input}

    Response:
"""
    
    return prompt_response, prompt

def encode_and_save(dataset_split, filename):

    pad_token = tokenizer.eos_token_id
    data_list = []
    max_length = 0
    outlier_count = 0
    for data in dataset_split:
        prompt_reponse, prompt = build_prompt(
            data.get('instruction'), 
            data.get('input'), 
            data.get('output')
        )

        input_ids = tokenizer.encode(prompt_reponse)
        if len(input_ids) > 1024:
            outlier_count += 1
            continue
        if len(input_ids) > max_length:
            max_length = len(input_ids)
        labels = input_ids.copy()
        prompt_ids = tokenizer.encode(prompt)
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
        data_list.append({"input_ids":input_ids, "labels": labels})
    
    all_input_ids = []
    all_labels = []

    for data in data_list:
        input_id = data['input_ids']
        label = data['labels']
        padding_needed = max_length - len(input_id)
        if padding_needed > 0:
            input_id = input_id + [pad_token] * padding_needed
            label = label + [-100] * padding_needed
        all_input_ids.extend(input_id)
        all_labels.extend(label)

    all_input_ids_array = np.array(all_input_ids, dtype=np.int32)
    all_label_array = np.array(all_labels, dtype=np.int32)
    input_filename = "input_" + filename
    label_filename = "label_" + filename
    config_name = filename.removesuffix(".bin") + "_dataset_config.json"
    all_input_ids_array.tofile(input_filename)
    all_label_array.tofile(label_filename)
    metadata = {
        "max_length": max_length,
        "vocab_size_type": "int32",
        "total_elements": len(all_input_ids)
    }
    with open(config_name, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Successfully wrote {len(all_input_ids_array):,} tokens to {input_filename}")
    print(f"✅ Successfully wrote {len(all_label_array):,} tokens to {label_filename}")

encode_and_save(train_data, "train.bin")
encode_and_save(val_data, "val.bin")