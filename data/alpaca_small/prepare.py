from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

ds = load_dataset("tatsu-lab/alpaca")
print(ds)

sub_dataset = ds['train'].select(range(1000))
print(sub_dataset)

split_data = sub_dataset.train_test_split(test_size=0.1, train_size=0.9, seed=42)
train_data = split_data['train']
val_data = split_data['test']
print(train_data)
print(val_data)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encode_and_save(dataset_split, filename):

    text_list = dataset_split['text']
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    joined_text = eos_token.join(text_list) + eos_token

    token_ids = tokenizer.encode(joined_text)
    ids_array = np.array(token_ids, dtype=np.uint16)

    ids_array.tofile(filename)
    print(f"✅ Successfully wrote {len(ids_array):,} tokens to {filename}")

encode_and_save(train_data, "train.bin")
encode_and_save(val_data, "val.bin")