# Tokenizer

It is used to convert the plain text into token ids using BPE(Byte pair encoding algorithm) which merges the pairs which are occuring consecutively into one. This tokenizer is used to convert raw text to token ids to feed the llm and the final generated token ids are further converted back to raw text using this tokenizer. The vocab size of the tokenizer determines how many unique token ids it contains which are generated using the BPE algorithm.

#### BPE(Byte Pair Encoding)

```
step 1: Convert the training data into raw bytes using utf-8 encoding
step 2: Then we convert the raw bytes to ids or list of integers from 0 - 255
step 3: Then we use this list of ids and generate the dictionary of most frequently occuring pairs.
step 4: most frequently occuring pair is given a new idx which start from 256, 257, ....
step 5: The most frequently occuring pair is merged in the original list and replaced with this new id.
step 6: We store the most frequently occuring pair for encoding and decoding later.
step 6: step 3, 4, 5 are repeated for num_merges = vocab_size-256
```

#### GPT2 and GPT4 regex

These regex were introduced to split the text before training into meaningfull chunks and then train the model so that we are not generating pairs like h1, l1 which might be arbitrary. 

So in GPT2 regex the task was to split words, numbers, punctuations and spaces.\
In GPT4 regex is more sophisticated to handle Unicode Languages, Numbers, Repeated Punctuation, Special Symbol, Emojis.
