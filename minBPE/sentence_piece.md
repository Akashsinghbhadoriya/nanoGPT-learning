## SentencePiece

It is a simple language independent text tokenizer and detokenizer mainly for neural network based text generation. The size of the vocabulary is predetermined prior to Neural Model Training. Sentencepiece implements 2 subword segmentation algorithms. This allows for end to end system that does not depend on any language specific processing.\
1. byte-pair encoding (BPE)
2. Unigram language model

#### Sentencepiece has 4 main components:
1. Normalizer -> Module to normalize semantically equivalent Unicode characters into canonical forms.
2. Trainer -> Trainer trains the subword segmentation model from the normalized corpus.
3. Encoder -> Encoder internally executes Normalizer to normalize the input text and tokenizes it into subword sequence with subword model trained by the trainer.
4. Decoder -> Decoder converts the sub-word sequence into the normalized text.

#### Lossless tokenization

• Raw text: Hello world.\
• Tokenized: [Hello] [world] [.]\
One observation is that the raw text and tokenized sequence are not reversibly convertible. The information that no space exists between “world” and “.” is not kept in the tokenized sequence. Detokenization, a process to restore the original raw input from the tokenized sequence, has to be language-dependent due to these irreversible operations. For example, while the detokenizer usually puts whitespaces between the primitive tokens in most European languages, no spaces are required in Japanese and Chinese.\
• Raw text: [こんにちは世界。] (Hello world.)\
• Tokenized: [こんにちは] [世界] [。]\
Such language specific processing has usually been implemented in manually crafted rules, which are expensive to write and maintain.SentencePiece implements the Decoder as an inverse operation of Encoder, i.e.,
$Decode(Encode(Normalize(text))) = Normalize(text)$
We call this design lossless tokenization, 

#### Core logic

BPE starts from the individual bytes and builds the vocabulary up by combining in pairs so the vocabulary size increases. Unigram does the opposite it builds a large vocab and then reduce its size one by one by removing the least bad ones means the ones with less probabilities.

Suppose our vocabulary is:\
```
h
he
hel
hello
l
lo
o
```

Now we want to tokenize "hello"\
Option 1:- h e l l o (5 tokens)\
Option 2:- he l l o (4 tokens)\
Option 3:- hel lo (3 tokens)\
Option 4:- hello (1 token)\

Now sentencepiece asks which segmentation is most probable and choses it:\
Suppose after analyzing the corpus:\
```
hello  = very common
hel    = common
he     = common
h      = common

Probabilites:
hello = 0.4
hel   = 0.2
he    = 0.1
h     = 0.05

So probas for each option is calculated and the one with maximum probability is chosen
```

#### Training Intuition

```
Suppose the Vocabulary is:
hello
hel
he
world
wor
ld

Now ask: If I delete "hello", does tokenization get much worse?
Maybe.
Because: hello 
would become: hel + lo
which is less efficient. Loss increases.
So: "hello" is important. Keep it.

Now ask: If I delete "wor"?
Nothing changes because: "world" already exists. Loss barely changes.
So: "wor" is useless. Delete it.
```

```
Where Viterbi Comes In?
Suppose: "unhug"
Vocabulary:
un
hug
u
n
h
ug

Possible segmentations:
u + n + h + ug
un + h + ug
un + hug

SentencePiece must find: best segmentation without checking every possibility manually. Viterbi is just a dynamic programming algorithm that efficiently finds: highest probability path through all segmentations.
```