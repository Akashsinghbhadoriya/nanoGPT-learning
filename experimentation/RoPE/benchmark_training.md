# Trained a model from scratch with rope implementation

The training was done a mac M4 and the configurations of the training were as follows:

1. python file for training -> train.py
2. configuration -> config/train_shakespeare_char.py
3. This the gpt2 architecture from the nanoGPT which is modified by removing absolute positional embeddings and using Rotary Positional Embeddings.
4. command for training `python3 train_rope.py config/train_shakespeare_char_rope.py`

### Training observations

1. Model is learning because the loss dropped from 4.1938 -> 1.9074
2. Training and validation losses are close so the model is not overfitting on training data and regularization is working.
3. most rapid learning in the 1st 100 steps loss from 4.1938 to 2.3860.
4. loss curve is still descending that means if we train further the loss will fall and the model will learn.
5. perplexity dropped from 66 → 6.

### Comparison between Rope and positional embedding model

Replacing learned absolute positional embeddings with Rotary Position Embeddings substantially improved optimization and generalization on the Shakespeare character-level language modeling task. Validation loss improved from 2.33 to 1.92 after 500 training iterations, corresponding to a perplexity reduction from approximately 10.3 to 6.8. The model exhibits faster convergence, lower validation error, and no signs of increased overfitting.

1. Rope learns faster.
2. training continues improving aggresively.
3. Rope perplexity is much better.
4. Generated text from rope trained model will be much better than the other one.

### training data

```tokens per iteration will be: 16,384
found vocab_size = 65
Initializing a new model from scratch
number of parameters: 10.65M
num decayed parameter tensors: 25, with 10,641,792 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: False
step 0: train loss 4.1938, val loss 4.2005
iter 0: loss 4.2140, time 39477.73ms, mfu -100.00%
iter 10: loss 3.4401, time 1838.43ms, mfu 0.20%
iter 20: loss 3.0300, time 1862.77ms, mfu 0.20%
iter 30: loss 2.7470, time 1922.55ms, mfu 0.20%
iter 40: loss 2.6613, time 1906.55ms, mfu 0.20%
iter 50: loss 2.6116, time 1991.00ms, mfu 0.20%
iter 60: loss 2.5870, time 1947.46ms, mfu 0.20%
iter 70: loss 2.5274, time 1935.31ms, mfu 0.20%
iter 80: loss 2.4928, time 1968.61ms, mfu 0.20%
iter 90: loss 2.4572, time 1952.74ms, mfu 0.20%
step 100: train loss 2.3860, val loss 2.4009
saving checkpoint to out-shakespeare-char-rope
iter 100: loss 2.4245, time 49671.09ms, mfu 0.18%
iter 110: loss 2.3888, time 2190.14ms, mfu 0.18%
iter 120: loss 2.3701, time 2131.09ms, mfu 0.18%
iter 130: loss 2.3425, time 2131.70ms, mfu 0.18%
iter 140: loss 2.3356, time 2125.29ms, mfu 0.18%
iter 150: loss 2.3148, time 2149.56ms, mfu 0.18%
iter 160: loss 2.2819, time 2159.94ms, mfu 0.18%
iter 170: loss 2.2485, time 2163.22ms, mfu 0.18%
iter 180: loss 2.2122, time 2172.14ms, mfu 0.18%
iter 190: loss 2.2026, time 2169.32ms, mfu 0.17%
step 200: train loss 2.1190, val loss 2.1602
saving checkpoint to out-shakespeare-char-rope
iter 200: loss 2.1955, time 59281.34ms, mfu 0.16%
iter 210: loss 2.1692, time 2431.03ms, mfu 0.16%
iter 220: loss 2.1445, time 2374.66ms, mfu 0.16%
iter 230: loss 2.1124, time 2233.27ms, mfu 0.16%
iter 240: loss 2.0919, time 2379.75ms, mfu 0.16%
iter 250: loss 2.0807, time 2321.85ms, mfu 0.16%
iter 260: loss 2.0631, time 2320.08ms, mfu 0.16%
iter 270: loss 2.0626, time 2275.80ms, mfu 0.16%
iter 280: loss 2.0397, time 2391.96ms, mfu 0.16%
iter 290: loss 2.0385, time 2364.06ms, mfu 0.16%
step 300: train loss 1.9237, val loss 1.9967
saving checkpoint to out-shakespeare-char-rope
iter 300: loss 2.0085, time 54391.99ms, mfu 0.14%
iter 310: loss 2.0004, time 2230.13ms, mfu 0.15%
iter 320: loss 1.9738, time 2246.90ms, mfu 0.15%
iter 330: loss 1.9942, time 2229.19ms, mfu 0.15%
iter 340: loss 1.9681, time 2214.41ms, mfu 0.15%
iter 350: loss 1.9654, time 2207.77ms, mfu 0.15%
iter 360: loss 1.9607, time 2216.80ms, mfu 0.15%
iter 370: loss 1.9740, time 2216.32ms, mfu 0.16%
iter 380: loss 1.9450, time 2337.65ms, mfu 0.16%
iter 390: loss 1.9562, time 2226.53ms, mfu 0.16%
step 400: train loss 1.8377, val loss 1.9360
saving checkpoint to out-shakespeare-char-rope
iter 400: loss 1.9406, time 52746.58ms, mfu 0.14%
iter 410: loss 1.9249, time 2246.04ms, mfu 0.14%
iter 420: loss 1.9259, time 2246.70ms, mfu 0.15%
iter 430: loss 1.9218, time 2239.60ms, mfu 0.15%
iter 440: loss 1.9351, time 2242.06ms, mfu 0.15%
iter 450: loss 1.9367, time 2237.26ms, mfu 0.15%
iter 460: loss 1.9006, time 2263.86ms, mfu 0.15%
iter 470: loss 1.9090, time 2247.80ms, mfu 0.15%
iter 480: loss 1.9241, time 2264.52ms, mfu 0.16%
iter 490: loss 1.9189, time 2226.66ms, mfu 0.16%
step 500: train loss 1.8105, val loss 1.9168
saving checkpoint to out-shakespeare-char-rope