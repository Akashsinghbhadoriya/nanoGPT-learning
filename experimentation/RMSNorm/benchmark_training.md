# Trained a model from scratch with rope + RMSNorm implementation

The training was done a mac M4 and the configurations of the training were as follows:

1. python file for training -> train.py
2. configuration -> config/train_shakespeare_char.py
3. This the gpt2 architecture from the nanoGPT which is modified by removing absolute positional embeddings and using Rotary Positional Embeddings also layerNormalization is removed and we use the RMSNorm.
4. command for training `python3 train_rope.py config/train_shakespeare_char_rope.py`

### Training observations

1. Model is learning because the loss dropped from 4.1979 -> 1.9073
2. Training and validation losses are close so the model is not overfitting on training data and regularization is working.
3. most rapid learning in the 1st 100 steps loss from 4.1979 to 2.3848.
4. loss curve is still descending that means if we train further the loss will fall and the model will learn.
5. As discussed in the RMSNorm paper the assumption that re-centering does not impact the layernorm performance it is true since the training and validation losses are similar to that we had when we were using layernormalization even though we removed the re-centering.
### Training Data
```
tokens per iteration will be: 16,384
found vocab_size = 65
Initializing a new model from scratch
number of parameters: 10.65M
num decayed parameter tensors: 25, with 10,641,792 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: False
step 0: train loss 4.1979, val loss 4.2041
iter 0: loss 4.2163, time 40632.86ms, mfu -100.00%
iter 10: loss 3.4410, time 1970.20ms, mfu 0.19%
iter 20: loss 3.0326, time 2001.26ms, mfu 0.19%
iter 30: loss 2.7473, time 2092.32ms, mfu 0.19%
iter 40: loss 2.6612, time 2081.06ms, mfu 0.19%
iter 50: loss 2.6118, time 2073.70ms, mfu 0.19%
iter 60: loss 2.5869, time 2060.92ms, mfu 0.19%
iter 70: loss 2.5273, time 2191.82ms, mfu 0.18%
iter 80: loss 2.4923, time 2282.74ms, mfu 0.18%
iter 90: loss 2.4565, time 2329.04ms, mfu 0.18%
step 100: train loss 2.3848, val loss 2.3995
saving checkpoint to out-shakespeare-char-rope
iter 100: loss 2.4233, time 56530.17ms, mfu 0.16%
iter 110: loss 2.3873, time 2308.90ms, mfu 0.16%
iter 120: loss 2.3682, time 2445.44ms, mfu 0.16%
iter 130: loss 2.3400, time 2417.22ms, mfu 0.16%
iter 140: loss 2.3327, time 2475.48ms, mfu 0.16%
iter 150: loss 2.3132, time 2481.62ms, mfu 0.16%
iter 160: loss 2.2810, time 2459.39ms, mfu 0.16%
iter 170: loss 2.2484, time 2431.80ms, mfu 0.16%
iter 180: loss 2.2112, time 2354.28ms, mfu 0.16%
iter 190: loss 2.2045, time 2459.69ms, mfu 0.16%
step 200: train loss 2.1235, val loss 2.1641
saving checkpoint to out-shakespeare-char-rope
iter 200: loss 2.1998, time 58630.57ms, mfu 0.14%
iter 210: loss 2.1703, time 2429.08ms, mfu 0.14%
iter 220: loss 2.1449, time 2380.85ms, mfu 0.14%
iter 230: loss 2.1109, time 2374.95ms, mfu 0.15%
iter 240: loss 2.0927, time 2402.30ms, mfu 0.15%
iter 250: loss 2.0817, time 2949.28ms, mfu 0.14%
iter 260: loss 2.0637, time 2508.00ms, mfu 0.14%
iter 270: loss 2.0637, time 2458.68ms, mfu 0.15%
iter 280: loss 2.0413, time 2509.68ms, mfu 0.15%
iter 290: loss 2.0395, time 2503.42ms, mfu 0.15%
step 300: train loss 1.9241, val loss 1.9985
saving checkpoint to out-shakespeare-char-rope
iter 300: loss 2.0081, time 57992.48ms, mfu 0.13%
iter 310: loss 2.0005, time 2505.70ms, mfu 0.13%
iter 320: loss 1.9747, time 2510.03ms, mfu 0.14%
iter 330: loss 1.9932, time 2681.44ms, mfu 0.14%
iter 340: loss 1.9680, time 2716.21ms, mfu 0.14%
iter 350: loss 1.9648, time 2543.87ms, mfu 0.14%
iter 360: loss 1.9632, time 2684.98ms, mfu 0.14%
iter 370: loss 1.9714, time 2724.26ms, mfu 0.14%
iter 380: loss 1.9448, time 2563.04ms, mfu 0.14%
iter 390: loss 1.9589, time 2734.20ms, mfu 0.14%
step 400: train loss 1.8380, val loss 1.9391
saving checkpoint to out-shakespeare-char-rope
iter 400: loss 1.9420, time 60568.30ms, mfu 0.12%
iter 410: loss 1.9234, time 2542.94ms, mfu 0.13%
iter 420: loss 1.9283, time 2563.32ms, mfu 0.13%
iter 430: loss 1.9201, time 3526.10ms, mfu 0.13%
iter 440: loss 1.9345, time 2841.85ms, mfu 0.13%
iter 450: loss 1.9367, time 2820.20ms, mfu 0.13%
iter 460: loss 1.8995, time 2581.26ms, mfu 0.13%
iter 470: loss 1.9075, time 2577.21ms, mfu 0.13%
iter 480: loss 1.9235, time 3702.06ms, mfu 0.13%
iter 490: loss 1.9189, time 2691.37ms, mfu 0.13%
step 500: train loss 1.8099, val loss 1.9183
saving checkpoint to out-shakespeare-char-rope
iter 500: loss 1.9073, time 61118.11ms, mfu 0.12%
```