# Trained a model from scratch using shakespeare char data

The training was done a mac M4 and the configurations of the training were as follows:

1. python file for training -> train.py
2. configuration -> config/train_shakespeare_char.py
3. This the gpt2 architecture from the nanoGPT which is based on the Attention is all you need paper.
4. command for training `python3 train.py config/train_shakespeare_char.py`

### Training observations

1. Model is learning because the loss dropped from 4.2882 -> 2.3697
2. Training and validation losses are close so the model is not overfitting on training data and regularization is working.
3. most rapid learning in the 1st 100 steps loss from 4.2882 to 2.4989.
4. loss curve is still descending that means if we train further the loss will fall and the model will learn.
5. perplexity dropped from 72 → 10.


### training data

```tokens per iteration will be: 16,384
found vocab_size = 65 
Initializing a new model from scratch
number of parameters: 10.65M
step 0: train loss 4.2882, val loss 4.2813
iter 0: loss 4.2661, time 33197.22ms, mfu -100.00%
iter 10: loss 3.4788, time 1705.53ms, mfu 0.22%
iter 20: loss 3.1577, time 1675.87ms, mfu 0.22%
iter 30: loss 2.8780, time 1721.61ms, mfu 0.22%
iter 40: loss 2.7190, time 1887.46ms, mfu 0.22%
iter 50: loss 2.6496, time 1701.52ms, mfu 0.22%
iter 60: loss 2.5860, time 1721.87ms, mfu 0.22%
iter 70: loss 2.5653, time 1727.00ms, mfu 0.22%
iter 80: loss 2.5434, time 1729.94ms, mfu 0.22%
iter 90: loss 2.5483, time 1825.68ms, mfu 0.22%
step 100: train loss 2.4989, val loss 2.5072
saving checkpoint to out-shakespeare-char
iter 100: loss 2.5311, time 44996.39ms, mfu 0.19%
iter 110: loss 2.5264, time 10904.19ms, mfu 0.18%
iter 120: loss 2.5032, time 2099.90ms, mfu 0.18%
iter 130: loss 2.5002, time 2091.04ms, mfu 0.18%
iter 140: loss 2.4792, time 2102.75ms, mfu 0.18%
iter 150: loss 2.5031, time 2126.74ms, mfu 0.18%
iter 160: loss 2.4845, time 2144.97ms, mfu 0.18%
iter 170: loss 2.4914, time 2233.21ms, mfu 0.18%
iter 180: loss 2.4662, time 2229.62ms, mfu 0.18%
iter 190: loss 2.4611, time 2269.97ms, mfu 0.17%
step 200: train loss 2.4448, val loss 2.4613
saving checkpoint to out-shakespeare-char
iter 200: loss 2.4631, time 51572.00ms, mfu 0.16%
iter 210: loss 2.4603, time 2230.54ms, mfu 0.16%
iter 220: loss 2.4627, time 2454.28ms, mfu 0.16%
iter 230: loss 2.4679, time 2356.62ms, mfu 0.16%
iter 240: loss 2.4405, time 2279.08ms, mfu 0.16%
iter 250: loss 2.4478, time 2383.56ms, mfu 0.16%
iter 260: loss 2.4185, time 2252.62ms, mfu 0.16%
iter 270: loss 2.4477, time 2565.25ms, mfu 0.16%
iter 280: loss 2.4247, time 2378.09ms, mfu 0.16%
iter 290: loss 2.4233, time 2426.10ms, mfu 0.16%
step 300: train loss 2.3945, val loss 2.4105
saving checkpoint to out-shakespeare-char
iter 300: loss 2.4106, time 52517.82ms, mfu 0.14%
iter 310: loss 2.4281, time 2315.22ms, mfu 0.14%
iter 320: loss 2.4133, time 2235.39ms, mfu 0.15%
iter 330: loss 2.4075, time 2338.58ms, mfu 0.15%
iter 340: loss 2.4154, time 2129.34ms, mfu 0.15%
iter 350: loss 2.4071, time 2155.61ms, mfu 0.15%
iter 360: loss 2.4042, time 2137.99ms, mfu 0.15%
iter 370: loss 2.4060, time 2176.32ms, mfu 0.16%
iter 380: loss 2.3835, time 2078.20ms, mfu 0.16%
iter 390: loss 2.3965, time 2073.96ms, mfu 0.16%
step 400: train loss 2.3432, val loss 2.3612
saving checkpoint to out-shakespeare-char
iter 400: loss 2.3743, time 49071.98ms, mfu 0.15%
iter 410: loss 2.3736, time 2048.43ms, mfu 0.15%
iter 420: loss 2.3745, time 2225.94ms, mfu 0.15%
iter 430: loss 2.3734, time 2242.31ms, mfu 0.15%
iter 440: loss 2.4017, time 2217.40ms, mfu 0.15%
iter 450: loss 2.3781, time 2114.84ms, mfu 0.16%
iter 460: loss 2.3546, time 2232.22ms, mfu 0.16%
iter 470: loss 2.3862, time 2116.83ms, mfu 0.16%
iter 480: loss 2.3440, time 2193.15ms, mfu 0.16%
iter 490: loss 2.3742, time 2073.40ms, mfu 0.16%
step 500: train loss 2.3098, val loss 2.3338
saving checkpoint to out-shakespeare-char
iter 500: loss 2.3697, time 50348.15ms, mfu 0.15%```