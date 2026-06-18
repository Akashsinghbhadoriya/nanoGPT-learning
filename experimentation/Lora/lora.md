# Lora (Low Rank Adaptation of Large Language Models)

Applications in NLP rely on adapting one large-scale pre-trained model to multiple downstream applications. Such adaptations is done using Fine-tuning but it is quite expensive and daunting task for larger models such as GPT-3 175B parameters so people tried by adapting only some parameters or learning external modules for new tasks. But these methods often introduce inference latency by increasing the model depth or reduce model's usable sequence length.

LoRA possesses several key advantages.
- A pre-trained model can be shared and used to build many small LoRA modules for different tasks. We can freeze the shared model and efficiently switch tasks by replacing the matrices A and B in Figure 1, reducing the storage requirement and task-switching overhead significantly.
- LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.
- Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.
- LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning. We provide an example in Appendix E.
### Simple Intution
```
The Core Problem Suppose you have a pretrained model. Inside it is a weight matrix: W
Maybe: W.shape = (4096, 4096)
This matrix already contains all the knowledge learned during pretraining. Now you want to teach the model something new:
Medical QA, Legal documents, Your company data, Kabaddi analysis
Traditional fine-tuning says: Modify W directly
So during training: W = W + ΔW
The problem: W has millions of parameters
For a 7B model: Billions of trainable parameters, Huge optimizer states, Huge memory usage

LoRA's Key Observation Researchers looked at the actual updates made during fine-tuning. They found: The model doesn't need to change everything. Most of the change can be represented by a much smaller structure.
Imagine this: You have a giant encyclopedia. To specialize it for medicine, you don't rewrite the entire book.
You add: A few pages of corrections LoRA is those correction pages.

Instead of Updating W
Normally: W_new = W + ΔW
LoRA says: Freeze W Never touch it.
Instead learn: ΔW separately.
So: W_new = W + ΔW
W is frozen
ΔW is trainable
Already a huge win.

But ΔW Is Still Huge
Suppose: W.shape = (4096,4096)
Then: ΔW.shape = (4096,4096)
Still: 16 million parameters Not good. The Brilliant Trick
LoRA says: Don't learn ΔW directly. Approximate it.
Instead of: ΔW
learn: B @ A
where: A.shape = (16,4096) B.shape = (4096,16)
Notice: 16 is tiny 4096 is huge
Instead of training: 16 million parameters
you train: 4096×16 + 16×4096 ≈ 131k parameters
Massive reduction. Why Does This Work? Think geometrically. 
Suppose you have a huge image: 4000 x 4000 pixels
Most information isn't random. There are patterns.
Similarly: The update needed for fine-tuning is much simpler than the original model. 
The model already knows: Language, Reasoning, Grammar, Facts You're only nudging it slightly.
So the update has: Low complexity and can be compressed. That's what "low rank" means.
What Are A and B Doing?
Think of them as:
A = Compress
B = Expand
Input: x
First: x @ A.T compresses information.
Example:
4096 dimensions
↓
16 dimensions
Then: @ B.T expands it back:
16 dimensions
↓
4096 dimensions

So LoRA learns: A tiny detour around the frozen weights 

Visual Intuition
Without LoRA:

Input
   |
   v
   W
   |
Output

With LoRA:

Input
   |
   +----> A ----> B ----+
   |                    |
   v                    |
   W (frozen)           |
   |                    |
   +---------+----------+
             |
          Output

The original model still works. LoRA adds a small correction.
Forward Pass
Normal linear layer: y = xWᵀ
LoRA: y = xWᵀ + xAᵀBᵀ
Or: y = base + lora_update

where: 
base = F.linear(x,W)
lora_update = x @ A.T
lora_update = lora_update @ B.T
y = base + lora_update

That's literally the entire forward pass.
Why Freeze W? This is the biggest memory saving.
Normally:
Need gradients for W
Need optimizer states for W
Need momentum for W

For billions of parameters that's enormous.

LoRA:
W.requires_grad = False
Now only: A B need gradients.
Memory drops dramatically.
What Gets Saved?
Traditional fine-tuning: Save entire model
Maybe: 14 GB

LoRA:
Save only: A B
Maybe: 50 MB or less.

That's why people share LoRA adapters.

The One-Sentence Mental Model
Whenever you see LoRA, think: Take a frozen pretrained model. Instead of changing the giant weight matrix, learn a tiny correction matrix represented as B @ A and add it during the forward pass.

```
1. Rank(r) -> A(r, d_in) B(d_out, r) the value of r is called rank. Rnak tells how much freedom does lora have to modify the model. small rank means less parameters, memory and less expressive. Large rank means more parameters, more memory, more expressive.
2. On the Initial load we initialize B = 0 and A to some random vector so that together BA = 0 and when we add to W before training the model starts as a pretrained model.
3. Alpha scaling -> W + (alpha / r)BA since rank changes the magnitude so to normalize the update alpha was introduced.
4. Lora Dropout -> We use lora dropout sometimes to reduce overfitting on smaller datasets.
5. Lora is applied to which layers -> Attention(Q, K, V, Out_proj) MLP(LN1, LN2). Most commonly Q, V since researchers found that most adaptation happens there and gives great performance and very few trainable parameters.
6. Adapter Injection -> we do not modify the model but use the below implementation for replacing the linear layers this is also the idea behind PEFT
```
for name, module in model.named_modules():

    if name in target_modules:

        replace_with_lora(module)
```
7. saving the adapters -> Instead of saving the model.pt we save adapter.pt which contains the A, B values.
8. Loading adapters -> Base Model + Lora Adapters.
9. After training $\Delta W=BA$ we replace the $Wmerged = W + BA$ and Inference becomes $y = xWmerged$
10. Multiple Loras -> This is where LoRA becomes really powerful we can combine the base model with any Medical, Legal, Coding Lora.
