# Rotary Positional Embedding (RoPE)

RoPE is a technique used in modern transformer models to encode token positions by rotating the query and key vectors in high-dimensional space. It natively encodes relative distances making it gold standard for LLMs for using large context. It alters the orientation of the vector and not the magnitude.

The core implementations is:

1. Pairing Dimensions : The embedding vector is split in to pairs of dimensions.
2. Rotation : Each pair is treated as 2D complex number and is rotated by an angle determined by the absolute position and dimension index of the vector.
3. Relative Relationships : when the self - attention calculates the dot product of the rotated query and key vector the absolute positions are canceled out and we are left with only the relative postions

### practical significance

1. Rope provides positional information while improving models ability to generalize across longer context.
2. Learned positional embeddings are tied to positions seen during the training and for longer context the performance degrades where the trainging context length is smaller than the one provided in the input.
3. Rope encodes position without adding embeddings by rotating the query and key vectors based on the position. It preserves vector magnitude while introducing positional information. This allows relative positions to automaticallly emerge in the attention vectors.
4. Rotations are only applied to query and key because position must influence token relationships not the actual value if we rotate the value it will degrade the model performance also dot product calculates the relative position between query and key as it give relative rotation.
5. The relative angle between token rotations affect the similarity scores
6. If we have same frequency for all the dimension the positional information will become repetitive and less expressive. Different frequencies capture both local and long-range positional information. Rope generalizes over long range because it uses a mathematical pattern rather than positional ids making the model extrapolate beyond training lengths.
7. the Rope should be applied before attention because after that the token relationships will be fixed.
8. Lower validation loss and improved convergence compared to baseline model convinced me that rope is working.

Detailed Explaination:

### Sinusoidal positional embeddings

formula 
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

pos : absolute position of the word in the sequence\
i : 0, 1, 2, ..., d/2-1 (since the dimension is divided in pairs of 2)\
d : dimension of the model\
Note: we use sine for even position and cosine for odd position

The problem with sinusoidal waves is it does not have predictable behaviour also the absolute positions are affected since we are adding the positional encoding to the token encoding. The perplexity of the model skyrockets or explodes beyong the training length (sequence length). This perplexity explosion happens for both sinusoidal and absolute positional embeddings So We use RoPE to fix this issue.

### RoPE (approach)

We again divide the dimensions into pairs of 2 where each pair is rotated by a certain angle $\theta_i$.

### 1. Vector Partitioning
A high-dimensional vector $\mathbf{x} \in \mathbb{R}^d$ (where $d$ is the attention head dimension) is divided into $d/2$ independent two-dimensional vectors:

$$
\mathbf{x}=\left(\begin{matrix}x_{1}\\ x_{2}\\ x_{3}\\ x_{4}\\ \vdots \\ x_{d-1}\\ x_{d}\end{matrix}\right)\rightarrow \mathbf{x}^{(i)}=\left(\begin{matrix}x_{2i-1}\\ x_{2i}\end{matrix}\right)\quad \text{for\ }i\in \left\{1,2,\dots ,\frac{d}{2}\right\}
$$

### 2. Angle Calculation
For a token at absolute position index $m$, the rotation angle $\theta _{i}$ for the $i$-th two-dimensional slice is calculated using a geometric progression:

$$
\theta _{i}=10000^{-2(i-1)/d}
$$

### 3. The 2D Rotation Formula
Each 2D vector slice is multiplied by a 2D rotation matrix determined by its position $m$ and frequency component $\theta _{i}$:

$$
\mathbf{R}_{\Theta ,m}^{(i)}\mathbf{x}^{(i)}=\left(\begin{matrix}\cos (m\theta _{i})&-\sin (m\theta _{i})\\ \sin (m\theta _{i})&\cos (m\theta _{i})\end{matrix}\right)\left(\begin{matrix}x_{2i-1}\\ x_{2i}\end{matrix}\right)=\left(\begin{matrix}x_{2i-1}\cos (m\theta _{i})-x_{2i}\sin (m\theta _{i})\\ x_{2i-1}\sin (m\theta _{i})+x_{2i}\cos (m\theta _{i})\end{matrix}\right)
$$

### 4. Full Matrix Representation
Aggregating all $d/2$ dimensions together, the full RoPE transformation applied to a Query or Key vector $\mathbf{x}$ at position $m$ is expressed as a block-diagonal matrix multiplication:

$$
\text{RoPE}(\mathbf{x},m)=\mathbf{R}_{\Theta ,m}\mathbf{x}=\left(\begin{matrix}\cos (m\theta _{1})&-\sin (m\theta _{1})&0&0&\dots &0&0\\ \sin (m\theta _{1})&\cos (m\theta _{1})&0&0&\dots &0&0\\ 0&0&\cos (m\theta _{2})&-\sin (m\theta _{2})&\dots &0&0\\ 0&0&\sin (m\theta _{2})&\cos (m\theta _{2})&\dots &0&0\\ \vdots &\vdots &\vdots &\vdots &\ddots &\vdots &\vdots \\ 0&0&0&0&\dots &\cos (m\theta _{d/2})&-\sin (m\theta _{d/2})\\ 0&0&0&0&\dots &\sin (m\theta _{d/2})&\cos (m\theta _{d/2})\end{matrix}\right)\left(\begin{matrix}x_{1}\\ x_{2}\\ x_{3}\\ x_{4}\\ \vdots \\ x_{d-1}\\ x_{d}\end{matrix}\right)
$$

\
\
Now when we have to calculate the inner product of query at position m and key at position n it depends purely on the relative offset position (n - m). The calculation is below.\\

### 1. The Transpose Property of a Rotation Matrix
The standard vector dot product (inner product) between two transformed vectors $\mathbf{Ax}$ and $\mathbf{By}$ can be rewritten by moving the matrix of the first vector to the other side as a transpose: 
$$\langle \mathbf{R}_{m}\mathbf{q},\mathbf{R}_{n}\mathbf{k}\rangle =\mathbf{q}^{T}\mathbf{R}_{m}^{T}\mathbf{R}_{n}\mathbf{k}$$

A 2D rotation matrix $\mathbf{R}_{m}$ that rotates a vector by an angle of $m\theta$ looks like this:
$$\mathbf{R}_{m}=\left(\begin{matrix}\cos (m\theta )&-\sin (m\theta )\\ \sin (m\theta )&\cos (m\theta )\end{matrix}\right)$$

Taking the transpose ($\mathbf{R}_{m}^{T}$) swaps its rows and columns:
$$\mathbf{R}_{m}^{T}=\left(\begin{matrix}\cos (m\theta )&\sin (m\theta )\\ -\sin (m\theta )&\cos (m\theta )\end{matrix}\right)$$

Because $\cos(-x) = \cos(x)$ and $\sin(-x) = -\sin(x)$, this transposed matrix is identical to rotating by a negative angle:
$$\mathbf{R}_{m}^{T}=\mathbf{R}_{-m}$$

**Geometric Meaning:** Transposing a rotation matrix means rotating in the opposite direction (spinning backwards instead of forwards).

### 2. Combining the Rotations
Evaluating the core matrix product in the middle of the formula: $\mathbf{R}_m^T \mathbf{R}_n$. Replacing the transpose with the negative rotation property yields:
$$\mathbf{R}_{m}^{T}\mathbf{R}_{n}=\mathbf{R}_{-m}\mathbf{R}_{n}$$

When multiplying two rotation matrices, their angles add up. Rotating a vector backward by $m\theta$ and then forward by $n\theta$ results in a net rotation of:
$$\text{Net Rotation} = n\theta - m\theta = (n-m)\theta$$

Therefore:
$$\mathbf{R}_{m}^{T}\mathbf{R}_{n}=\mathbf{R}_{n-m}$$

### 3. Expanding the Trigonometry
Multiplying the matrices explicitly demonstrates this relationship:
$$\mathbf{R}_{m}^{T}\mathbf{R}_{n}=\left(\begin{matrix}\cos (m\theta )&\sin (m\theta )\\ -\sin (m\theta )&\cos (m\theta )\end{matrix}\right)\left(\begin{matrix}\cos (n\theta )&-\sin (n\theta )\\ \sin (n\theta )&\cos (n\theta )\end{matrix}\right)$$

Using standard matrix multiplication rules for the top-left element:
$$\text{Top-Left Element}=\cos (m\theta )\cos (n\theta )+\sin (m\theta )\sin (n\theta )$$

Using the trigonometric subtraction identity—$\cos(A-B) = \cos(A)\cos(B) + \sin(A)\sin(B)$—this collapses perfectly into:
$$\cos (n\theta - m\theta) = \cos ((n-m)\theta )$$

Completing the multiplication for all four elements using standard sine/cosine subtraction identities yields the combined matrix:
$$\mathbf{R}_{m}^{T}\mathbf{R}_{n}=\left(\begin{matrix}\cos ((n-m)\theta )&-\sin ((n-m)\theta )\\ \sin ((n-m)\theta )&\cos ((n-m)\theta )\end{matrix}\right)=\mathbf{R}_{n-m}$$

### 4. Final Substitution
Plugging this matrix identity back into the original inner product equation gives:
$$\mathbf{q}^{T}\left(\mathbf{R}_{m}^{T}\mathbf{R}_{n}\right)\mathbf{k}=\mathbf{q}^{T}\mathbf{R}_{n-m}\mathbf{k}$$

Because the matrix $\mathbf{R}_{n-m}$ only contains the variable $(n-m)$, the final attention score calculation no longer depends on the absolute values of $m$ or $n$. It scales strictly based on the distance separating them.

