# A15 — Written Viva Prep: Module 3 Numericals
**Format:** 4 questions per student — 2 from Module 3, 2 from Module 4  
**Style:** Numerical / problem-solving  

---

# MODULE 3 — NUMERICAL QUESTIONS (GAN / Autoencoder / RNN topics)

---

## TOPIC 1: GAN Loss and Training Dynamics

### Q1. BCE Loss Calculation
**Question:** A Discriminator receives a batch of 4 images. Its outputs are [0.9, 0.8, 0.3, 0.2] where the first two are real (label=1) and the last two are fake (label=0). Compute the Binary Cross-Entropy loss for the Discriminator.

**Solution:**
$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

For real images (y=1):
- Sample 1: −log(0.9) = 0.1054
- Sample 2: −log(0.8) = 0.2231

For fake images (y=0):
- Sample 3: −log(1−0.3) = −log(0.7) = 0.3567
- Sample 4: −log(1−0.2) = −log(0.8) = 0.2231

$$L_D = \frac{0.1054 + 0.2231 + 0.3567 + 0.2231}{4} = \frac{0.9083}{4} = \boxed{0.2271}$$

---

### Q2. Generator Loss Calculation
**Question:** In the same batch above, what is the Generator's loss? (G wants D to output 1 for fakes; D outputs 0.3 and 0.2 for the two fake images.)

**Solution:**  
G's loss = −mean[log D(G(z))] (G wants D to say fakes are real, so label=1)

- Sample 3 (D output 0.3): −log(0.3) = 1.2040
- Sample 4 (D output 0.2): −log(0.2) = 1.6094

$$L_G = \frac{1.2040 + 1.6094}{2} = \frac{2.8134}{2} = \boxed{1.4067}$$

---

### Q3. Nash Equilibrium Condition
**Question:** At Nash equilibrium in a GAN, what does the Discriminator output for any input? Derive this result.

**Solution:**  
At equilibrium, G has learned the true data distribution: p_g = p_data.  
The optimal D for a fixed G is:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

When p_g = p_data:
$$D^*(x) = \frac{p_{data}(x)}{2 \cdot p_{data}(x)} = \boxed{0.5}$$

D cannot distinguish real from fake — outputs 0.5 for every input.

---

### Q4. Parameter Count
**Question:** A Generator has a linear layer mapping z (dim=100) to a (256×7×7) feature map, followed by a ConvTranspose2d(256→128, k=4, s=2, p=1) with BatchNorm, a ConvTranspose2d(128→64, k=4, s=2, p=1) with BatchNorm, and a Conv2d(64→1, k=3, s=1, p=1). Calculate the total trainable parameters (excluding biases, using bias=False).

**Solution:**

| Layer | Calculation | Params |
|-------|-------------|--------|
| Linear(100 → 256×7×7 = 12544) | 100 × 12544 | 1,254,400 |
| BN(12544) | 2 × 12544 | 25,088 |
| ConvTranspose2d(256→128, k=4) | 256×128×4×4 | 524,288 |
| BN(128) | 2 × 128 | 256 |
| ConvTranspose2d(128→64, k=4) | 128×64×4×4 | 131,072 |
| BN(64) | 2 × 64 | 128 |
| Conv2d(64→1, k=3) | 64×1×3×3 | 576 |
| **Total** | | **≈ 1,935,808** |

---

## TOPIC 2: Backpropagation in RNNs

### Q5. Vanishing Gradient Problem
**Question:** In an RNN unrolled for T=5 time steps with tanh activation (max derivative = 1), and weight matrix W_hh with largest eigenvalue λ = 0.4, estimate the gradient magnitude at step t=1 when ∂L/∂h₅ = 1.

**Solution:**  
Gradient at step t=1 requires multiplying T−1 = 4 Jacobian matrices:

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_5} \cdot \prod_{k=2}^{5} \frac{\partial h_k}{\partial h_{k-1}}$$

Each factor ≤ |λ| × max(tanh') = 0.4 × 1 = 0.4

$$\left|\frac{\partial L}{\partial h_1}\right| \approx 1 \times (0.4)^4 = 0.0256$$

This illustrates **vanishing gradients** — the signal diminishes exponentially with sequence length.

---

### Q6. LSTM Forget Gate
**Question:** An LSTM cell at time t has: h_{t-1} = [0.5, −0.3], x_t = [1.0, 0.2], W_f (2×4 matrix):
```
W_f = [[0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]]
```
b_f = [0.1, 0.1, 0.1, 0.1]. Compute the forget gate f_t = σ([h_{t-1}; x_t]·W_f + b_f) for the first element.

**Solution:**  
Input vector: [h_{t-1}; x_t] = [0.5, −0.3, 1.0, 0.2]

First element of W_f column: [0.1, 0.5, 0.1, 0.5]... 

Actually using the standard notation: f_t = σ(W_f · [h_{t-1}; x_t] + b_f)

First output element:
$$z_1 = (0.1)(0.5) + (0.2)(−0.3) + (0.3)(1.0) + (0.4)(0.2) + 0.1$$
$$= 0.05 − 0.06 + 0.30 + 0.08 + 0.10 = 0.47$$
$$f_{t,1} = \sigma(0.47) = \frac{1}{1+e^{-0.47}} = \frac{1}{1+0.6250} = \boxed{0.6153}$$

This gate value of ~0.615 means the cell retains ~61.5% of its previous state.

---

### Q7. GRU Update Gate
**Question:** A GRU has update gate z_t = σ(0.8) and previous hidden state h_{t-1} = 0.6, candidate hidden state h̃_t = 0.9. Compute the new hidden state h_t.

**Solution:**  
$$z_t = \sigma(0.8) = \frac{1}{1+e^{-0.8}} = 0.6900$$

$$h_t = (1-z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$$
$$= (1 − 0.69)(0.6) + (0.69)(0.9)$$
$$= (0.31)(0.6) + (0.69)(0.9)$$
$$= 0.186 + 0.621 = \boxed{0.807}$$

---

### Q8. Autoencoder Reconstruction Loss
**Question:** An autoencoder takes a 4-dimensional input x = [1, 0, 1, 0] and produces reconstruction x̂ = [0.9, 0.1, 0.8, 0.2]. Compute the Mean Squared Error (MSE) reconstruction loss.

**Solution:**
$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2$$
$$= \frac{(1-0.9)^2 + (0-0.1)^2 + (1-0.8)^2 + (0-0.2)^2}{4}$$
$$= \frac{0.01 + 0.01 + 0.04 + 0.04}{4} = \frac{0.10}{4} = \boxed{0.025}$$

---

### Q9. VAE ELBO / KL Divergence
**Question:** A VAE encodes an input to a latent distribution with μ = [1.0, 0.5] and log σ² = [0.0, −1.0]. Compute the KL divergence loss term: KL(q(z|x) || p(z)) where p(z) = N(0,I).

**Solution:**
$$KL = -\frac{1}{2}\sum_{j=1}^{J}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

For j=1: μ₁=1.0, log σ₁² = 0.0 → σ₁² = e⁰ = 1.0
$$= -\frac{1}{2}(1 + 0 - 1 - 1) = -\frac{1}{2}(-1) = 0.5$$

For j=2: μ₂=0.5, log σ₂² = −1.0 → σ₂² = e⁻¹ = 0.3679
$$= -\frac{1}{2}(1 + (-1) - 0.25 - 0.3679) = -\frac{1}{2}(-0.6179) = 0.3090$$

$$KL_{total} = 0.5 + 0.3090 = \boxed{0.8090}$$

---

### Q10. Convolution Output Size
**Question:** A transposed convolution (ConvTranspose2d) is applied to a 7×7 feature map with kernel_size=4, stride=2, padding=1. What is the output spatial size?

**Solution:**
$$H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}$$
$$= (7-1) \times 2 - 2 \times 1 + 4 = 12 - 2 + 4 = \boxed{14}$$

Output: **14×14** — this is exactly the upsampling step in our Generator.

---

# MODULE 4 — NUMERICAL QUESTIONS (Transformers / Attention)

---

### Q11. Scaled Dot-Product Attention
**Question:** Given Q = [[1, 0], [0, 1]], K = [[1, 1], [0, 1]], V = [[1, 2], [3, 4]], d_k = 2. Compute the self-attention output for query 1.

**Solution:**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For Q₁ = [1, 0]:
$$QK^T: [1,0]\cdot[1,1]^T = 1, \quad [1,0]\cdot[0,1]^T = 0$$
$$\text{scores} = [1, 0] / \sqrt{2} = [0.707, 0]$$
$$\text{softmax}: e^{0.707}=2.028, e^0=1 \Rightarrow [2.028/(2.028+1), 1/(2.028+1)] = [0.670, 0.330]$$
$$\text{Output} = 0.670\times[1,2] + 0.330\times[3,4] = [0.670, 1.340] + [0.990, 1.320] = \boxed{[1.660, 2.660]}$$

---

### Q12. Positional Encoding
**Question:** Compute the positional encoding for position pos=1, dimension i=0, with d_model=4.

**Solution:**
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

For pos=1, i=0, d_model=4:
$$PE(1, 0) = \sin\left(\frac{1}{10000^{0/4}}\right) = \sin\left(\frac{1}{1}\right) = \sin(1) = \boxed{0.8415}$$

---

### Q13. Multi-Head Attention Parameter Count
**Question:** A multi-head attention layer has d_model=512, num_heads=8. What is d_k? How many parameters are in the W_Q projection matrix for one head?

**Solution:**  
$$d_k = d_v = \frac{d_{model}}{h} = \frac{512}{8} = \boxed{64}$$

W_Q for one head: shape (d_model × d_k) = 512 × 64 = **32,768 parameters**  
Total W_Q across all heads: 8 × 32,768 = 262,144 parameters.

---

### Q14. Transformer Feed-Forward Layer
**Question:** A Transformer FFN sublayer takes a d_model=256 vector through a hidden layer of size d_ff=1024, then back to 256. How many parameters does this FFN sublayer have (including biases)?

**Solution:**
- Layer 1: W₁ (256×1024) + b₁ (1024) = 262,144 + 1,024 = 263,168
- Layer 2: W₂ (1024×256) + b₂ (256) = 262,144 + 256 = 262,400

$$\text{Total} = 263,168 + 262,400 = \boxed{525,568}$$

---

## KEY FORMULAS CHEAT SHEET

| Formula | Expression |
|---------|-----------|
| BCE Loss | −[y log ŷ + (1−y)log(1−ŷ)] |
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) |
| Tanh | tanh(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) |
| MSE | (1/n)Σ(xᵢ−x̂ᵢ)² |
| KL(N(μ,σ²) ∥ N(0,1)) | −½Σ(1+log σ²−μ²−σ²) |
| ConvTranspose output | (Hᵢₙ−1)×s − 2p + k |
| Conv output | ⌊(Hᵢₙ + 2p − k)/s⌋ + 1 |
| Attention | softmax(QKᵀ/√dₖ)V |
| Positional Encoding (even) | sin(pos/10000^(2i/d_model)) |
| LSTM forget gate | f_t = σ(W_f[h_{t-1};x_t] + b_f) |
| GRU hidden state | h_t = (1−z_t)h_{t-1} + z_t h̃_t |
| Nash equilibrium D output | D*(x) = 0.5 |

---

## WHAT THE EXAMINER MAY ASK DURING DEMO

1. "Run the training for 1 epoch and show me the output."
2. "Explain why we use Tanh at the Generator output."
3. "What does the Discriminator output mean?"
4. "Why is LeakyReLU used in D but ReLU in G?"
5. "What would happen if both G and D losses go to zero?"
6. "What is mode collapse? Did you observe it?"
7. "Change the latent dimension from 100 to 50 and rerun a cell."
8. "Show me the latent interpolation and explain what it means."
9. "Why do we use strided convolutions instead of pooling in DCGAN?"
10. "What is the minimax objective of a GAN?"

---

## INDIVIDUAL CONTRIBUTION STATEMENTS (template)

**Amrith Saras:**  
Implemented the Generator architecture (linear projection, transposed convolution blocks, weight initialisation). Coded the training loop including the Discriminator and Generator update steps. Implemented latent space interpolation experiment.

**Annabel Marianne Victor:**  
Implemented the Discriminator architecture (strided convolution blocks). Coded the visualisation pipeline — epoch-by-epoch sample grids, training loss curve plots, real vs. fake comparison. Analysed training dynamics and documented observations.

**Diya Jothish:**  
Handled data loading and preprocessing (MNIST download, normalisation, DataLoader setup). Compiled the project report. Prepared the PowerPoint presentation and the individual contribution logs.
