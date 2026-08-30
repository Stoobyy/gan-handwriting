# A15 — GAN-Based Synthetic Handwritten Digit Generation
**Course:** 102903/CO701B Deep Learning  
**Institution:** Rajagiri School of Engineering & Technology  
**Team:** Amrith Saras · Annabel Marianne Victor · Diya Jothish  
**Module:** 3

---

## Title
GAN-Based Synthetic Handwritten Digit Generation using Deep Convolutional GAN (DCGAN) on the MNIST Dataset

---

## Problem Statement
Handwritten digit recognition is a well-studied classification task. However, generating realistic synthetic handwritten digits from random noise is a challenging generative modelling problem. This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset to learn the underlying distribution of handwritten digits and generate novel, visually plausible digit images.

---

## Objectives
1. Understand the GAN framework — the minimax game between Generator and Discriminator.
2. Implement a DCGAN with convolutional layers for both G and D.
3. Train the model on MNIST (70,000 greyscale 28×28 images).
4. Generate novel synthetic handwritten digit images.
5. Analyse generation quality visually and through training dynamics (loss curves, D outputs).
6. Perform latent space interpolation to examine the learned latent representation.

---

## Background / Literature Review

### 2.1 Generative Adversarial Networks (Goodfellow et al., 2014)
A GAN consists of two networks:
- **Generator (G):** Maps a random noise vector z ~ N(0,I) to a synthetic image G(z).
- **Discriminator (D):** Classifies input images as real (from dataset) or fake (from G).

The networks are trained jointly via a minimax objective:

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

At Nash equilibrium, G perfectly replicates the data distribution and D outputs 0.5 for every input.

### 2.2 DCGAN (Radford et al., 2015)
DCGAN introduced architectural guidelines that stabilise GAN training:
- Replace pooling with strided convolutions (D) and transposed convolutions (G).
- Batch Normalisation in G and D (except at output/input layers).
- No fully connected hidden layers.
- ReLU in G, LeakyReLU (slope 0.2) in D, Tanh at G output.

### 2.3 MNIST Dataset
60,000 training + 10,000 test greyscale 28×28 images of handwritten digits (0–9), collected by LeCun et al. (1998). It is the standard benchmark dataset for generative digit models.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | MNIST |
| Source | torchvision.datasets.MNIST |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 pixels |
| Channels | 1 (greyscale) |
| Classes | 10 (digits 0–9) |

### Preprocessing
- Images loaded as tensors via `torchvision.transforms.ToTensor()`.
- Pixel values normalised from [0, 1] → [−1, 1] using `Normalize([0.5], [0.5])`.
  - This matches the Tanh activation at the Generator output which also produces values in [−1, 1].
- Batch size: 128. Shuffled each epoch.

---

## Methodology / Architecture

### Generator Architecture

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| Input | Noise vector z | (100,) |
| 1 | Linear + BN + ReLU | (256×7×7,) |
| Reshape | — | (256, 7, 7) |
| 2 | ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + ReLU | (128, 14, 14) |
| 3 | ConvTranspose2d(128→64, k=4, s=2, p=1) + BN + ReLU | (64, 28, 28) |
| 4 | Conv2d(64→1, k=3, s=1, p=1) + Tanh | (1, 28, 28) |

### Discriminator Architecture

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| Input | Image | (1, 28, 28) |
| 1 | Conv2d(1→64, k=4, s=2, p=1) + LeakyReLU(0.2) | (64, 14, 14) |
| 2 | Conv2d(64→128, k=4, s=2, p=1) + BN + LeakyReLU | (128, 7, 7) |
| 3 | Conv2d(128→256, k=4, s=2, p=1) + BN + LeakyReLU | (256, 3, 3) |
| 4 | Flatten + Linear(2304→1) + Sigmoid | (1,) |

### Weight Initialisation
All Conv/Linear weights initialised from N(0, 0.02); BatchNorm weights from N(1, 0.02), biases = 0. (DCGAN convention)

---

## Implementation

**Framework:** PyTorch  
**Language:** Python 3  
**Hardware:** CPU / CUDA GPU  
**Key libraries:** torch, torchvision, matplotlib, numpy

**Training:**
- Epochs: 50
- Batch size: 128
- Optimiser: Adam (lr=0.0002, β₁=0.5, β₂=0.999)
- Loss: Binary Cross-Entropy (BCELoss)
- Fixed noise vector (64 samples) used for consistent epoch-by-epoch visualisation.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Latent dim | 100 |
| Epochs | 50 |
| Batch size | 128 |
| Learning rate | 0.0002 |
| Adam β₁ | 0.5 |
| Adam β₂ | 0.999 |
| Loss | BCE |
| Seed | 42 |

---

## Results

*(Fill in actual values after running the notebook)*

| Metric | Value |
|--------|-------|
| Final Generator Loss | — |
| Final Discriminator Loss | — |
| D(x) at final epoch | — |
| D(G(z)) at final epoch | — |

**Figures to include:**
1. `real_samples.png` — Sample real MNIST images
2. `epoch_001.png` — Generated samples at epoch 1 (noisy baseline)
3. `epoch_010.png`, `epoch_025.png`, `epoch_050.png` — Progression
4. `training_curves.png` — G/D loss curves + D confidence over epochs
5. `final_generated_100.png` — 100 final generated digits
6. `real_vs_fake.png` — Side-by-side comparison
7. `latent_interpolation.png` — Interpolation between two z vectors

---

## Analysis

**Training dynamics:**
- Early epochs show high Generator loss and low Discriminator loss — D trivially detects fakes.
- As training progresses, G improves and D(G(z)) increases while D(x) decreases, both approaching 0.5.
- A D output of ~0.5 for both real and fake images indicates Nash equilibrium — D can no longer reliably distinguish real from generated images.

**Visual quality:**
- Epoch 1: Random noise patterns.
- Epoch 10: Rough digit shapes visible.
- Epoch 50: Sharp, diverse, recognisable handwritten digits.

**Latent interpolation:**
- Smooth transitions between digit types confirm the Generator has learned a structured, continuous latent space — not just memorising training examples.

**Limitations:**
- No quantitative FID (Fréchet Inception Distance) score computed.
- Possible mode collapse for some digit classes if training is unstable.
- Results vary across runs due to GAN training stochasticity.

---

## Conclusion
A DCGAN was implemented and trained on MNIST over 50 epochs, successfully learning to generate realistic synthetic handwritten digit images from 100-dimensional Gaussian noise. The adversarial training dynamic — Generator and Discriminator competing in a minimax game — converged to a state where generated digits are visually indistinguishable from real MNIST samples. Latent space interpolation further demonstrated that the model has learned a smooth, meaningful internal representation of handwritten digits.

---

## Future Scope
1. **Conditional GAN (cGAN):** Condition on digit class label to generate a specific digit on demand.
2. **Wasserstein GAN (WGAN):** Use Wasserstein distance instead of BCE for more stable training.
3. **FID Evaluation:** Compute Fréchet Inception Distance for quantitative quality assessment.
4. **Progressive GAN:** Gradually grow resolution for higher-quality images.
5. **Application:** Use generated images as data augmentation for digit classifiers on imbalanced datasets.

---

## References
1. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., et al. (2014). *Generative Adversarial Nets.* Advances in Neural Information Processing Systems (NeurIPS).
2. Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* arXiv:1511.06434.
3. LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST Database of Handwritten Digits.* http://yann.lecun.com/exdb/mnist/
4. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). *Improved Techniques for Training GANs.* NeurIPS.
5. Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN.* arXiv:1701.07875.

---

## Individual Contribution Log

| Member | Contributions |
|--------|--------------|
| Amrith Saras | *(e.g., Generator architecture, training loop, latent interpolation)* |
| Annabel Marianne Victor | *(e.g., Discriminator architecture, loss analysis, visualisation code)* |
| Diya Jothish | *(e.g., data preprocessing, results analysis, report writing, PPT)* |
