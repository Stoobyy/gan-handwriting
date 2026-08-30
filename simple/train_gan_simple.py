"""
A15 - GAN-Based Synthetic Handwritten Digit Generation
Team: Amrith Saras | Annabel Marianne Victor | Diya Jothish
Course: 102903/CO701B Deep Learning, RSET

Simple GAN using fully-connected (Linear) layers only.
Easy to read, easy to explain, still produces real results.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Settings
# ─────────────────────────────────────────────────────────────────────────────

LATENT_DIM  = 64     # size of the random noise vector fed into Generator
IMAGE_SIZE  = 28     # MNIST images are 28x28
FLAT_SIZE   = IMAGE_SIZE * IMAGE_SIZE   # 784 pixels when flattened
BATCH_SIZE  = 128
NUM_EPOCHS  = 50
LR          = 0.0002

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('outputs/samples', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

print(f'Using: {DEVICE}')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load MNIST dataset
# ─────────────────────────────────────────────────────────────────────────────
# Normalize pixels from [0,1] to [-1,1] so they match Generator's Tanh output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = torchvision.datasets.MNIST(
    root='../data', train=True, download=True, transform=transform
)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)

print(f'Dataset: {len(dataset)} images, {len(loader)} batches per epoch')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define the Generator
#
# Job: take a small random noise vector (64 numbers) and produce a
#      fake 28x28 image (784 numbers).
#
# It learns to make images that look real enough to fool the Discriminator.
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: 64 -> 256
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2),          # activation: lets some negative values through

            # Layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            # Layer 3: 512 -> 1024
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            # Output layer: 1024 -> 784 pixels
            nn.Linear(1024, FLAT_SIZE),
            nn.Tanh()                   # squashes output to [-1,1], same range as our images
        )

    def forward(self, z):
        # z is the noise vector, shape: (batch_size, 64)
        img = self.model(z)             # shape: (batch_size, 784)
        img = img.view(-1, 1, 28, 28)  # reshape to image: (batch_size, 1, 28, 28)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Define the Discriminator
#
# Job: look at an image (real or fake) and output a single number between
#      0 and 1 - how likely the image is REAL.
#      0 = definitely fake, 1 = definitely real.
#
# It learns to tell apart real MNIST images from Generator's fakes.
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: flatten the 28x28 image to 784 numbers
            nn.Linear(FLAT_SIZE, 512),
            nn.LeakyReLU(0.2),          # LeakyReLU works better than ReLU in Discriminator

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            # Output: single number -> probability of being real
            nn.Linear(128, 1),
            nn.Sigmoid()                # squashes to (0,1) -> probability
        )

    def forward(self, img):
        # img shape: (batch_size, 1, 28, 28)
        flat = img.view(-1, FLAT_SIZE)  # flatten to (batch_size, 784)
        return self.model(flat)         # shape: (batch_size, 1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Create models, loss function, and optimisers
# ─────────────────────────────────────────────────────────────────────────────

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# Loss: Binary Cross-Entropy
# BCE = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
# Used because we have binary labels: real (1) or fake (0)
loss_fn = nn.BCELoss()

# Separate optimisers for G and D - they train independently
opt_G = optim.Adam(G.parameters(), lr=LR)
opt_D = optim.Adam(D.parameters(), lr=LR)

# Fixed noise: same 64 noise vectors used every epoch for visual comparison
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

print(f'Generator params:     {sum(p.numel() for p in G.parameters()):,}')
print(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Training loop
#
# Each batch does two things:
#   A) Train Discriminator: get better at spotting real vs fake
#   B) Train Generator:     get better at fooling the Discriminator
# ─────────────────────────────────────────────────────────────────────────────

G_losses = []
D_losses = []

print(f'\nTraining for {NUM_EPOCHS} epochs...\n')

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for real_imgs, _ in loader:       # we don't need the labels, only images
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=DEVICE)   # 1 = real
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)  # 0 = fake

        # ── A) Train the Discriminator ────────────────────────────────────────
        # Goal: correctly classify real images as real AND fake images as fake

        D.zero_grad()  # clear old gradients

        # Score real images - D should output close to 1
        score_real = D(real_imgs)
        loss_real = loss_fn(score_real, real_labels)

        # Generate fake images - D should output close to 0
        noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(noise)
        score_fake = D(fake_imgs.detach())   # .detach() stops gradients flowing into G
        loss_fake = loss_fn(score_fake, fake_labels)

        # Total D loss = how wrong it was on both real and fake
        loss_D = loss_real + loss_fake
        loss_D.backward()   # compute gradients
        opt_D.step()        # update D's weights

        # ── B) Train the Generator ────────────────────────────────────────────
        # Goal: make D think the fake images are real (label = 1)

        G.zero_grad()

        # Run fake images through D again (without .detach() - we want G's gradients)
        score_fake_for_G = D(fake_imgs)

        # G wants D to say these are real -> use real_labels
        loss_G = loss_fn(score_fake_for_G, real_labels)
        loss_G.backward()
        opt_G.step()

        epoch_g_loss += loss_G.item()
        epoch_d_loss += loss_D.item()

    # Average loss for this epoch
    avg_g = epoch_g_loss / len(loader)
    avg_d = epoch_d_loss / len(loader)
    G_losses.append(avg_g)
    D_losses.append(avg_d)

    print(f'Epoch [{epoch:>3}/{NUM_EPOCHS}]  Loss_G: {avg_g:.4f}  Loss_D: {avg_d:.4f}')

    # Save a grid of generated images every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        G.eval()
        with torch.no_grad():
            sample_imgs = G(fixed_noise).cpu()
        G.train()

        grid = make_grid(sample_imgs, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.title(f'Generated Digits - Epoch {epoch}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'outputs/samples/epoch_{epoch:03d}.png', dpi=120)
        plt.close()
        print(f'  -> Saved sample grid for epoch {epoch}')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, G_losses, label='Generator Loss',     color='royalblue', linewidth=2)
plt.plot(epochs, D_losses, label='Discriminator Loss', color='tomato',    linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/samples/training_curves.png', dpi=120)
plt.close()
print('\nSaved training curves')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Final outputs
# ─────────────────────────────────────────────────────────────────────────────

# Generate 100 new digits with random noise
G.eval()
with torch.no_grad():
    random_noise = torch.randn(100, LATENT_DIM, device=DEVICE)
    final_imgs = G(random_noise).cpu()

grid = make_grid(final_imgs, nrow=10, normalize=True, padding=2)
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
plt.title('100 GAN-Generated Handwritten Digits (Final Model)')
plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/samples/final_100.png', dpi=120)
plt.close()
print('Saved final 100 generated digits')

# Real vs Fake comparison
real_batch, _ = next(iter(loader))
real_grid = make_grid(real_batch[:32], nrow=16, normalize=True)

with torch.no_grad():
    fake_batch = G(torch.randn(32, LATENT_DIM, device=DEVICE)).cpu()
fake_grid = make_grid(fake_batch, nrow=16, normalize=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 5))
axes[0].imshow(real_grid.permute(1, 2, 0).numpy(), cmap='gray')
axes[0].set_title('Real MNIST Images')
axes[0].axis('off')
axes[1].imshow(fake_grid.permute(1, 2, 0).numpy(), cmap='gray')
axes[1].set_title('GAN-Generated Images')
axes[1].axis('off')
plt.suptitle('Real vs GAN-Generated MNIST Digits', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/samples/real_vs_fake.png', dpi=120)
plt.close()
print('Saved real vs fake comparison')

# Latent space interpolation: walk from noise vector z1 to z2
z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
z2 = torch.randn(1, LATENT_DIM, device=DEVICE)
steps = 10
interp_imgs = []
with torch.no_grad():
    for i in range(steps):
        alpha = i / (steps - 1)              # goes from 0.0 to 1.0
        z = (1 - alpha) * z1 + alpha * z2   # linear blend between z1 and z2
        interp_imgs.append(G(z).cpu())

grid = make_grid(torch.cat(interp_imgs), nrow=steps, normalize=True)
plt.figure(figsize=(14, 2))
plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
plt.title('Latent Space Interpolation (z1 -> z2)')
plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/samples/latent_interpolation.png', dpi=120)
plt.close()
print('Saved latent interpolation')

# Save trained models
torch.save(G.state_dict(), 'outputs/models/generator.pth')
torch.save(D.state_dict(), 'outputs/models/discriminator.pth')
print('Saved models')

print(f'\nDone.')
print(f'Final Loss_G: {G_losses[-1]:.4f}')
print(f'Final Loss_D: {D_losses[-1]:.4f}')
print('All outputs in simple/outputs/')
