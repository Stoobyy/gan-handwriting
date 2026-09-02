"""
A15 — GAN-Based Synthetic Handwritten Digit Generation
Team: Amrith Saras · Annabel Marianne Victor · Diya Jothish
Course: 102903/CO701B Deep Learning, RSET
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to files
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 100
IMAGE_SIZE   = 28
CHANNELS     = 1
BATCH_SIZE   = 128
NUM_EPOCHS   = 50
LR           = 0.0002
BETA1        = 0.5
BETA2        = 0.999
FEATURES_G   = 64
FEATURES_D   = 64
SAMPLE_EVERY = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('outputs/samples', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, latent_dim, features_g, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, features_g * 4 * 7 * 7, bias=False),
            nn.BatchNorm1d(features_g * 4 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(features_g * 4, features_g * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features_g * 2, features_g,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(inplace=True),
            nn.Conv2d(features_g, channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(x.size(0), -1, 7, 7)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    def __init__(self, channels, features_d):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, features_d,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(features_d * 4 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_sample_grid(G, noise, epoch):
    G.eval()
    with torch.no_grad():
        imgs = G(noise).cpu()
    G.train()
    grid = make_grid(imgs, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title(f'Generated Digits - Epoch {epoch}', fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/samples/epoch_{epoch:03d}.png', dpi=150)
    plt.close()
    print(f'  Saved sample grid -> outputs/samples/epoch_{epoch:03d}.png')


def plot_curves(G_losses, D_losses, D_real_acc, D_fake_acc):
    epochs = range(1, len(G_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, G_losses, label='Generator', color='royalblue', linewidth=2)
    axes[0].plot(epochs, D_losses, label='Discriminator', color='tomato', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Generator vs Discriminator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, D_real_acc, label='D(x) real', color='green', linewidth=2)
    axes[1].plot(epochs, D_fake_acc, label='D(G(z)) fake', color='orange', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Equilibrium 0.5')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Discriminator Output')
    axes[1].set_title('Discriminator Confidence Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('DCGAN Training Dynamics on MNIST', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/samples/training_curves.png', dpi=150)
    plt.close()
    print('Saved -> outputs/samples/training_curves.png')


def plot_real_vs_fake(G, train_loader):
    real_batch, _ = next(iter(train_loader))
    real_grid = make_grid(real_batch[:32], nrow=16, normalize=True)

    G.eval()
    with torch.no_grad():
        fake_batch = G(torch.randn(32, LATENT_DIM, device=DEVICE)).cpu()
    G.train()
    fake_grid = make_grid(fake_batch, nrow=16, normalize=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(real_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[0].set_title('Real MNIST Images', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(fake_grid.permute(1, 2, 0).numpy(), cmap='gray')
    axes[1].set_title('GAN-Generated Images', fontsize=12)
    axes[1].axis('off')
    plt.suptitle('Real vs GAN-Generated MNIST Digits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/samples/real_vs_fake.png', dpi=150)
    plt.close()
    print('Saved -> outputs/samples/real_vs_fake.png')


def plot_interpolation(G):
    G.eval()
    z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
    z2 = torch.randn(1, LATENT_DIM, device=DEVICE)
    alphas = torch.linspace(0, 1, 10)
    imgs = []
    with torch.no_grad():
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            imgs.append(G(z).cpu())
    grid = make_grid(torch.cat(imgs), nrow=10, normalize=True)
    plt.figure(figsize=(15, 2))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Latent Space Interpolation (z1 -> z2)', fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/samples/latent_interpolation.png', dpi=150)
    plt.close()
    print('Saved -> outputs/samples/latent_interpolation.png')
    G.train()


def main():
    print(f'Device  : {DEVICE}')
    print(f'Epochs  : {NUM_EPOCHS}')
    print(f'Batch   : {BATCH_SIZE}')
    print(f'Latent  : {LATENT_DIM}')
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    print(f'MNIST loaded: {len(train_dataset):,} samples, {len(train_loader)} batches/epoch\n')

    # Save a grid of real images
    real_batch, _ = next(iter(train_loader))
    grid = make_grid(real_batch[:64], nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Real MNIST Samples', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/samples/real_samples.png', dpi=150)
    plt.close()
    print('Saved -> outputs/samples/real_samples.png')

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator(LATENT_DIM, FEATURES_G, CHANNELS).to(DEVICE)
    D = Discriminator(CHANNELS, FEATURES_D).to(DEVICE)
    G.apply(init_weights)
    D.apply(init_weights)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f'Generator params    : {total_g:,}')
    print(f'Discriminator params: {total_d:,}\n')

    # ── Loss & Optimisers ─────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

    # ── Training ──────────────────────────────────────────────────────────────
    G_losses, D_losses, D_real_acc, D_fake_acc = [], [], [], []

    print(f'Training for {NUM_EPOCHS} epochs...\n')
    for epoch in range(1, NUM_EPOCHS + 1):
        eg, ed, er, ef = 0.0, 0.0, 0.0, 0.0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            bsz = real_imgs.size(0)

            # Train Discriminator
            D.zero_grad()
            real_labels = torch.ones(bsz, 1, device=DEVICE)
            fake_labels = torch.zeros(bsz, 1, device=DEVICE)

            out_real = D(real_imgs)
            loss_d_real = criterion(out_real, real_labels)

            z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z)
            out_fake = D(fake_imgs.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            opt_D.step()

            # Train Generator
            G.zero_grad()
            out_fake_g = D(fake_imgs)
            loss_G = criterion(out_fake_g, real_labels)
            loss_G.backward()
            opt_G.step()

            eg += loss_G.item()
            ed += loss_D.item()
            er += out_real.mean().item()
            ef += out_fake.mean().item()

        n = len(train_loader)
        G_losses.append(eg / n)
        D_losses.append(ed / n)
        D_real_acc.append(er / n)
        D_fake_acc.append(ef / n)

        print(f'Epoch [{epoch:>3}/{NUM_EPOCHS}]  '
              f'Loss_D: {D_losses[-1]:.4f}  Loss_G: {G_losses[-1]:.4f}  '
              f'D(x): {D_real_acc[-1]:.3f}  D(G(z)): {D_fake_acc[-1]:.3f}')

        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            save_sample_grid(G, fixed_noise, epoch)

    # ── Post-training plots ───────────────────────────────────────────────────
    print('\nGenerating final outputs...')
    plot_curves(G_losses, D_losses, D_real_acc, D_fake_acc)
    plot_real_vs_fake(G, train_loader)
    plot_interpolation(G)

    # Final 100-sample grid
    G.eval()
    with torch.no_grad():
        final_imgs = G(torch.randn(100, LATENT_DIM, device=DEVICE)).cpu()
    grid = make_grid(final_imgs, nrow=10, normalize=True, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('100 GAN-Generated Handwritten Digits (Final)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/samples/final_generated_100.png', dpi=150)
    plt.close()
    print('Saved -> outputs/samples/final_generated_100.png')

    # Save models
    torch.save(G.state_dict(), 'outputs/models/generator.pth')
    torch.save(D.state_dict(), 'outputs/models/discriminator.pth')
    print('Models saved -> outputs/models/')

    print(f'\nDone.')
    print(f'Final Loss_G : {G_losses[-1]:.4f}')
    print(f'Final Loss_D : {D_losses[-1]:.4f}')
    print(f'Final D(x)   : {D_real_acc[-1]:.4f}')
    print(f'Final D(G(z)): {D_fake_acc[-1]:.4f}')
    print('\nAll outputs saved to outputs/')


if __name__ == '__main__':
    main()
