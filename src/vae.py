import time
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.nn_utils import standard_normal_logprob, kl_divergence, MLP


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.input_dim_flat = np.prod(input_dim) if isinstance(input_dim, Iterable) else input_dim

        # Flatten module for encoder
        self.flatten = nn.Flatten()
        # Unflatten module for decoder output
        self.unflatten = nn.Unflatten(1, (input_dim,) if isinstance(input_dim, int) else tuple(input_dim))

        # Encoder: flatten input -> latent_dim*2
        self.encoder = nn.Sequential(
            self.flatten,
            MLP(self.input_dim_flat, latent_dim * 2, hidden_dim)
        )

        # Decoder: latent -> 2*input_dim_flat (flat tensor)
        self.decoder = MLP(latent_dim, self.input_dim_flat * 2, hidden_dim)

        self.nlogpxz = nn.GaussianNLLLoss(reduction='sum')

    def encode(self, x):
        h = self.encoder(x)
        mu_z, logvar_z = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu_z, logvar_z

    def decode(self, z):
        h = self.decoder(z)
        mu_x, logvar_x = h[:, :self.input_dim_flat], h[:, self.input_dim_flat:]
        return mu_x, logvar_x  # still flat

    def reparameterize(self, mu_z, logvar_z):
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return mu_z + eps * std

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, _ = self.decode(z)
        return self.unflatten(mu_x)

    def sample(self, n):
        device = next(self.parameters()).device
        z = torch.randn((n, self.latent_dim), device=device)
        mu_x, _ = self.decode(z)
        return self.unflatten(mu_x)

    def get_loss(self, x, n_samples_z=1):
        """
        Compute VAE loss with optional multiple latent samples per data point.

        Args:
            x: input tensor [batch, ...]
            n_samples_z: number of latent samples to average over (default 1)
        Returns:
            scalar loss (reconstruction + KL)
        """
        mu_z, logvar_z = self.encode(x)

        # KL divergence (same for all latent samples)
        kl = kl_divergence(mu_z, logvar_z).sum()

        # Flatten input once
        x_flat = x.flatten(start_dim=1)

        if n_samples_z == 1:
            # Single-sample (original behavior)
            z = self.reparameterize(mu_z, logvar_z)
            mu_x, logvar_x = self.decode(z)
            var_x = torch.exp(logvar_x)
            x_flat_repeat = x_flat
            mu_x_flat = mu_x
            var_x_flat = var_x
            recon_loss = self.nlogpxz(mu_x_flat, x_flat_repeat, var_x_flat)
            return recon_loss + kl

        # -------------------------------
        # Multi-sample (vectorized)
        # -------------------------------
        mu_repeat = mu_z.unsqueeze(0).expand(n_samples_z, -1, -1)        # [n_samples_z, batch, latent_dim]
        logvar_repeat = logvar_z.unsqueeze(0).expand(n_samples_z, -1, -1)
        eps = torch.randn_like(mu_repeat)

        # Reparameterize in batch
        z = mu_repeat + eps * torch.exp(0.5 * logvar_repeat)             # [n_samples_z, batch, latent_dim]
        z_flat = z.reshape(-1, z.shape[-1])                               # [n_samples_z*batch, latent_dim]

        # Decode all at once
        mu_x, logvar_x = self.decode(z_flat)
        var_x = torch.exp(logvar_x)

        # Repeat x_flat for all latent samples
        x_repeat = x_flat.unsqueeze(0).expand(n_samples_z, -1, -1).reshape(-1, x_flat.shape[-1])

        # Total reconstruction loss
        recon_loss_total = self.nlogpxz(mu_x, x_repeat, var_x)

        # Average over latent samples
        recon_loss_total /= n_samples_z

        # Total loss
        return recon_loss_total + kl

    def train_model(self, x, batch_size=64, lr=1e-3, epochs=10, max_time=None, n_samples_z=1, device=None, verbose=True):
        """
        Train the VAE on raw tensor x without DataLoader/shuffling.
        Deterministic batching, optionally multiple latent samples per input.
        """

        if device is None:
            device = next(self.parameters()).device

        self.to(device)
        self.train()

        x = x.to(device)
        n_samples = x.shape[0]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses_per_epoch = []
        epoch_times = []

        start_time = time.time()

        # Print progress every 10% of training
        print_interval = max(1, epochs // 10)

        for epoch in range(epochs):

            epoch_start = time.time()
            total_loss = 0.0

            for start in range(0, n_samples, batch_size):

                end = start + batch_size
                xb = x[start:end]

                optimizer.zero_grad()

                loss = self.get_loss(xb, n_samples_z=n_samples_z)

                (loss/len(xb)).backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_time = time.time() - epoch_start

            avg_loss = total_loss / n_samples
            losses_per_epoch.append(avg_loss)
            epoch_times.append(epoch_time)

            if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0):

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"loss={avg_loss:.4f} | "
                    f"time={epoch_time:.2f}s"
                )

            if max_time is not None and (time.time() - start_time) >= max_time:
                print(f"Time budget reached at epoch {epoch+1}")
                break

        total_training_time = time.time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        print(
            f"Training finished: {len(losses_per_epoch)} epochs, "
            f"total time {total_training_time:.2f}s, "
            f"avg epoch {avg_epoch_time:.2f}s"
        )

        return losses_per_epoch, epoch_times