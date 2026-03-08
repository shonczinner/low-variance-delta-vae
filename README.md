# low-variance-delta-vae

A comparison of vanilla VAEs, deterministic delta-method VAEs, and a low-variance VAE using Hutchinson’s trace estimator.

## Motivation

Variational autoencoders (VAEs) rely on stochastic gradient estimators that can introduce high variance during training. This repository explores two alternatives:

- **Deterministic Delta VAE:** Uses the delta method to approximate expectations, removing sampling variance.
- **Low-Variance Delta VAE:** Combines the delta method with Hutchinson’s trace estimator to reduce gradient variance while remaining efficient.

The goal is to study the trade-offs between bias, variance, and training stability.

## Methods

Three models will be implemented:

1. **Vanilla VAE** – Standard VAE with the reparameterization trick.
2. **Deterministic Delta VAE** – Approximates expectations deterministically using the delta method.
3. **Low-Variance Delta VAE** – Uses the delta method plus Hutchinson’s trace estimator for lower-variance gradient estimates.

## Results

*(Placeholders for now; will be generated once experiments are run)*

**Benchmark Table (example placeholder)**  

![Benchmark Results](results/results.jpg)

**Plots (example placeholders)**

- ELBO vs Epochs: `results/plots/elbo_curve.png`
- Gradient Variance vs Epochs: `results/plots/gradient_variance.png`
- Latent Space Projections: `results/plots/latent_space.png`

## Repository Structure
