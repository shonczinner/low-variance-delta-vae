import torch
from torch.func import jvp, vmap
from src.vae import VAE
from nn_utils import kl_divergence

class DeltaVAE(VAE):
    """
    VAE with first-order Delta / Gauss-Newton correction.
    GN correction is scaled using the decoder's predicted variance.

    Will reimplement the VAE get_loss function.

    gn_mode:
        "exact"       -> exact column norms (vmap over latent basis)
        "hutchinson"  -> stochastic Hutchinson trace estimator
    """

    def __init__(self, *args, gn_mode="hutchinson", **kwargs):
        super().__init__(*args, **kwargs)
        self.gn_mode = gn_mode

    def get_loss(self, x, n_samples_z=1):
        #n_samples_z used for n_hutchison
        n_hutchison = n_samples_z
        return