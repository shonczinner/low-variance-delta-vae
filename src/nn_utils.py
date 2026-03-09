import numpy as np
import torch.nn as nn

LOG_2PI = np.log(2 * np.pi)

def standard_normal_logprob(x):
    """
    x: [batch, dim]
    Returns: [batch] log-probability under N(0,I)
    """
    log_z = -0.5 * x.shape[1] * LOG_2PI
    return log_z - 0.5 * (x ** 2).sum(dim=1)

# kl between N(mu,sigma) and N(0,1)
def kl_divergence(mu, logvar):
    return 0.5 * (
        logvar.exp()
        + mu.pow(2)
        - 1
        - logvar
    ).sum(dim=1)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)





