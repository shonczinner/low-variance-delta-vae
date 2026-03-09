import matplotlib.pyplot as plt


def plot_metric(histories, metric, ylabel, save_path):
    """
    Plot multiple models’ metrics (loss, time, etc.) on the same figure.
    """
    plt.figure()
    for name, hist in histories.items():
        plt.plot(hist[metric], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_comparative_latents(latents, labels, save_path):
    """
    Plot latent spaces of multiple models side by side.
    """
    n_models = len(latents)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, z) in zip(axes, latents.items()):
        sc = ax.scatter(z[:,0], z[:,1], c=labels[:len(z)], cmap="tab10", s=5)
        ax.set_title(name)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    fig.colorbar(sc, ax=axes, fraction=0.05)
    plt.savefig(save_path)
    plt.close()

def plot_elbo_vs_time(histories, save_path):
    """
    Plot ELBO (per-sample loss) vs cumulative training time for each model.
    
    histories: dict with keys=model_name, values={'loss': [...], 'epoch_time': [...]}
    """

    plt.figure()

    for name, hist in histories.items():
        # cumulative time per epoch
        cum_time = [sum(hist['epoch_time'][:i+1]) for i in range(len(hist['epoch_time']))]
        plt.plot(cum_time, hist['loss'], label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("ELBO Loss per sample")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()