import os
import torch

from src.vae import VAE

from utils.csv_utils import save_final_results_csv, save_csv_as_png
from utils.data_utils import load_mnist_tensor
from utils.plot_utils import plot_metric, plot_comparative_latents,plot_elbo_vs_time

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"


# --- train single model and extract latent space ---
def train_and_get_latent(name, model_fn, x, y, config):
    device = config["device"]
    model = model_fn().to(device)

    print(f"\nRunning {name}")

    losses, epoch_times = model.train_model(
        x,
        batch_size=config["batch_size"],
        lr=config["lr"],
        epochs=config["epochs"],
        max_time=config["max_time"],
        n_samples_z=config["n_samples_z"],
        device=device
    )

    # latent space
    model.eval()
    with torch.no_grad():
        x_latent = x[:config["latent_plot_samples"]].to(device)
        mu_z, _ = model.encode(x_latent)
        z = mu_z.cpu()

    history = {"loss": losses, "epoch_time": epoch_times}
    return history, z


# --- orchestrate experiment ---
def run_experiment(models, config):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    x, y = load_mnist_tensor(config["dataset_size"])

    histories = {}
    latents = {}

    # Train each model
    for name, model_fn in models.items():
        history, z = train_and_get_latent(name, model_fn, x, y, config)
        histories[name] = history
        latents[name] = z

    # Comparative metric plots
    plot_metric(histories, "loss", "ELBO Loss per Epoch", os.path.join(PLOTS_DIR, "loss.png"))

    plot_elbo_vs_time(histories, os.path.join(PLOTS_DIR, "elbo_vs_time.png"))

    # Latent space comparison
    plot_comparative_latents(latents, y, os.path.join(PLOTS_DIR, "latents.png"))

    # Save CSV and PNG table
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    save_final_results_csv(histories, csv_path)
    save_csv_as_png(csv_path, os.path.join(RESULTS_DIR, "results_table.png"))

    print("\nExperiment complete. Results saved to results/")


# --- main ---
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "device": device,
        "dataset_size": 10000,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 200,
        "max_time": None,
        "n_samples_z": 1,
        "latent_plot_samples": 2000
    }

    models = {
        "VAE": lambda: VAE(input_dim=(1, 28, 28), latent_dim=2, hidden_dim=128)
        # Add DeltaVAE, LowVarDeltaVAE later
    }

    run_experiment(models, config)