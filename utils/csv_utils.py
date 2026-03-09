import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


def save_final_results_csv(histories, path, decimals=4):
    """
    Save final metrics per model to CSV (one row per model),
    with rounding for readability.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []

    for name, hist in histories.items():
        final_loss = round(hist["loss"][-1], decimals)
        total_time = round(sum(hist["epoch_time"]), decimals)
        avg_epoch_time = round(total_time / len(hist["epoch_time"]), decimals)

        rows.append({
            "model": name,
            "final_loss": final_loss,
            "total_time": total_time,
            "avg_epoch_time": avg_epoch_time
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "final_loss", "total_time", "avg_epoch_time"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_csv_as_png(csv_path, png_path, fontsize=12, decimals=4):
    """
    Save a CSV file as a PNG table.
    Assumes one row per model.
    Rounds numeric values for readability.
    """

    df = pd.read_csv(csv_path)

    # Round numeric columns
    for col in df.columns:
        if df[col].dtype in [float, int]:
            df[col] = df[col].round(decimals)

    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.8 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)  # row height multiplier

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()