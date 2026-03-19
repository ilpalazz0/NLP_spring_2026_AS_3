import os
import matplotlib.pyplot as plt
import csv

# Folder containing loss files
folder_path = "./"

for file_name in os.listdir(folder_path):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(folder_path, file_name)
    base_name = file_name.replace(".txt", "")

    epochs, train_losses, train_accs, val_losses, val_accs = [], [], [], [], []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            train_accs.append(float(row["train_acc"]))
            val_losses.append(float(row["val_loss"]))
            val_accs.append(float(row["val_acc"]))

    if not epochs:
        print(f"Skipping {file_name}: no data found.")
        continue

    # ── Loss curve ──────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', linestyle='-',  color='steelblue', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='s', linestyle='--', color='tomato',    label='Val Loss')
    plt.title(f"Loss Curve: {base_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(folder_path, f"{base_name}_loss_curve.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot  → {loss_plot_path}")

    # ── Accuracy curve ───────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker='o', linestyle='-',  color='steelblue', label='Train Acc')
    plt.plot(epochs, val_accs,   marker='s', linestyle='--', color='tomato',    label='Val Acc')
    plt.title(f"Accuracy Curve: {base_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = os.path.join(folder_path, f"{base_name}_acc_curve.png")
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()
    print(f"Saved acc plot   → {acc_plot_path}")