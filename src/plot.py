import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# Load the uploaded CSV file to inspect its contents
name = "batch16_imgsz128_7000_l_full_aug"
file_path = Path(f"runs/classify/MD_covers_multiclass/{name}/results.csv")
data = pd.read_csv(file_path)

# Extract data for plotting
epochs = data["epoch"]
train_loss = data["train/loss"]
val_loss = data["val/loss"]
top1_acc = data["metrics/accuracy_top1"]
top5_acc = data["metrics/accuracy_top5"]
# Updated code to place both plots together in a single figure

# Creating a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

# Subplot 1: Train vs Validation Loss
axes[0].plot(epochs, train_loss, label="Train Loss", marker="o")
axes[0].plot(epochs, val_loss, label="Validation Loss", marker="o", linestyle='--')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Train vs Validation Loss")
axes[0].legend()
axes[0].grid(True)

# Subplot 2: Top-1 vs Top-5 Accuracy
axes[1].plot(epochs, top1_acc, label="Top-1 Accuracy", marker="o")
axes[1].plot(epochs, top5_acc, label="Top-5 Accuracy", marker="o", linestyle='--')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Top-1 vs Top-5 Accuracy")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
fig.tight_layout()
fig_path = f"{file_path.parent}/loss_and_accuracy.png"
plt.savefig(fig_path)

# Display the combined plot
plt.show()

