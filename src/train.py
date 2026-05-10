"""
Train the waveform surrogate model.

Trains an MLP to map (m1, m2) -> (log_amplitude, phase) using MSE loss.

Usage:
    python src/train.py

Saves:
    - data/best_model.pt        (model weights + norm stats)
    - figures/training_curve.png (loss vs epoch)

Typical training time: 2-5 minutes on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from time import time

from dataset import WaveformDataset
from model import WaveformSurrogate


def train():
    # ── hyperparameters ─────────────────────────────────────────
    BATCH_SIZE = 256
    LR = 1e-3
    MAX_EPOCHS = 500
    PATIENCE = 50          # early stopping: stop if no improvement for this many epochs
    LR_PATIENCE = 20       # reduce LR if no improvement for this many epochs
    LR_FACTOR = 0.2        # multiply LR by this when reducing

    # ── paths (adjust if your data/ is elsewhere) ───────────────
    data_dir = Path("data")
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # ── load data ───────────────────────────────────────────────
    print("loading data ...")
    train_ds = WaveformDataset(data_dir / "train.h5")
    val_ds = WaveformDataset(data_dir / "val.h5", norm_stats=train_ds.norm_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  training:   {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  validation: {len(val_ds)} samples, {len(val_loader)} batches")

    # ── model, loss, optimiser ──────────────────────────────────
    model = WaveformSurrogate(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.output_dim,
    )
    print(f"  parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
    )

    # ── training loop ───────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    print(f"\ntraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE}) ...\n")
    print(f"{'epoch':>6}  {'train_loss':>11}  {'val_loss':>11}  {'lr':>10}  {'time':>6}")
    print("-" * 55)

    t_start = time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch = time()

        # ── train ───────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss /= len(train_ds)

        # ── validate ────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item() * x_batch.size(0)

        epoch_val_loss /= len(val_ds)

        # ── bookkeeping ────────────────────────────────────────
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        dt = time() - t_epoch

        # print every 10 epochs + first + last
        if epoch <= 5 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            print(f"{epoch:6d}  {epoch_train_loss:11.6f}  {epoch_val_loss:11.6f}  {current_lr:10.1e}  {dt:5.1f}s")

        # ── early stopping ─────────────────────────────────────
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # save best model + normalization stats (need these at inference)
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": train_ds.norm_stats,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, data_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    total_time = time() - t_start
    print(f"\ndone in {total_time:.1f}s")
    print(f"best val loss: {best_val_loss:.6f} at epoch {epoch - epochs_no_improve}")
    print(f"model saved to {data_dir / 'best_model.pt'}")

    # ── plot training curve ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(train_losses, label="train", alpha=0.8)
        ax.semilogy(val_losses, label="validation", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (normalised)")
        ax.set_title("Training curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=150)
        print(f"training curve saved to {out_dir / 'training_curve.png'}")
        plt.close()
    except ImportError:
        print("matplotlib not available -- skipping training curve plot")


if __name__ == "__main__":
    train()
