"""
Train Psl-CNN and baseline AE on I-channel only (1 x 1024) for channel ablation.

Saves checkpoints compatible with evaluate_channel_ablation.py:
  spectrum_data/psl_cnn_{epochs}epochs_ch1.pth
  spectrum_data/baseline_{epochs}epochs_ch1.pth

Usage:
  python spectrum_data/train_models_1channel.py --epochs 200
  python spectrum_data/train_models_1channel.py --epochs 50   # faster smoke test
"""

from __future__ import annotations

import os

# macOS / conda: multiple OpenMP runtimes (PyTorch + NumPy) can abort without this — set BEFORE torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d


def train_ae(model, optimizer, scheduler, model_name: str, epochs: int, loader: DataLoader) -> list[float]:
    device = next(model.parameters()).device
    model.train()
    losses: list[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            x = batch.to(device)
            recon = model.AE(x)
            if recon.shape[-1] != x.shape[-1]:
                recon = recon[..., : x.shape[-1]]
            loss = nn.MSELoss()(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        losses.append(avg)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  {model_name} epoch {epoch + 1:4d}/{epochs} | MSE {avg:.6f}")
    print(f"  {model_name} done in {time.time() - t0:.1f}s")
    return losses


def main() -> None:
    p = argparse.ArgumentParser(description="Train 1-channel Psl-CNN + baseline (I only).")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs (paper uses 200).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_full = torch.load("spectrum_data/train_noise.pt", weights_only=False)
    if train_full.dim() != 3 or train_full.shape[1] < 1:
        raise ValueError(f"Expected train_noise (N,C,L), got {tuple(train_full.shape)}")
    train_i = train_full[:, 0:1, :].contiguous()
    print(f"Train I-channel only: {tuple(train_i.shape)} (from full {tuple(train_full.shape)})")

    loader = DataLoader(TensorDataset(train_i), batch_size=args.batch_size, shuffle=True)

    # Match inverted PsiNN eval: k=5, nf=16, dropout
    model_psi = AE_Classifier1d(n_channels=1, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
    model_base = AE_Baseline_Classifier1d(n_channels=1, n_classes=1, nf=16, k=5, use_dropout=True).to(device)

    opt_psi = torch.optim.Adam(model_psi.parameters(), lr=args.lr)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=args.lr)
    sch_psi = torch.optim.lr_scheduler.StepLR(opt_psi, step_size=50, gamma=0.5)
    sch_base = torch.optim.lr_scheduler.StepLR(opt_base, step_size=50, gamma=0.5)

    print(f"\n=== Psl-CNN (1 ch) ===")
    train_ae(model_psi, opt_psi, sch_psi, "Psl-CNN ch1", args.epochs, loader)
    print(f"\n=== Baseline AE (1 ch) ===")
    train_ae(model_base, opt_base, sch_base, "Baseline ch1", args.epochs, loader)

    out_psi = f"spectrum_data/psl_cnn_{args.epochs}epochs_ch1.pth"
    out_base = f"spectrum_data/baseline_{args.epochs}epochs_ch1.pth"
    torch.save(model_psi.state_dict(), out_psi)
    torch.save(model_base.state_dict(), out_base)
    print(f"\nSaved:\n  {out_psi}\n  {out_base}")
    print("Run: python spectrum_data/evaluate_channel_ablation.py")


if __name__ == "__main__":
    main()
