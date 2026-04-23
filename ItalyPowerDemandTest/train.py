import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

# ── Data ──────────────────────────────────────────────────────────────────────
class UCR_Dataset(Dataset):
    def __init__(self, path):
        X, y = [], []
        with open(path) as f:
            in_data = False
            for line in f:
                line = line.strip()
                if line.lower() == "@data":
                    in_data = True
                    continue
                if not in_data or not line or line.startswith("#"):
                    continue
                *values, label = line.split(":")
                X.append([float(v) for v in values[0].split(",")])
                y.append(int(label))
        self.X = torch.FloatTensor(X)[:, None, :]  # [N, 1, L]
        self.y = torch.LongTensor(y) - 1           # labels start at 1

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(UCR_Dataset("/Users/lux/Downloads/ItalyPowerDemand/ItalyPowerDemand_TRAIN.ts"), batch_size=32, shuffle=True)
test_loader  = DataLoader(UCR_Dataset("/Users/lux/Downloads/ItalyPowerDemand/ItalyPowerDemand_TEST.ts"),  batch_size=32)

# ── Train & eval one model ────────────────────────────────────────────────────
def run(model, epochs=20):
    optimizer = Adam(model.parameters(), lr=1e-3)
    accs, mses = [], []
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            recon  = model.AE(X)[:, :, :X.shape[2]]
            loss   = F.cross_entropy(logits, y) + 0.1 * F.mse_loss(recon, X)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        recon_err = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X).argmax(dim=-1)
                correct    += (preds == y).sum().item()
                total      += len(y)
                recon_err  += F.mse_loss(model.AE(X)[:, :, :X.shape[2]], X).item()
        acc = correct / total
        mse = recon_err / len(test_loader)
        accs.append(acc)
        mses.append(mse)
        print(f"  Epoch {epoch+1:3d} | acc: {acc:.3f} | recon MSE: {mse:.4f}")
    return accs, mses

# ── Run all epoch configs ──────────────────────────────────────────────────────
n_channels, n_classes = 1, 2
epoch_configs = [50, 100, 200, 300, 500, 1000]
summary = []  # collect final results for comparison table

for n_epochs in epoch_configs:
    print(f"\n{'='*50}")
    print(f"  EPOCHS: {n_epochs}")
    print(f"{'='*50}")

    print("\n=== PsiNN (shared weights) ===")
    psinn          = AE_Classifier1d(n_channels=n_channels, n_classes=n_classes, nf=16, k=3)
    p_accs, p_mses = run(psinn, epochs=n_epochs)

    print("\n=== Baseline (independent weights) ===")
    baseline       = AE_Baseline_Classifier1d(n_channels=n_channels, n_classes=n_classes, nf=16, k=3)
    b_accs, b_mses = run(baseline, epochs=n_epochs)

    # ── Per-run summary ───────────────────────────────────────────────────────
    print("\n─────────────────────────────────────────────")
    print(f"{'Model':<30} {'Acc':>6}  {'Recon MSE':>10}")
    print(f"{'─'*30} {'─'*6}  {'─'*10}")
    print(f"{'PsiNN (shared weights)':<30} {p_accs[-1]:>6.3f}  {p_mses[-1]:>10.4f}")
    print(f"{'Baseline (indep. weights)':<30} {b_accs[-1]:>6.3f}  {b_mses[-1]:>10.4f}")
    print("─────────────────────────────────────────────")
    summary.append((n_epochs, p_accs[-1], p_mses[-1], b_accs[-1], b_mses[-1]))

    # ── Plots ─────────────────────────────────────────────────────────────────
    ep = range(1, n_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ItalyPowerDemand — {n_epochs} Epochs", fontsize=13)

    ax1.plot(ep, p_accs, label="PsiNN (shared weights)")
    ax1.plot(ep, b_accs, label="Baseline (indep. weights)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Classification Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(ep, p_mses, label="PsiNN (shared weights)")
    ax2.plot(ep, b_mses, label="Baseline (indep. weights)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reconstruction MSE")
    ax2.set_title("Reconstruction MSE")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fname = f"results_{n_epochs}epochs.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plots saved to {fname}")

# ── Final comparison table ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL RESULTS ACROSS ALL EPOCH CONFIGS")
print(f"{'='*65}")
print(f"{'Epochs':>8}  {'PsiNN Acc':>10}  {'PsiNN MSE':>10}  {'Base Acc':>10}  {'Base MSE':>10}")
print(f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
for n_epochs, p_acc, p_mse, b_acc, b_mse in summary:
    print(f"{n_epochs:>8}  {p_acc:>10.3f}  {p_mse:>10.4f}  {b_acc:>10.3f}  {b_mse:>10.4f}")
print(f"{'='*65}")
