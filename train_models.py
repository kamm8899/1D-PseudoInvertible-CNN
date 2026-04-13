import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import csv
from pathlib import Path

# Paste your PsiNNConv1d, AE_Classifier1d, AE_Baseline_Classifier1d classes here
# (or import them if they are in another file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_noise = torch.load("spectrum_data/train_noise.pt")
train_loader = DataLoader(TensorDataset(train_noise), batch_size=128, shuffle=True)

# Models
model_psi = AE_Classifier1d(n_channels=2, n_classes=0, nf=16, k=3, use_dropout=True).to(device)
model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=0, nf=16, k=3, use_dropout=True).to(device)

optimizer_psi = torch.optim.Adam(model_psi.parameters(), lr=1e-3)
optimizer_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train_ae(model, optimizer, epochs=30, model_name="model"):
    model.train()
    losses = []                     # ← NEW: save loss per epoch
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in train_loader:
            x = batch.to(device)
            recon = model.AE(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"{model_name} Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f}")
    print(f"✅ {model_name} training finished in {time.time()-start:.1f} seconds")
    return losses

print("Training 1D Psl-CNN (PsiNN)...")
psi_losses = train_ae(model_psi, optimizer_psi, model_name="Psl-CNN")

print("\nTraining Baseline AE...")
base_losses = train_ae(model_base, optimizer_base, model_name="Baseline")

# === SAVE RESULTS ===
Path("spectrum_data").mkdir(exist_ok=True)

# Save loss history as CSV (easy to plot later)
with open("spectrum_data/loss_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "psl_cnn_loss", "baseline_loss"])
    for i in range(len(psi_losses)):
        writer.writerow([i+1, f"{psi_losses[i]:.6f}", f"{base_losses[i]:.6f}"])

# Save quick summary
with open("spectrum_data/training_summary.txt", "w") as f:
    f.write("=== TRAINING SUMMARY ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Epochs: {len(psi_losses)}\n")
    f.write(f"Psl-CNN final loss: {psi_losses[-1]:.6f}\n")
    f.write(f"Baseline final loss: {base_losses[-1]:.6f}\n")
    f.write(f"Psl-CNN parameters: {sum(p.numel() for p in model_psi.parameters()):,}\n")
    f.write(f"Baseline parameters: {sum(p.numel() for p in model_base.parameters()):,}\n")

torch.save(model_psi.state_dict(), "spectrum_data/psl_cnn_ae.pth")
torch.save(model_base.state_dict(), "spectrum_data/baseline_ae.pth")
print("✅ Models and results saved in spectrum_data/")