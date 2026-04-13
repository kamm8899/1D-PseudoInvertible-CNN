import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import csv
import matplotlib.pyplot as plt
from pathlib import Path

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data once
train_noise = torch.load("spectrum_data/train_noise.pt")
train_loader = DataLoader(TensorDataset(train_noise), batch_size=128, shuffle=True)

EPOCH_LIST = [30, 50, 100]          

all_psi_losses = {}
all_base_losses = {}

for epochs in EPOCH_LIST:
    print(f"\n{'='*60}")
    print(f"🔄 TRAINING FOR {epochs} EPOCHS")
    print(f"{'='*60}")

    # Fresh models every time
    model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)
    model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)

    optimizer_psi = torch.optim.Adam(model_psi.parameters(), lr=1e-3)
    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    def train_ae(model, optimizer, model_name):
        model.train()
        losses = []
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
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs-1:
                print(f"  {model_name} Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
        print(f"✅ {model_name} finished in {time.time()-start:.1f} seconds")
        return losses

    # Train both
    psi_losses = train_ae(model_psi, optimizer_psi, "Psl-CNN")
    base_losses = train_ae(model_base, optimizer_base, "Baseline")

    all_psi_losses[epochs] = psi_losses
    all_base_losses[epochs] = base_losses

    # Save models for this epoch count
    torch.save(model_psi.state_dict(), f"spectrum_data/psl_cnn_{epochs}epochs.pth")
    torch.save(model_base.state_dict(), f"spectrum_data/baseline_{epochs}epochs.pth")

    print(f"✅ Saved models for {epochs} epochs")

# === COMBINED LOSS PLOT ===
plt.figure(figsize=(10, 6))
for epochs in EPOCH_LIST:
    plt.plot(all_psi_losses[epochs], label=f'Psl-CNN ({epochs} epochs)', linewidth=2)
    plt.plot(all_base_losses[epochs], label=f'Baseline ({epochs} epochs)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Loss Curves Comparison — 30 / 50 / 100 Epochs')
plt.legend()
plt.grid(True)
plt.savefig("spectrum_data/loss_comparison_30_50_100.png")
plt.show()

# === COMBINED SUMMARY TABLE ===
print("\n" + "="*70)
print("FINAL SUMMARY TABLE (30 / 50 / 100 epochs)")
print("="*70)
print(f"{'Epochs':<8} {'Psl-CNN Final Loss':<20} {'Baseline Final Loss':<20} {'Psl-CNN Params':<15}")
print("-" * 70)

for epochs in EPOCH_LIST:
    psi_final = all_psi_losses[epochs][-1]
    base_final = all_base_losses[epochs][-1]
    psi_params = sum(p.numel() for p in model_psi.parameters())   # same for all
    print(f"{epochs:<8} {psi_final:<20.6f} {base_final:<20.6f} {psi_params:<15,}")

print("\n✅ All results saved in spectrum_data/ folder:")
print("   • loss_comparison_30_50_100.png")
print("   • psl_cnn_XXepochs.pth and baseline_XXepochs.pth for each run")
print("   • You can now run evaluate_anomaly.py on any of the saved models")

# Optional: save the summary as a text file
with open("spectrum_data/experiment_summary_30_50_100.txt", "w") as f:
    f.write("EXPERIMENT SUMMARY - 30 / 50 / 100 EPOCHS\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"{'Epochs':<8} {'Psl-CNN Loss':<18} {'Baseline Loss':<18} {'Psl-CNN Params'}\n")
    for epochs in EPOCH_LIST:
        f.write(f"{epochs:<8} {all_psi_losses[epochs][-1]:<18.6f} {all_base_losses[epochs][-1]:<18.6f} {psi_params:,}\n")

print("✅ Full summary saved as experiment_summary_30_50_100.txt")