import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from pathlib import Path

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data once
train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)
train_loader = DataLoader(TensorDataset(train_noise), batch_size=128, shuffle=True)

EPOCH_LIST = [200]          

all_psi_losses = {}
all_base_losses = {}

# ====================== TRAINING FUNCTION ======================
def train_ae(model, optimizer, scheduler, model_name, epochs):
    model.train()
    losses = []
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in train_loader:
            x = batch.to(device)
            recon = model.AE(x)
            loss = nn.MSELoss()(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        scheduler.step()   # ← Learning rate decay

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  {model_name} Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
    print(f"✅ {model_name} finished in {time.time()-start:.1f} seconds")
    return losses

# ====================== MAIN TRAINING LOOP ======================
for epochs in EPOCH_LIST:
    print(f"\n{'='*60}")
    print(f"🔄 TRAINING FOR {epochs} EPOCHS")
    print(f"{'='*60}")

    # Fresh models every time
    model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)
    model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)

    optimizer_psi = torch.optim.Adam(model_psi.parameters(), lr=1e-3)
    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)

    # Scheduler
    scheduler_psi = torch.optim.lr_scheduler.StepLR(optimizer_psi, step_size=50, gamma=0.5)
    scheduler_base = torch.optim.lr_scheduler.StepLR(optimizer_base, step_size=50, gamma=0.5)

    # Train both models
    psi_losses = train_ae(model_psi, optimizer_psi, scheduler_psi, "Psl-CNN", epochs)
    base_losses = train_ae(model_base, optimizer_base, scheduler_base, "Baseline", epochs)

    all_psi_losses[epochs] = psi_losses
    all_base_losses[epochs] = base_losses

    # Save models
    torch.save(model_psi.state_dict(), f"spectrum_data/psl_cnn_{epochs}epochs.pth")
    torch.save(model_base.state_dict(), f"spectrum_data/baseline_{epochs}epochs.pth")

    print(f"✅ Saved models for {epochs} epochs")

# Combined Loss Plot
plt.figure(figsize=(10, 6))
for epochs in EPOCH_LIST:
    plt.plot(all_psi_losses[epochs], label=f'Psl-CNN ({epochs} epochs)', linewidth=2)
    plt.plot(all_base_losses[epochs], label=f'Baseline ({epochs} epochs)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Loss Curves Comparison')
plt.legend()
plt.grid(True)
plt.savefig("spectrum_data/loss_comparison.png")
plt.close()

print("\n✅ All training completed and models saved!")
print("You can now run evaluate_anomaly.py")