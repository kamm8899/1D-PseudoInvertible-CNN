import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

from psinn_layer_1d_pablos import AE_Pablos1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)
train_noise = train_noise[:, 0:1, :]   # I channel only → (N, 1, 1024)
train_loader = DataLoader(TensorDataset(train_noise), batch_size=256, shuffle=True)

EPOCHS = 200

model = AE_Pablos1d(nf=16, k=5, use_dropout=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training for {EPOCHS} epochs...\n")

losses = []
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for (batch,) in train_loader:
        x = batch.to(device)
        recon = model.AE(x)
        if recon.shape[-1] != x.shape[-1]:
            recon = recon[..., :x.shape[-1]]
        loss = nn.MSELoss()(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    scheduler.step()
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        print(f"  Pablos Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

print(f"\n✅ Finished in {time.time()-start:.1f} seconds")

torch.save(model.state_dict(), f"spectrum_data/pablos_{EPOCHS}epochs.pth")
print(f"Saved spectrum_data/pablos_{EPOCHS}epochs.pth")

plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2, label='AE_Pablos1d')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('AE_Pablos1d Training Loss')
plt.legend()
plt.grid(True)
plt.savefig("spectrum_data/pablos_loss.png")
plt.close()
print("Saved spectrum_data/pablos_loss.png")
