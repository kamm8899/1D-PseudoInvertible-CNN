'''
CAE architecture for spectrum sensing anomaly detection.
Based on Section 3.2 of the paper.

Architecture:
    Encoder: 3 Conv1d layers (16, 64, 128 filters), kernel=5, stride=2, LeakyReLU + BatchNorm
    Decoder: 4 layers with Upsample x2 + Conv1d (8, 8, 16 filters), ReLU, final sigmoid

Input:  (N, 1, 1024) — one channel, 1024 amplitude values
Latent: (N, 128, 8)  — compressed representation (~128 values)
Output: (N, 1, 1024) — reconstructed signal
'''

import torch
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        # Each layer: kernel=5, stride=2 → halves the sequence length
        # 1024 → 512 → 256 → 128
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   16,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16,  64,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64,  128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # First 3 layers: Upsample x2 + Conv1d + ReLU
        # 128 → 256 → 512 → 1024, then final output conv
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 8,  kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(8,   8,  kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(8,   16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            # Final output layer: sigmoid activation
            nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Training ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset, random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training noise (shape: [N, 2, 1024] — take channel 0 as 1-channel input)
    train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)
    # Paper uses 1-channel input of 1024 values — use I channel only
    train_noise = train_noise[:, 0:1, :]   # (N, 1, 1024)

    # Normalize to [0, 1] for sigmoid output (paper uses raw amplitude values)
    min_val = train_noise.min()
    max_val = train_noise.max()
    train_noise = (train_noise - min_val) / (max_val - min_val + 1e-8)

    # Split: 90% train, 10% validation (mirrors paper's 450k/50k split ratio)
    n_val   = int(0.1 * len(train_noise))
    n_train = len(train_noise) - n_val
    train_set, val_set = random_split(TensorDataset(train_noise), [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

    print(f"Train samples: {n_train}  |  Val samples: {n_val}")

    model     = CAE().to(device)
    criterion = nn.MSELoss()

    # Paper: SGD, lr=0.02, 50 epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for 50 epochs...\n")

    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(50):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            x = batch.to(device)
            recon = model(x)
            loss  = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                x = batch.to(device)
                recon = model(x)
                val_loss += criterion(recon, x).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:2d}/50 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best model (early stopping criterion from paper)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "spectrum_data/cae_best.pth")

    print(f"\n✅ Training complete in {time.time()-start:.1f}s")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Model saved to spectrum_data/cae_best.pth")
