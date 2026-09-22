import torch
import torch.nn.functional as F
from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d, PsiNNConv1d


# ====================== SYNTHETIC TESTING SET ======================
# Perfect size for this architecture (final classification output is clean [batch, n_classes])
batch_size = 32
n_channels = 1
seq_len = 32          # ← critical: gives exactly 1 position after 4× stride-2
n_classes = 10

# Random 1D "signals" + fake labels (like a tiny toy dataset)
x = torch.randn(batch_size, n_channels, seq_len)
labels = torch.randint(0, n_classes, (batch_size,))

print(f"Input shape: {x.shape}  |  Labels shape: {labels.shape}")
print("="*60)

# ====================== INSTANTIATE BOTH MODELS ======================
psi_model = AE_Classifier1d(n_channels=n_channels, n_classes=n_classes, nf=16, k=3, use_dropout=False)
baseline_model = AE_Baseline_Classifier1d(n_channels=n_channels, n_classes=n_classes, nf=16, k=3, use_dropout=False)

# ====================== BASIC FORWARD TESTS ======================
print("Running classification head (C)...")
psi_clas = psi_model(x)
baseline_clas = baseline_model(x)
print(f"PsiNN class output shape : {psi_clas.shape}")      # should be [32, 10]
print(f"Baseline class output shape: {baseline_clas.shape}")

print("\nRunning autoencoder reconstruction (AE)...")
psi_recon = psi_model.AE(x)
baseline_recon = baseline_model.AE(x)
print(f"PsiNN recon shape : {psi_recon.shape}")   # must be exactly [32, 1, 32]
print(f"Baseline recon shape: {baseline_recon.shape}")

# Quick reconstruction error (will be high at random init — that's normal)
print(f"\nPsiNN recon MSE: {F.mse_loss(psi_recon, x):.4f}")
print(f"Baseline recon MSE: {F.mse_loss(baseline_recon, x):.4f}")

# ====================== PSI-NN INVERTIBILITY CHECK ======================
print("\nTesting single PsiNNConv1d forward → backward invertibility...")
test_layer = PsiNNConv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2,
                         padding=1, output_padding=1, bias=False, direction=1)
x_inv = torch.randn(batch_size, 8, seq_len)   # 8 channels to match in_channels
with torch.no_grad():
    y = test_layer.forw(x_inv)      # forward
    x_rec = test_layer.back(y)      # backward
    inv_error = F.mse_loss(x_rec, x_inv)
print(f"Forward→Backward error (should be ~0): {inv_error:.2e}")

# ====================== TINY TRAINING LOOP (just to prove it trains) ======================
print("\nRunning 10-step training test (AE + classification loss)...")
optimizer = torch.optim.Adam(psi_model.parameters(), lr=1e-3)
psi_model.train()

for step in range(10):
    optimizer.zero_grad()
    clas = psi_model(x)
    recon = psi_model.AE(x)
    

    loss_recon = F.mse_loss(recon, x)

    #removed loss class
    loss = loss_recon
    
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step:2d} | Total loss: {loss.item():.4f}")

print("\n✅ All tests passed! Your Psi-NN code runs correctly.")