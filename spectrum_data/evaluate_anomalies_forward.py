import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import norm   # for Q^{-1}(P_fa)

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD DATA ======================
test_dict = torch.load("spectrum_data/test_data.pt", weights_only=False)
test_data = test_dict["data"]
test_labels = test_dict["labels"].numpy()
test_snr = test_dict["snrs"].numpy()
test_mods = np.array(test_dict["signals"])

#need this in order to get gamma
train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)  # plain tensor

# ====================== LOAD MODELS ======================
model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)
model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)

model_psi.load_state_dict(torch.load("spectrum_data/psl_cnn_100epochs.pth", weights_only=False))
model_base.load_state_dict(torch.load("spectrum_data/baseline_100epochs.pth", weights_only=False))
model_psi.eval()
model_base.eval()

def compute_beta(model, data):
    """Compute β (Coefficient of Determination) per sample - Eq. (5) in Pablos et al."""
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i:i+128].to(device)                    # (B, 2, 1024)
            recon = model.AE(batch)

            sse = torch.sum((batch - recon)**2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst = torch.sum((batch - mean_x)**2, dim=[1, 2])

            beta = 1.0 - (sse / (sst + 1e-8))                   # avoid div-by-zero
            betas.append(beta.cpu())
    return torch.cat(betas).numpy()

# ====================== TRAINING NOISE STATISTICS (H0) ======================
print("Computing β on training noise (H0) for threshold estimation...")
train_beta_psi  = compute_beta(model_psi,  train_noise)
train_beta_base = compute_beta(model_base, train_noise)

mu_psi,  sigma_psi  = np.mean(train_beta_psi),  np.std(train_beta_psi)
mu_base, sigma_base = np.mean(train_beta_base), np.std(train_beta_base)
print(f"Psl-CNN  H0 β → mean = {mu_psi:.4f},  std = {sigma_psi:.4f}")
print(f"Baseline H0 β → mean = {mu_base:.4f}, std = {sigma_base:.4f}")

# ====================== NEYMAN-PEARSON THRESHOLD γ ======================
#pfa = probability of false alarm = P(β > γ | H0) = Q((γ - μ_β) / σ_β)
#setting pfa to 1 percent, better youdan index to determine optimal pfa
target_pfa = 0.01
gamma_psi  = mu_psi  + norm.ppf(target_pfa) * sigma_psi   # Eq. (7)
gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ Psl-CNN = {gamma_psi:.4f}, γ Baseline = {gamma_base:.4f}")

# ====================== TEST SET EVALUATION ======================
print("\nComputing β scores on test set...")
beta_psi = compute_beta(model_psi, test_data)
beta_base = compute_beta(model_base, test_data)

# ROC / AUC (lower β = anomaly → negate)
fpr_psi, tpr_psi, _ = roc_curve(test_labels, -beta_psi)
fpr_base, tpr_base, _ = roc_curve(test_labels, -beta_base)
auc_psi = auc(fpr_psi, tpr_psi)
auc_base = auc(fpr_base, tpr_base)

param_psi = sum(p.numel() for p in model_psi.parameters())
param_base = sum(p.numel() for p in model_base.parameters())


print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  γ = {gamma_base:.4f}  params = {param_base:,}")

# Save results
out_dir = Path("anomalies_PSi-NN_forward")
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()
with open("spectrum_data/evaluation_results.txt", "w") as f:
    f.write("=== EVALUATION RESULTS (Pablos et al. Step 3.3 - β + Neyman-Pearson) ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Psl-CNN AUC: {auc_psi:.4f}\n")
    f.write(f"Baseline AUC: {auc_base:.4f}\n")
    f.write(f"Psl-CNN  γ (P_fa={target_pfa}): {gamma_psi:.4f}\n")
    f.write(f"Baseline γ (P_fa={target_pfa}): {gamma_base:.4f}\n")
    f.write(f"Psl-CNN parameters: {param_psi:,}\n")
    f.write(f"Baseline parameters: {param_base:,}\n")

# ROC plot
plt.figure(figsize=(8,6))
plt.plot(fpr_psi, tpr_psi, label=f'1D Psl-CNN (AUC = {auc_psi:.3f})')
plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {auc_base:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modulation-Agnostic Anomaly Detection (β statistic)')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_PSi-NN_forward/roc_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Evaluation complete! Results saved in spectrum_data/")

# TPR by SNR range (P_fa = 0.01)
print(f"\n{'='*70}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {'Psl-CNN':<12} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {'Baseline':<12} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR by modulation type (P_fa = 0.01)
print(f"\n{'='*70}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for mod in ['qpsk', 'bpsk', '16qam', 'fm']:
    # include all noise samples so both classes (H0 and H1) are present for ROC
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    print(f"{mod:<16} {'Psl-CNN':<12} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {'Baseline':<12} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR for P_fa = 0.05
target_pfa = 0.05
gamma_psi  = mu_psi  + norm.ppf(1 - target_pfa) * sigma_psi
gamma_base = mu_base + norm.ppf(1 - target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ Psl-CNN = {gamma_psi:.4f}, γ Baseline = {gamma_base:.4f}")

# TPR by SNR range (P_fa = 0.05)
print(f"\n{'='*70}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {'Psl-CNN':<12} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {'Baseline':<12} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR by modulation type (P_fa = 0.05)
print(f"\n{'='*70}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for mod in ['qpsk', 'bpsk', '16qam', 'fm']:
    # include all noise samples so both classes (H0 and H1) are present for ROC
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    print(f"{mod:<16} {'Psl-CNN':<12} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {'Baseline':<12} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

#============================ PLOTS ======================

for (name, beta, gamma, color, fname) in [
    ("Psl-CNN",  beta_psi,  gamma_psi,  "red",    "psl_cnn"),
    ("Baseline", beta_base, gamma_base, "orange", "baseline"),
]:
    mse = 1 - beta

#Beta Signal Vs Beta Noise Plot
plt.figure(figsize=(8,6))
plt.hist(beta_psi[test_labels==0], bins=50, alpha=0.5, label='Psl-CNN Noise (H0)')
plt.hist(beta_psi[test_labels==1], bins=50, alpha=0.5, label='Psl-CNN Anomaly (H1)')
plt.axvline(gamma_psi, color='red', linestyle='--', label=f'Psl-CNN γ (P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title('Distribution of β Scores - Psl-CNN')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_PSi-NN_forward/beta_distribution_psl_cnn_forwardPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#MSE Signal Vs MSE Noise Plot
mse_psi = 1 - beta_psi  # since β = 1 - (SSE/SST), MSE is proportional to 1 - β
plt.figure(figsize=(8,6))
plt.hist(mse_psi[test_labels==0], bins=50, alpha=0.5, label='Psl-CNN Noise (H0)')
plt.hist(mse_psi[test_labels==1], bins=50, alpha=0.5, label='Psl-CNN Anomaly (H1)')
plt.axvline(1 - gamma_psi, color='red', linestyle='--', label=f'Psl-CNN MSE Threshold (P_fa={target_pfa})')
plt.xlabel('MSE Score (1 - β)')
plt.ylabel('Frequency')
plt.title('Distribution of MSE Scores - Psl-CNN')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_PSi-NN_forward/mse_distribution_forwardPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#Baseline Beta Signal Vs Baseline Beta Noise Plot
plt.figure(figsize=(8,6))
plt.hist(beta_base[test_labels==0], bins=50, alpha=0.5, label='Baseline Noise (H0)')
plt.hist(beta_base[test_labels==1], bins=50, alpha=0.5, label='Baseline Anomaly (H1)')
plt.axvline(gamma_base, color='orange', linestyle='--', label=f'Baseline γ (P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title('Distribution of β Scores - Baseline')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_PSi-NN_forward/beta_distribution_baseline_forwardPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#Baseline MSE Signal Vs Baseline MSE Noise Plot
mse_base = 1 - beta_base
plt.figure(figsize=(8,6))
plt.hist(mse_base[test_labels==0], bins=50, alpha=0.5, label='Baseline Noise (H0)')
plt.hist(mse_base[test_labels==1], bins=50, alpha=0.5, label='Baseline Anomaly (H1)')
plt.axvline(1 - gamma_base, color='orange', linestyle='--', label=f'Baseline MSE Threshold (P_fa={target_pfa})')
plt.xlabel('MSE Score (1 - β)')
plt.ylabel('Frequency')
plt.title('Distribution of MSE Scores - Baseline')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_PSi-NN_forward/mse_distribution_baseline_forwardPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close() 
