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

from cae_spectrum import CAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD DATA ======================
test_dict = torch.load("spectrum_data/test_data.pt", weights_only=False)
test_data_raw = test_dict["data"]          # (N, 2, 1024)
test_labels = test_dict["labels"].numpy()
test_snr = test_dict["snrs"].numpy()
test_mods = np.array(test_dict["signals"])

# CAE expects (N, 1, 1024) — use I channel only, then normalize to [0,1]
test_data = test_data_raw[:, 0:1, :]      # (N, 1, 1024)

train_noise_raw = torch.load("spectrum_data/train_noise.pt", weights_only=False)  # plain tensor
train_noise = train_noise_raw[:, 0:1, :]  # (N, 1, 1024)

# Normalize using train statistics (must match training preprocessing)
min_val = train_noise.min()
max_val = train_noise.max()
train_noise = (train_noise - min_val) / (max_val - min_val + 1e-8)
test_data   = (test_data   - min_val) / (max_val - min_val + 1e-8)

# ====================== LOAD MODEL ======================
model_cae = CAE().to(device)
model_cae.load_state_dict(torch.load("spectrum_data/cae_best.pth", weights_only=False))
model_cae.eval()

def compute_beta(model, data):
    """Compute β (Coefficient of Determination) per sample - Eq. (5) in Pablos et al."""
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i:i+128].to(device)   # (B, 1, 1024)
            recon = model(batch)

            sse = torch.sum((batch - recon)**2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst = torch.sum((batch - mean_x)**2, dim=[1, 2])

            beta = 1.0 - (sse / (sst + 1e-8))
            betas.append(beta.cpu())
    return torch.cat(betas).numpy()

# ====================== TRAINING NOISE STATISTICS (H0) ======================
print("Computing β on training noise (H0) for threshold estimation...")
train_beta = compute_beta(model_cae, train_noise)

mu_e, sigma_e = np.mean(train_beta), np.std(train_beta)
print(f"CAE H0 β → mean = {mu_e:.4f},  std = {sigma_e:.4f}")

# ====================== NEYMAN-PEARSON THRESHOLD γ ======================
target_pfa = 0.01
# gamma = mu_e + norm.ppf(1 - target_pfa) * sigma_e  # upper-tail (inverted)
gamma = mu_e - norm.ppf(1 - target_pfa) * sigma_e   # lower-tail: P(β < γ | H0) = P_fa
print(f"Target P_fa = {target_pfa} → γ = {gamma:.4f}")

# ====================== TEST SET EVALUATION ======================
print("\nComputing β scores on test set...")
beta_cae = compute_beta(model_cae, test_data)

# ROC / AUC (higher β = anomaly — model reconstructs signals better than noise)
# fpr_cae, tpr_cae, thresholds_cae = roc_curve(test_labels, beta_cae)   # inverted score
fpr_cae, tpr_cae, thresholds_cae = roc_curve(test_labels, -beta_cae)
auc_cae = auc(fpr_cae, tpr_cae)

# Youden Index
youden_cae = tpr_cae - fpr_cae
best_idx = np.argmax(youden_cae)
optimal_pfa = fpr_cae[best_idx]
optimal_tpr = tpr_cae[best_idx]

param_cae = sum(p.numel() for p in model_cae.parameters())

print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"CAE  AUC: {auc_cae:.4f}  γ = {gamma:.4f}  params = {param_cae:,}")
print(f"\n=== YOUDEN INDEX (Optimal Pfa) ===")
print(f"CAE  optimal Pfa = {optimal_pfa:.4f}  TPR = {optimal_tpr:.4f}  Youden = {youden_cae[best_idx]:.4f}")

# ====================== OUTPUT FOLDER ======================
out_dir = Path("anomalies_CAE_inverted")
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()

with open("spectrum_data/evaluation_results_cae_inverted.txt", "w") as f:
    f.write("=== EVALUATION RESULTS (Pablos et al. Step 3.3 - β + Neyman-Pearson) ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"CAE AUC: {auc_cae:.4f}\n")
    f.write(f"CAE γ (P_fa={target_pfa}): {gamma:.4f}\n")
    f.write(f"CAE parameters: {param_cae:,}\n")

# ROC plot
plt.figure(figsize=(8,6))
plt.plot(fpr_cae, tpr_cae, label=f'CAE Inverted (AUC = {auc_cae:.3f})')
plt.scatter(optimal_pfa, optimal_tpr, marker='*', s=200, color='blue', zorder=5,
            label=f'CAE Inverted Youden (Pfa={optimal_pfa:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modulation-Agnostic Anomaly Detection (β statistic) — CAE Inverted')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "roc_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# TPR by SNR range (P_fa = 0.01)
print(f"\n{'='*60}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*60}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    # fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta_cae[mask])   # inverted score
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], -beta_cae[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {'CAE Inv':<12} {auc_s:>6.4f}  {tpr_at_g:>8.4f}")
    print(f"{'─'*60}")

# TPR by modulation type (P_fa = 0.01)
print(f"\n{'='*60}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*60}")

for mod in ['qpsk', 'bpsk', '16qam', 'fm']:
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    # fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta_cae[mask])   # inverted score
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], -beta_cae[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"{mod:<16} {'CAE Inv':<12} {auc_s:>6.4f}  {tpr_at_g:>8.4f}")
    print(f"{'─'*60}")

# TPR for P_fa = 0.05
target_pfa = 0.05
# gamma = mu_e + norm.ppf(1 - target_pfa) * sigma_e  # upper-tail (inverted)
gamma = mu_e - norm.ppf(1 - target_pfa) * sigma_e   # lower-tail: P(β < γ | H0) = P_fa
print(f"\nTarget P_fa = {target_pfa} → γ = {gamma:.4f}")

print(f"\n{'='*60}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*60}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    # fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta_cae[mask])   # inverted score
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], -beta_cae[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {'CAE Inv':<12} {auc_s:>6.4f}  {tpr_at_g:>8.4f}")
    print(f"{'─'*60}")

print(f"\n{'='*60}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*60}")

for mod in ['qpsk', 'bpsk', '16qam', 'fm']:
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    # fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta_cae[mask])   # inverted score
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], -beta_cae[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"{mod:<16} {'CAE Inv':<12} {auc_s:>6.4f}  {tpr_at_g:>8.4f}")
    print(f"{'─'*60}")

# ====================== DISTRIBUTION PLOTS ======================
plt.figure(figsize=(8,6))
plt.hist(beta_cae[test_labels==0], bins=50, alpha=0.5, label='CAE Inverted Noise (H0)')
plt.hist(beta_cae[test_labels==1], bins=50, alpha=0.5, label='CAE Inverted Anomaly (H1)')
plt.axvline(gamma, color='red', linestyle='--', label=f'CAE Inverted γ (P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title('Distribution of β Scores - CAE Inverted')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "beta_distribution_cae_inverted.png", dpi=300, bbox_inches='tight')
plt.close()

mse_cae = 1 - beta_cae
plt.figure(figsize=(8,6))
plt.hist(mse_cae[test_labels==0], bins=50, alpha=0.5, label='CAE Inverted Noise (H0)')
plt.hist(mse_cae[test_labels==1], bins=50, alpha=0.5, label='CAE Inverted Anomaly (H1)')
plt.axvline(1 - gamma, color='red', linestyle='--', label=f'CAE Inverted MSE Threshold (P_fa={target_pfa})')
plt.xlabel('MSE Score (1 - β)')
plt.ylabel('Frequency')
plt.title('Distribution of MSE Scores - CAE Inverted')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "mse_distribution_cae_inverted.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Evaluation complete! Results saved in {out_dir}/")
