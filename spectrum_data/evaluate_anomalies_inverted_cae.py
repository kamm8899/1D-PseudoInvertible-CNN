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

from cae_spectrum import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD DATA ======================
test_dict = torch.load("spectrum_data/test_data.pt", weights_only=False)
test_data = test_dict["data"]
test_labels = test_dict["labels"].numpy()
test_snr = test_dict["snrs"].numpy()
test_mods = test_dict["signals"]  # list of modulation types (strings)

#need this in order to get gamma
train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)  # plain tensor

# ====================== LOAD MODELS ======================
model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)
model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=3, use_dropout=True).to(device)

model_psi.load_state_dict(torch.load("spectrum_data/psl_cnn_200epochs.pth", weights_only=False))
model_base.load_state_dict(torch.load("spectrum_data/baseline_200epochs.pth", weights_only=False))
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
gamma_psi  = mu_psi  + norm.ppf(1 - target_pfa) * sigma_psi   # upper tail: β > γ → anomaly
gamma_base = mu_base + norm.ppf(1 - target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ Psl-CNN = {gamma_psi:.4f}, γ Baseline = {gamma_base:.4f}")

# ====================== TEST SET EVALUATION ======================
print("\nComputing β scores on test set...")
beta_psi = compute_beta(model_psi, test_data)
beta_base = compute_beta(model_base, test_data)

# ROC / AUC (higher β = anomaly — model reconstructs signals better than noise)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels, beta_psi)
fpr_base, tpr_base, thresholds_base = roc_curve(test_labels, beta_base)
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)

# Youden Index: optimal point on ROC = max(TPR - FPR)
# This gives the best trade-off between sensitivity and specificity
#youden_psi  = tpr_psi  - fpr_psi
#youden_base = tpr_base - fpr_base

#best_idx_psi  = np.argmax(youden_psi)
#best_idx_base = np.argmax(youden_base)

#optimal_pfa_psi  = fpr_psi[best_idx_psi]   # FPR at optimal point = Pfa
#optimal_pfa_base = fpr_base[best_idx_base]
#optimal_tpr_psi  = tpr_psi[best_idx_psi]
#optimal_tpr_base = tpr_base[best_idx_base]

param_psi  = sum(p.numel() for p in model_psi.parameters())
param_base = sum(p.numel() for p in model_base.parameters())

print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  γ = {gamma_base:.4f}  params = {param_base:,}")
print(f"\n=== YOUDEN INDEX (Optimal Pfa) ===")
print(f"Psl-CNN  optimal Pfa = {optimal_pfa_psi:.4f}  TPR = {optimal_tpr_psi:.4f}  Youden = {youden_psi[best_idx_psi]:.4f}")
print(f"Baseline optimal Pfa = {optimal_pfa_base:.4f}  TPR = {optimal_tpr_base:.4f}  Youden = {youden_base[best_idx_base]:.4f}")

# Save results
Path("spectrum_data").mkdir(exist_ok=True)
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
plt.plot(fpr_psi,  tpr_psi,  label=f'1D Psl-CNN (AUC = {auc_psi:.3f})')
plt.plot(fpr_base, tpr_base, label=f'Baseline  (AUC = {auc_base:.3f})')
plt.scatter(optimal_pfa_psi,  optimal_tpr_psi,  marker='*', s=200, color='blue',  zorder=5, label=f'Psl-CNN  Youden (Pfa={optimal_pfa_psi:.3f})')
plt.scatter(optimal_pfa_base, optimal_tpr_base, marker='*', s=200, color='orange', zorder=5, label=f'Baseline Youden (Pfa={optimal_pfa_base:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modulation-Agnostic Anomaly Detection (β statistic)')
plt.legend()
plt.grid(True)
plt.savefig("spectrum_data/roc_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Evaluation complete! Results saved in spectrum_data/")

#Things to do: 
#target pfa == set 1#
# Do I need to do this? #setting pfa to 1 percent, better youdan index to determine optimal pfa
#how to use the ROC curve to get a better target pfa using the youdan index?

# Signal to noise ratio SNR to accuracy -- for each of the mods

#Ask professor --- Do I need to show anything on the accuracy for the epochs--do we need to show convergence 
# do 50 Epochs 

#Deep Learning -- Transformers Base Model , Autoencoder Decoder, ResNet, 

#TPR Based of SNR
# ROC / AUC (higher β = anomaly — model reconstructs signals better than noise)
mask= (test_snr >= -10) & (test_snr <= -5)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= -5) & (test_snr <= -0)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= 0) & (test_snr <= 5)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= 5) & (test_snr <= 10)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

#Modulation Type TPR breakdown 

mask= test_mods == 'qpsk'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")


mask= test_mods == 'bpsk'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == '16qam'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == 'fm'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == 'none'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

##Maybe make a plot at some point add for loop to make easier 

#TPR for .05 pfa
#change print statements to add SNR range
target_pfa = 0.05
gamma_psi  = mu_psi  + norm.ppf(1 - target_pfa) * sigma_psi   # upper tail: β > γ → anomaly
gamma_base = mu_base + norm.ppf(1 - target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ Psl-CNN = {gamma_psi:.4f}, γ Baseline = {gamma_base:.4f}")

print("\nComputing β scores on test set...")
beta_psi = compute_beta(model_psi, test_data)
beta_base = compute_beta(model_base, test_data)

mask= (test_snr >= -10) & (test_snr <= -5)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= -5) & (test_snr <= -0)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= 0) & (test_snr <= 5)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= (test_snr >= 5) & (test_snr <= 10)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == 'bpsk'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == '16qam'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == 'fm'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

mask= test_mods == 'none'
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels[mask], beta_psi[mask])
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)
print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"Psl-CNN  AUC: {auc_psi:.4f}  tpr: { tpr_psi:.4f} γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"Baseline AUC: {auc_base:.4f}  tpr_base: {tpr_base:.4f} γ = {gamma_base:.4f}  params = {param_base:,}")

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
plt.savefig("spectrum_data/beta_distribution_psl_cnn.png", dpi=300, bbox_inches='tight')
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
plt.savefig("spectrum_data/mse_distribution_psl_cnn.png", dpi=300, bbox_inches='tight')
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
plt.savefig("spectrum_data/beta_distribution_baseline.png", dpi=300, bbox_inches='tight')
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
plt.savefig("spectrum_data/mse_distribution_baseline.png", dpi=300, bbox_inches='tight')
plt.close() 

