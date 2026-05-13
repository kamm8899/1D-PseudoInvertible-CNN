import os

# macOS / conda: multiple OpenMP runtimes (PyTorch + NumPy/SciPy) can abort without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np

from spectrum_paths import get_psinn_test_data_path, assert_psinn_full_channel_metadata
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import norm   # for Q^{-1}(P_fa)

from experiment_labels import ROC_CONV_AE_BASELINE, ROC_PSL_CNN, TABLE_CONV_AE_BASELINE, TABLE_PSL_CNN

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD DATA ======================
_psinn_test_path = get_psinn_test_data_path()
test_dict = torch.load(_psinn_test_path, weights_only=False)
assert_psinn_full_channel_metadata(test_dict)
if test_dict.get("generation"):
    print(f"Loaded PsiNN test set {_psinn_test_path!r}  generation={test_dict['generation']}")
test_data = test_dict["data"]
test_labels = test_dict["labels"].numpy()
test_snr = test_dict["snrs"].numpy()
test_mods = np.array(test_dict["signals"])  # convert to array so boolean indexing works

#need this in order to get gamma
train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)  # plain tensor

# ====================== LOAD MODELS ======================
model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=5, use_dropout=True).to(device)

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
            if recon.shape[-1] != batch.shape[-1]:
                recon = recon[..., :batch.shape[-1]]

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
print(f"{TABLE_PSL_CNN} H0 β → mean = {mu_psi:.4f},  std = {sigma_psi:.4f}")
print(f"{TABLE_CONV_AE_BASELINE} H0 β → mean = {mu_base:.4f}, std = {sigma_base:.4f}")

# ====================== NEYMAN-PEARSON THRESHOLD γ ======================
# Mixed tails (same P_fa for both): Psl-CNN = upper (β > γ → anomaly, ROC on +β);
# Baseline AE = lower (β < γ → anomaly, ROC on −β).
# --- Both lower (commented):
# gamma_psi  = mu_psi  + norm.ppf(target_pfa) * sigma_psi
# gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base
# --- Both upper (commented):
# gamma_psi  = mu_psi  + norm.ppf(1 - target_pfa) * sigma_psi
# gamma_base = mu_base + norm.ppf(1 - target_pfa) * sigma_base
target_pfa = 0.01
gamma_psi  = mu_psi  + norm.ppf(1.0 - target_pfa) * sigma_psi
gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ {TABLE_PSL_CNN} (upper) = {gamma_psi:.4f}, γ {TABLE_CONV_AE_BASELINE} (lower) = {gamma_base:.4f}")

# ====================== TEST SET EVALUATION ======================
print("\nComputing β scores on test set...")
beta_psi = compute_beta(model_psi, test_data)
beta_base = compute_beta(model_base, test_data)

# ROC / AUC — Psl-CNN upper tail (+β); Baseline lower tail (−β)
# fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels, -beta_psi)
# fpr_base, tpr_base, thresholds_base = roc_curve(test_labels, beta_base)
fpr_psi,  tpr_psi,  thresholds_psi  = roc_curve(test_labels, beta_psi)
fpr_base, tpr_base, thresholds_base = roc_curve(test_labels, -beta_base)
auc_psi  = auc(fpr_psi,  tpr_psi)
auc_base = auc(fpr_base, tpr_base)

# Youden Index: optimal point on ROC = max(TPR - FPR)
youden_psi  = tpr_psi  - fpr_psi
youden_base = tpr_base - fpr_base

best_idx_psi  = np.argmax(youden_psi)
best_idx_base = np.argmax(youden_base)

optimal_pfa_psi  = fpr_psi[best_idx_psi]
optimal_pfa_base = fpr_base[best_idx_base]
optimal_tpr_psi  = tpr_psi[best_idx_psi]
optimal_tpr_base = tpr_base[best_idx_base]

param_psi  = sum(p.numel() for p in model_psi.parameters())
param_base = sum(p.numel() for p in model_base.parameters())

print(f"\n=== FINAL RESULTS (Pablos et al. Step 3.3) ===")
print(f"{TABLE_PSL_CNN}  AUC: {auc_psi:.4f}  γ = {gamma_psi:.4f}  params = {param_psi:,}")
print(f"{TABLE_CONV_AE_BASELINE} AUC: {auc_base:.4f}  γ = {gamma_base:.4f}  params = {param_base:,}")
print(f"\n=== YOUDEN INDEX (Optimal Pfa) ===")
print(f"{TABLE_PSL_CNN}  optimal Pfa = {optimal_pfa_psi:.4f}  TPR = {optimal_tpr_psi:.4f}  Youden = {youden_psi[best_idx_psi]:.4f}")
print(f"{TABLE_CONV_AE_BASELINE} optimal Pfa = {optimal_pfa_base:.4f}  TPR = {optimal_tpr_base:.4f}  Youden = {youden_base[best_idx_base]:.4f}")

# Save results
out_dir = Path("anomalies_Psi-NN_inverted")
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()

with open("spectrum_data/evaluation_results.txt", "w") as f:
    f.write("=== EVALUATION RESULTS (Pablos et al. Step 3.3 - β + Neyman-Pearson) ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"{TABLE_PSL_CNN} AUC: {auc_psi:.4f}\n")
    f.write(f"{TABLE_CONV_AE_BASELINE} AUC: {auc_base:.4f}\n")
    f.write(f"{TABLE_PSL_CNN}  γ (P_fa={target_pfa}): {gamma_psi:.4f}\n")
    f.write(f"{TABLE_CONV_AE_BASELINE} γ (P_fa={target_pfa}): {gamma_base:.4f}\n")
    f.write(f"{TABLE_PSL_CNN} parameters: {param_psi:,}\n")
    f.write(f"{TABLE_CONV_AE_BASELINE} parameters: {param_base:,}\n")

# ROC plot
plt.figure(figsize=(8,6))
plt.plot(fpr_psi,  tpr_psi,  label=f'{ROC_PSL_CNN} (AUC = {auc_psi:.3f})')
plt.plot(fpr_base, tpr_base, label=f'{ROC_CONV_AE_BASELINE} (AUC = {auc_base:.3f})')
plt.scatter(optimal_pfa_psi,  optimal_tpr_psi,  marker='*', s=200, color='blue',   zorder=5, label=f'{ROC_PSL_CNN} Youden (Pfa={optimal_pfa_psi:.3f})')
plt.scatter(optimal_pfa_base, optimal_tpr_base, marker='*', s=200, color='orange', zorder=5, label=f'{ROC_CONV_AE_BASELINE} Youden (Pfa={optimal_pfa_base:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modulation-Agnostic Anomaly Detection (β statistic)')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_Psi-NN_inverted/roc_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Evaluation complete! Results saved in spectrum_data/")

# TPR by SNR range (P_fa = 0.01)
print(f"\n{'='*70}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<18} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], -beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {TABLE_PSL_CNN:<18} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {TABLE_CONV_AE_BASELINE:<18} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR by modulation type (P_fa = 0.01)
print(f"\n{'='*70}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<18} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for mod in ['qpsk', 'bpsk', '16qam', '32qam']:
    # include all noise samples so both classes (H0 and H1) are present for ROC
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], -beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    print(f"{mod:<16} {TABLE_PSL_CNN:<18} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {TABLE_CONV_AE_BASELINE:<18} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR for P_fa = 0.05 (same mixed tails as P_fa=0.01 block above)
target_pfa = 0.05
gamma_psi  = mu_psi  + norm.ppf(1.0 - target_pfa) * sigma_psi
gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base
print(f"Target P_fa = {target_pfa} → γ {TABLE_PSL_CNN} = {gamma_psi:.4f}, γ {TABLE_CONV_AE_BASELINE} = {gamma_base:.4f}")

# TPR by SNR range (P_fa = 0.05)
print(f"\n{'='*70}")
print(f"TPR BY SNR RANGE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'SNR Range':<16} {'Model':<18} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], -beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    label = f"[{snr_low:+d}, {snr_high:+d}) dB"
    print(f"{label:<16} {TABLE_PSL_CNN:<18} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {TABLE_CONV_AE_BASELINE:<18} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

# TPR by modulation type (P_fa = 0.05)
print(f"\n{'='*70}")
print(f"TPR BY MODULATION TYPE (Pablos et al. Step 3.3, γ P_fa={target_pfa})")
print(f"{'Modulation':<16} {'Model':<18} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*70}")

for mod in ['qpsk', 'bpsk', '16qam', '32qam']:
    # include all noise samples so both classes (H0 and H1) are present for ROC
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psi[mask])
    fpr_b, tpr_b, _ = roc_curve(test_labels[mask], -beta_base[mask])
    auc_p = auc(fpr_p, tpr_p)
    auc_b = auc(fpr_b, tpr_b)
    tpr_p_at_g = tpr_p[np.searchsorted(fpr_p, target_pfa, side='right') - 1]
    tpr_b_at_g = tpr_b[np.searchsorted(fpr_b, target_pfa, side='right') - 1]
    print(f"{mod:<16} {TABLE_PSL_CNN:<18} {auc_p:>6.4f}  {tpr_p_at_g:>8.4f}")
    print(f"{'':16} {TABLE_CONV_AE_BASELINE:<18} {auc_b:>6.4f}  {tpr_b_at_g:>8.4f}")
    print(f"{'─'*70}")

#============================ PLOTS (P_fa = 0.01; reset after 0.05 block above) ======================
target_pfa = 0.01
gamma_psi  = mu_psi  + norm.ppf(1.0 - target_pfa) * sigma_psi
gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base

#Beta Signal Vs Beta Noise Plot
plt.figure(figsize=(8,6))
plt.hist(beta_psi[test_labels==0], bins=50, alpha=0.5, label=f'{ROC_PSL_CNN} noise (H0)')
plt.hist(beta_psi[test_labels==1], bins=50, alpha=0.5, label=f'{ROC_PSL_CNN} signal (H1)')
plt.axvline(gamma_psi, color='red', linestyle='--', label=f'γ ({ROC_PSL_CNN}, P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title(f'Distribution of β — {ROC_PSL_CNN}')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_Psi-NN_inverted/beta_distribution_psl_cnn_nvertedPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#MSE Signal Vs MSE Noise Plot
mse_psi = 1 - beta_psi  # since β = 1 - (SSE/SST), MSE is proportional to 1 - β
plt.figure(figsize=(8,6))
plt.hist(mse_psi[test_labels==0], bins=50, alpha=0.5, label=f'{ROC_PSL_CNN} noise (H0)')
plt.hist(mse_psi[test_labels==1], bins=50, alpha=0.5, label=f'{ROC_PSL_CNN} signal (H1)')
plt.axvline(1 - gamma_psi, color='red', linestyle='--', label=f'MSE threshold ({ROC_PSL_CNN}, P_fa={target_pfa})')
plt.xlabel('MSE Score (1 - β)')
plt.ylabel('Frequency')
plt.title(f'Distribution of MSE — {ROC_PSL_CNN}')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_Psi-NN_inverted/mse_distribution_psl_cnn_nvertedPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#Baseline Beta Signal Vs Baseline Beta Noise Plot
plt.figure(figsize=(8,6))
plt.hist(beta_base[test_labels==0], bins=50, alpha=0.5, label=f'{ROC_CONV_AE_BASELINE} noise (H0)')
plt.hist(beta_base[test_labels==1], bins=50, alpha=0.5, label=f'{ROC_CONV_AE_BASELINE} signal (H1)')
plt.axvline(gamma_base, color='orange', linestyle='--', label=f'γ ({ROC_CONV_AE_BASELINE}, P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title(f'Distribution of β — {ROC_CONV_AE_BASELINE}')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_Psi-NN_inverted/beta_distribution_baseline_nvertedPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

#Baseline MSE Signal Vs Baseline MSE Noise Plot
mse_base = 1 - beta_base
plt.figure(figsize=(8,6))
plt.hist(mse_base[test_labels==0], bins=50, alpha=0.5, label=f'{ROC_CONV_AE_BASELINE} noise (H0)')
plt.hist(mse_base[test_labels==1], bins=50, alpha=0.5, label=f'{ROC_CONV_AE_BASELINE} signal (H1)')
plt.axvline(1 - gamma_base, color='orange', linestyle='--', label=f'MSE threshold ({ROC_CONV_AE_BASELINE}, P_fa={target_pfa})')
plt.xlabel('MSE Score (1 - β)')
plt.ylabel('Frequency')
plt.title(f'Distribution of MSE — {ROC_CONV_AE_BASELINE}')
plt.legend()
plt.grid(True)
plt.savefig("anomalies_Psi-NN_inverted/mse_distribution_baseline_nvertedPsi-nn.png", dpi=300, bbox_inches='tight')
plt.close()

# ====================== β/MSE DISTRIBUTIONS AT -5, 0, +5 dB ======================
# Professor Ask: "We should plot beta and MMSE for different specific SNRs, a low, medium
# and high value — we want to see how H0 and H1 space and overlap with SNR.
# For the pdfs, you can do for three different SNR values: -5dB, 0, 5dB."
# Produces a 2x2 figure per SNR: top row = β, bottom row = MSE (1-β), columns = Psl-CNN / Baseline.
# Reset to Pfa=0.01 (may have been overwritten above)
target_pfa = 0.01
gamma_psi  = mu_psi  + norm.ppf(1.0 - target_pfa) * sigma_psi
gamma_base = mu_base + norm.ppf(target_pfa) * sigma_base

for snr_val in [-6, 0, 6]:
    mask = (test_snr == snr_val)
    if mask.sum() == 0:
        print(f"No samples at SNR={snr_val:+d} dB, skipping.")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'β and MSE Distributions at SNR = {snr_val:+d} dB  (Pfa={target_pfa})', fontsize=13)

    for col, (name, beta, gamma) in enumerate([
        (TABLE_PSL_CNN,  beta_psi,  gamma_psi),
        (TABLE_CONV_AE_BASELINE, beta_base, gamma_base),
    ]):
        h0 = mask & (test_labels == 0)
        h1 = mask & (test_labels == 1)

        beta_label = '(a)' if col == 0 else '(b)'
        mse_label  = '(c)' if col == 0 else '(d)'

        # β row
        axes[0, col].hist(beta[h0], bins=40, alpha=0.6, color='steelblue',  label='Noise (H0)')
        axes[0, col].hist(beta[h1], bins=40, alpha=0.6, color='darkorange', label='Signal (H1)')
        axes[0, col].axvline(gamma, color='red', linestyle='--', label='γ threshold')
        axes[0, col].set_title(f'{name} — β Score')
        axes[0, col].set_xlabel('β')
        axes[0, col].set_ylabel('Frequency')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True)
        beta_rule = (
            'Upper-tail test: declare anomaly when β > γ.'
            if name == TABLE_PSL_CNN
            else 'Lower-tail test: declare anomaly when β < γ.'
        )
        axes[0, col].text(
            0.5, -0.22,
            f'{beta_label} {name} β score: reconstruction fidelity of H0 (noise)\n'
            f'vs H1 (signal). {beta_rule}',
            transform=axes[0, col].transAxes,
            ha='center', va='top', fontsize=8, style='italic'
        )

        # MSE row
        mse = 1 - beta
        axes[1, col].hist(mse[h0], bins=40, alpha=0.6, color='steelblue',  label='Noise (H0)')
        axes[1, col].hist(mse[h1], bins=40, alpha=0.6, color='darkorange', label='Signal (H1)')
        axes[1, col].axvline(1 - gamma, color='red', linestyle='--', label='MSE threshold')
        axes[1, col].set_title(f'{name} — MSE Score (1−β)')
        axes[1, col].set_xlabel('MSE')
        axes[1, col].set_ylabel('Frequency')
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True)
        axes[1, col].text(
            0.5, -0.22,
            f'{mse_label} {name} MSE score (1−β): reconstruction error of H0 (noise)\n'
            f'vs H1 (signal). Lower MSE = better reconstruction.',
            transform=axes[1, col].transAxes,
            ha='center', va='top', fontsize=8, style='italic'
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, wspace=0.3)
    plt.savefig(out_dir / f"snr_distributions_{snr_val:+d}dB.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SNR={snr_val:+d} dB distribution plot.")

# ====================== Pd vs SNR (Pfa = 0.01) ======================
# Professor Ask: "Redo experiments for specific SNRs: -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10
# so we can plot Pd vs SNR for Pfa = 0.01 — Three lines: ED, baseline CAE, Psi-NN."
# This section produces the Psl-CNN and Baseline lines. ED line is in energy_detector.py.
# Results saved to spectrum_data/ and combined into one figure by plot_pd_vs_snr.py.
target_pfa  = 0.01
gamma_psi   = mu_psi  + norm.ppf(1.0 - target_pfa) * sigma_psi
gamma_base  = mu_base + norm.ppf(target_pfa) * sigma_base

snr_points  = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
pd_psi_arr  = []
pd_base_arr = []

print(f"\n{'='*55}")
print(f"Pd vs SNR  (Pfa = {target_pfa})")
print(f"  P_d column 1: {TABLE_PSL_CNN}")
print(f"  P_d column 2: {TABLE_CONV_AE_BASELINE}")
print(f"{'SNR (dB)':>10}  {'P_d':>12}  {'P_d':>12}")
print(f"{'─'*55}")

for snr_db in snr_points:
    sig_mask = (test_snr == snr_db) & (test_labels == 1)
    pd_p = float(np.mean(beta_psi[sig_mask]  > gamma_psi))  if sig_mask.sum() > 0 else np.nan
    pd_b = float(np.mean(beta_base[sig_mask] < gamma_base)) if sig_mask.sum() > 0 else np.nan
    pd_psi_arr.append(pd_p)
    pd_base_arr.append(pd_b)
    print(f"{snr_db:>10d}  {pd_p:>12.4f}  {pd_b:>12.4f}")

print(f"{'─'*55}")

np.save("spectrum_data/pd_vs_snr_psinn.npy",    np.array(pd_psi_arr))
np.save("spectrum_data/pd_vs_snr_baseline.npy", np.array(pd_base_arr))
np.save("spectrum_data/snr_points.npy",         np.array(snr_points, dtype=float))
print("Saved pd_vs_snr_psinn.npy and pd_vs_snr_baseline.npy")

# ====================== AUC PER MODULATION × SNR TABLE ======================
# Professor Ask: "Determine AUC per modulation for different SNRs table."
# Rows = modulation type (qpsk, bpsk, 16qam, 32qam), columns = each of the 11 SNR points.
# Prints one table for Psl-CNN and one for Baseline.
modulations_list = ['qpsk', 'bpsk', '16qam', '32qam']

for model_name, beta_scores, use_neg_beta in [
    (TABLE_PSL_CNN, beta_psi, False),
    (TABLE_CONV_AE_BASELINE, beta_base, True),
]:
    print(f"\n{'='*90}")
    print(f"AUC PER MODULATION × SNR — {model_name}")
    print(f"{'':12}" + "".join(f"  {s:>5}" for s in snr_points))
    print("─" * 90)
    for mod in modulations_list:
        row = f"{mod:<12}"
        for snr_db in snr_points:
            snr_mask = (test_snr == snr_db)
            mask = (snr_mask & (test_mods == mod)) | (snr_mask & (test_labels == 0))
            if mask.sum() == 0 or len(np.unique(test_labels[mask])) < 2:
                row += "    N/A"
                continue
            scores = -beta_scores[mask] if use_neg_beta else beta_scores[mask]
            fpr_s, tpr_s, _ = roc_curve(test_labels[mask], scores)
            row += f"  {auc(fpr_s, tpr_s):.3f}"
        print(row)
    print("─" * 90)

