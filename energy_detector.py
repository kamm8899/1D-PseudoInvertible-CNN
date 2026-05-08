'''
Energy Detector (ED) — Benchmark for Spectrum Sensing
Implements the classical energy detection approach for comparison against
PsiNN and CAE anomaly detectors.

Detection rule: E = mean(I² + Q²) per sample.
Threshold set via Neyman-Pearson criterion on unnormalized training noise (H0).
Higher energy → signal present (H1).
'''

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

# ====================== LOAD DATA ======================
# Unnormalized training noise — used to estimate H0 energy distribution
train_noise_raw = torch.load("spectrum_data/train_noise_raw.pt", weights_only=False)  # (N, 2, 1024)

# Unnormalized test data — per-sample normalization would destroy power info
test_dict = torch.load("spectrum_data/test_data_raw.pt", weights_only=False)
test_data_raw = test_dict["data"]          # (N, 2, 1024)
test_labels   = test_dict["labels"].numpy()
test_snr      = test_dict["snrs"].numpy()
test_mods     = np.array(test_dict["signals"])

# ====================== ENERGY SCORE ======================
def energy_score(data: torch.Tensor) -> np.ndarray:
    """Mean squared power per sample across both I and Q channels: mean(I² + Q²)."""
    return data.pow(2).mean(dim=[1, 2]).numpy()

# ====================== H0 DISTRIBUTION FROM TRAINING NOISE ======================
print("Computing energy scores on training noise (H0)...")
train_energy = energy_score(train_noise_raw)

mu_e    = np.mean(train_energy)
sigma_e = np.std(train_energy)
print(f"H0 energy → mean = {mu_e:.6f},  std = {sigma_e:.6f}")

# ====================== NEYMAN-PEARSON THRESHOLD ======================
# Upper tail: E > γ → signal detected
# P(E > γ | H0) = 1 - Φ((γ - μ) / σ) = Pfa  →  γ = μ + Φ⁻¹(1 - Pfa) · σ
target_pfa = 0.01
gamma = mu_e + norm.ppf(1 - target_pfa) * sigma_e
print(f"Target Pfa = {target_pfa} → γ = {gamma:.6f}")

# ====================== TEST SET ENERGY SCORES ======================
print("\nComputing energy scores on test set...")
test_energy = energy_score(test_data_raw)

# ====================== OVERALL ROC / AUC ======================
fpr, tpr, _ = roc_curve(test_labels, test_energy)
auc_ed = auc(fpr, tpr)

# Youden index
youden_idx  = np.argmax(tpr - fpr)
optimal_pfa = fpr[youden_idx]
optimal_tpr = tpr[youden_idx]

print(f"\n=== ENERGY DETECTOR RESULTS ===")
print(f"AUC       : {auc_ed:.4f}")
print(f"γ (Pfa={target_pfa}): {gamma:.6f}")
print(f"Youden optimal Pfa = {optimal_pfa:.4f}  TPR = {optimal_tpr:.4f}")

# ====================== Pd vs SNR ======================
snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
pd_ed = []

print(f"\n{'='*55}")
print(f"Pd vs SNR  (Pfa = {target_pfa})")
print(f"{'SNR (dB)':>10}  {'ED Pd':>10}  {'N signal':>10}")
print(f"{'─'*55}")

for snr_db in snr_points:
    snr_mask = (test_snr == snr_db)
    sig_mask = snr_mask & (test_labels == 1)

    if sig_mask.sum() == 0:
        pd_ed.append(np.nan)
        print(f"{snr_db:>10d}  {'N/A':>10}")
        continue

    pd = float(np.mean(test_energy[sig_mask] > gamma))
    pd_ed.append(pd)
    print(f"{snr_db:>10d}  {pd:>10.4f}  {int(sig_mask.sum()):>10d}")

print(f"{'─'*55}")

pd_ed = np.array(pd_ed)

# Save for combined Pd vs SNR plot
np.save("spectrum_data/pd_vs_snr_ed.npy",   pd_ed)
np.save("spectrum_data/snr_points.npy",     np.array(snr_points, dtype=float))

# ====================== TPR BY SNR RANGE ======================
print(f"\n{'='*55}")
print(f"TPR BY SNR RANGE  (Pfa = {target_pfa})")
print(f"{'SNR Range':<18}  {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*55}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0 or len(np.unique(test_labels[mask])) < 2:
        continue
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], test_energy[mask])
    auc_s    = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"[{snr_low:+d}, {snr_high:+d}) dB        {auc_s:>6.4f}  {tpr_at_g:>8.4f}")

# ====================== TPR BY MODULATION ======================
print(f"\n{'='*55}")
print(f"TPR BY MODULATION  (Pfa = {target_pfa})")
print(f"{'Modulation':<16}  {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*55}")

for mod in ['qpsk', 'bpsk', '16qam', '32qam']:
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0 or len(np.unique(test_labels[mask])) < 2:
        continue
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], test_energy[mask])
    auc_s    = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"{mod:<16}  {auc_s:>6.4f}  {tpr_at_g:>8.4f}")

# ====================== OUTPUT FOLDER ======================
out_dir = Path("anomalies_ED")
out_dir.mkdir(exist_ok=True)

# ROC plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Energy Detector (AUC = {auc_ed:.3f})')
plt.scatter(optimal_pfa, optimal_tpr, marker='*', s=200, color='red', zorder=5,
            label=f'Youden (Pfa={optimal_pfa:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Energy Detector')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "roc_ed.png", dpi=300, bbox_inches='tight')
plt.close()

# Energy score distribution (H0 vs H1)
plt.figure(figsize=(8, 6))
plt.hist(test_energy[test_labels == 0], bins=50, alpha=0.6, color='steelblue',  label='Noise (H0)')
plt.hist(test_energy[test_labels == 1], bins=50, alpha=0.6, color='darkorange', label='Signal (H1)')
plt.axvline(gamma, color='red', linestyle='--', label=f'γ (Pfa={target_pfa})')
plt.xlabel('Energy Score  mean(I² + Q²)')
plt.ylabel('Frequency')
plt.title('Energy Score Distribution — ED')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "energy_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Energy Detector complete! Results saved in {out_dir}/")
print(f"   pd_vs_snr_ed.npy saved to spectrum_data/ for combined Pd vs SNR plot")
