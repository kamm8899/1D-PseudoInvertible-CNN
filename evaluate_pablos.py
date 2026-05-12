'''
Evaluation script for AE_Pablos1d — PsiNN version of the Pablos et al. CAE.
Uses I channel only (1, 1024) to match the CAE input format.
Computes β statistic, ROC/AUC, Pd vs SNR, and saves pd_vs_snr_pablos.npy.

Run order:
    train_pablos.py         → spectrum_data/pablos_200epochs.pth
    evaluate_pablos.py      → spectrum_data/pd_vs_snr_pablos.npy
    plot_pd_vs_snr.py       → pd_vs_snr_combined.png
'''

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
from scipy.stats import norm

from psinn_layer_1d_pablos import AE_Pablos1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== LOAD DATA (I channel only) ======================
_psinn_test_path = get_psinn_test_data_path()
test_dict   = torch.load(_psinn_test_path, weights_only=False)
assert_psinn_full_channel_metadata(test_dict)
if test_dict.get("generation"):
    print(f"Loaded Pablos test set {_psinn_test_path!r}  generation={test_dict['generation']}")
test_data   = test_dict["data"][:, 0:1, :]        # (N, 1, 1024)
test_labels = test_dict["labels"].numpy()
test_snr    = test_dict["snrs"].numpy()
test_mods   = np.array(test_dict["signals"])

train_noise_full = torch.load("spectrum_data/train_noise.pt", weights_only=False)
train_noise      = train_noise_full[:, 0:1, :]     # (N, 1, 1024)

# ====================== LOAD MODEL ======================
model = AE_Pablos1d(nf=16, k=5, use_dropout=True).to(device)
model.load_state_dict(torch.load("spectrum_data/pablos_200epochs.pth", weights_only=False))
model.eval()
print(f"Loaded AE_Pablos1d — {sum(p.numel() for p in model.parameters()):,} parameters")


def compute_beta(model, data):
    """Coefficient of determination β = 1 − SSE/SST per sample."""
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i:i+128].to(device)
            recon = model.AE(batch)
            if recon.shape[-1] != batch.shape[-1]:
                recon = recon[..., :batch.shape[-1]]
            sse    = torch.sum((batch - recon) ** 2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst    = torch.sum((batch - mean_x) ** 2, dim=[1, 2])
            betas.append((1.0 - sse / (sst + 1e-8)).cpu())
    return torch.cat(betas).numpy()


# ====================== H0 STATISTICS ======================
print("Computing β on training noise (H0) for threshold estimation...")
train_beta = compute_beta(model, train_noise)
mu, sigma  = np.mean(train_beta), np.std(train_beta)
print(f"Pablos H0 β → mean = {mu:.4f}, std = {sigma:.4f}")

target_pfa = 0.01
gamma = mu + norm.ppf(1 - target_pfa) * sigma
print(f"Target P_fa = {target_pfa} → γ = {gamma:.4f}")

# ====================== TEST SET EVALUATION ======================
print("\nComputing β scores on test set...")
beta = compute_beta(model, test_data)

fpr, tpr, _ = roc_curve(test_labels, beta)
auc_score   = auc(fpr, tpr)

youden   = tpr - fpr
best_idx = np.argmax(youden)
print(f"\nPablos AUC:          {auc_score:.4f}")
print(f"Youden optimal Pfa:  {fpr[best_idx]:.4f}  TPR = {tpr[best_idx]:.4f}")

# ====================== SAVE RESULTS ======================
out_dir = Path("anomalies_Pablos")
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()

with open("spectrum_data/evaluation_results_pablos.txt", "w") as f:
    f.write("=== AE_Pablos1d Evaluation Results ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"AUC:        {auc_score:.4f}\n")
    f.write(f"γ (Pfa=0.01): {gamma:.4f}\n")
    f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ROC plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AE_Pablos1d (AUC = {auc_score:.3f})', linewidth=2)
plt.scatter(fpr[best_idx], tpr[best_idx], marker='*', s=200, color='red', zorder=5,
            label=f'Youden (Pfa={fpr[best_idx]:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — AE_Pablos1d')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "roc_pablos.png", dpi=300, bbox_inches='tight')
plt.close()

# β distribution
plt.figure(figsize=(8, 6))
plt.hist(beta[test_labels == 0], bins=50, alpha=0.5, label='Noise (H0)')
plt.hist(beta[test_labels == 1], bins=50, alpha=0.5, label='Signal (H1)')
plt.axvline(gamma, color='red', linestyle='--', label=f'γ (P_fa={target_pfa})')
plt.xlabel('β Score')
plt.ylabel('Frequency')
plt.title('β Distribution — AE_Pablos1d')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "beta_distribution_pablos.png", dpi=300, bbox_inches='tight')
plt.close()

# MSE distribution
mse = 1 - beta
plt.figure(figsize=(8, 6))
plt.hist(mse[test_labels == 0], bins=50, alpha=0.5, label='Noise (H0)')
plt.hist(mse[test_labels == 1], bins=50, alpha=0.5, label='Signal (H1)')
plt.axvline(1 - gamma, color='red', linestyle='--', label=f'MSE threshold (P_fa={target_pfa})')
plt.xlabel('MSE Score (1 − β)')
plt.ylabel('Frequency')
plt.title('MSE Distribution — AE_Pablos1d')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "mse_distribution_pablos.png", dpi=300, bbox_inches='tight')
plt.close()

# ====================== SNR-SPECIFIC DISTRIBUTIONS (-6, 0, +6 dB) ======================
for snr_val in [-6, 0, 6]:
    mask = (test_snr == snr_val)
    if mask.sum() == 0:
        print(f"No samples at SNR={snr_val:+d} dB, skipping.")
        continue

    h0 = mask & (test_labels == 0)
    h1 = mask & (test_labels == 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'AE_Pablos1d — β and MSE at SNR = {snr_val:+d} dB  (Pfa={target_pfa})', fontsize=13)

    # β
    axes[0].hist(beta[h0], bins=40, alpha=0.6, color='steelblue',  label='Noise (H0)')
    axes[0].hist(beta[h1], bins=40, alpha=0.6, color='darkorange', label='Signal (H1)')
    axes[0].axvline(gamma, color='red', linestyle='--', label='γ threshold')
    axes[0].set_title('β Score')
    axes[0].set_xlabel('β')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # MSE
    axes[1].hist(mse[h0], bins=40, alpha=0.6, color='steelblue',  label='Noise (H0)')
    axes[1].hist(mse[h1], bins=40, alpha=0.6, color='darkorange', label='Signal (H1)')
    axes[1].axvline(1 - gamma, color='red', linestyle='--', label='MSE threshold')
    axes[1].set_title('MSE Score (1 − β)')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / f"snr_distributions_pablos_{snr_val:+d}dB.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SNR={snr_val:+d} dB distribution plot.")

# ====================== TPR BY SNR ======================
print(f"\n{'='*55}")
print(f"TPR BY SNR RANGE (P_fa = {target_pfa})")
print(f"{'SNR Range':<16} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*55}")

for snr_low, snr_high in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
    mask = (test_snr >= snr_low) & (test_snr < snr_high)
    if mask.sum() == 0:
        continue
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"[{snr_low:+d}, {snr_high:+d}) dB  {auc_s:>6.4f}  {tpr_at_g:>8.4f}")

# ====================== TPR BY MODULATION ======================
print(f"\n{'='*55}")
print(f"TPR BY MODULATION (P_fa = {target_pfa})")
print(f"{'Modulation':<12} {'AUC':>6}  {'TPR@γ':>8}")
print(f"{'─'*55}")

for mod in ['qpsk', 'bpsk', '16qam', '32qam']:
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0:
        continue
    fpr_s, tpr_s, _ = roc_curve(test_labels[mask], beta[mask])
    auc_s = auc(fpr_s, tpr_s)
    tpr_at_g = tpr_s[np.searchsorted(fpr_s, target_pfa, side='right') - 1]
    print(f"{mod:<12} {auc_s:>6.4f}  {tpr_at_g:>8.4f}")

# ====================== Pd vs SNR ======================
snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
pd_arr = []

print(f"\n{'='*40}")
print(f"Pd vs SNR  (P_fa = {target_pfa})")
print(f"{'SNR (dB)':>10}  {'Pablos Pd':>12}")
print(f"{'─'*40}")

for snr_db in snr_points:
    sig_mask = (test_snr == snr_db) & (test_labels == 1)
    pd_val = float(np.mean(beta[sig_mask] > gamma)) if sig_mask.sum() > 0 else np.nan
    pd_arr.append(pd_val)
    print(f"{snr_db:>10d}  {pd_val:>12.4f}")

print(f"{'─'*40}")
np.save("spectrum_data/pd_vs_snr_pablos.npy", np.array(pd_arr))
print("Saved spectrum_data/pd_vs_snr_pablos.npy")
print("✅ Evaluation complete!")
