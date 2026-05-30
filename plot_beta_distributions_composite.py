"""
Generate composite beta-score distribution figure at SNR = -6 dB for all four models.
Output: snr_distributions_all_-6dB.png

Run: python plot_beta_distributions_composite.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d
from cae_spectrum import CAE
from psinn_layer_1d_pablos import AE_Pablos1d

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SNR = -6
TARGET_PFA = 0.01

# ── Load test data ──────────────────────────────────────────────────────────
for path in ["spectrum_data/test_data_full.pt", "spectrum_data/test_data.pt"]:
    if Path(path).exists():
        test_dict = torch.load(path, weights_only=False)
        print(f"Loaded {path}")
        break

test_2ch  = test_dict["data"]           # (N, 2, 1024)
test_1ch  = test_2ch[:, 0:1, :]        # (N, 1, 1024)
labels    = test_dict["labels"].numpy()
snrs      = test_dict["snrs"].numpy()

train_2ch = torch.load("spectrum_data/train_noise.pt", weights_only=False)
train_1ch = train_2ch[:, 0:1, :]

snr_mask = (snrs == TARGET_SNR)
h0 = snr_mask & (labels == 0)
h1 = snr_mask & (labels == 1)
print(f"SNR={TARGET_SNR:+d} dB — H0 samples: {h0.sum()}, H1 samples: {h1.sum()}")

# ── Load models ─────────────────────────────────────────────────────────────
model_psi = AE_Classifier1d(n_channels=2, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
model_psi.load_state_dict(torch.load("spectrum_data/psl_cnn_200epochs.pth", weights_only=False))
model_psi.eval()

model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
model_base.load_state_dict(torch.load("spectrum_data/baseline_200epochs.pth", weights_only=False))
model_base.eval()

model_cae = CAE().to(device)
model_cae.load_state_dict(torch.load("spectrum_data/cae_best.pth", weights_only=False))
model_cae.eval()

model_pablos = AE_Pablos1d(nf=16, k=5, use_dropout=True).to(device)
model_pablos.load_state_dict(torch.load("spectrum_data/pablos_200epochs.pth", weights_only=False))
model_pablos.eval()

# ── β computation ───────────────────────────────────────────────────────────
def compute_beta(model, data, use_ae=True):
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i:i+128].to(device)
            recon = model.AE(batch) if use_ae else model(batch)
            if recon.shape[-1] != batch.shape[-1]:
                recon = recon[..., :batch.shape[-1]]
            sse    = torch.sum((batch - recon) ** 2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst    = torch.sum((batch - mean_x) ** 2, dim=[1, 2])
            betas.append((1.0 - sse / (sst + 1e-8)).cpu())
    return torch.cat(betas).numpy()

# ── Thresholds from training noise ──────────────────────────────────────────
print("Computing thresholds from training noise...")
b_tr_psi  = compute_beta(model_psi,    train_2ch, use_ae=True)
b_tr_base = compute_beta(model_base,   train_2ch, use_ae=True)
b_tr_cae  = compute_beta(model_cae,    train_1ch, use_ae=False)
b_tr_pab  = compute_beta(model_pablos, train_1ch, use_ae=True)

# Psl-CAE2ch, LowCap-CAE1ch, Psl-CAE1ch → upper-tail
# ConvAE-Baseline2ch → lower-tail
gamma_psi  = np.mean(b_tr_psi)  + norm.ppf(1 - TARGET_PFA) * np.std(b_tr_psi)
gamma_base = np.mean(b_tr_base) + norm.ppf(TARGET_PFA)      * np.std(b_tr_base)
gamma_cae  = np.mean(b_tr_cae)  + norm.ppf(1 - TARGET_PFA) * np.std(b_tr_cae)
gamma_pab  = np.mean(b_tr_pab)  + norm.ppf(1 - TARGET_PFA) * np.std(b_tr_pab)

print(f"  Psl-CAE2ch  γ (upper) = {gamma_psi:.4f}")
print(f"  ConvAE-Base γ (lower) = {gamma_base:.4f}")
print(f"  LowCap-CAE1 γ (upper) = {gamma_cae:.4f}")
print(f"  Psl-CAE1ch  γ (upper) = {gamma_pab:.4f}")

# ── β on test set ────────────────────────────────────────────────────────────
print("Computing β on test set...")
b_psi  = compute_beta(model_psi,    test_2ch, use_ae=True)
b_base = compute_beta(model_base,   test_2ch, use_ae=True)
b_cae  = compute_beta(model_cae,    test_1ch, use_ae=False)
b_pab  = compute_beta(model_pablos, test_1ch, use_ae=True)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

configs = [
    ("Psl-CAE2ch",         b_psi,  gamma_psi,  "upper-tail"),
    ("ConvAE-Baseline2ch", b_base, gamma_base, "lower-tail"),
    ("LowCap-CAE1ch",      b_cae,  gamma_cae,  "upper-tail"),
    ("Psl-CAE1ch",         b_pab,  gamma_pab,  "upper-tail"),
]

for ax, (name, beta, gamma, tail) in zip(axes, configs):
    ax.hist(beta[h0], bins=40, alpha=0.65, color='steelblue',
            label='$H_0$ (noise)', density=True)
    ax.hist(beta[h1], bins=40, alpha=0.65, color='darkorange',
            label='$H_1$ (signal)', density=True)
    ax.axvline(gamma, color='red', linestyle='--', linewidth=1.5,
               label=f'$\\gamma$  ($P_{{\\rm fa}}={TARGET_PFA}$)')
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel(r'$\beta$ score', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=6.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, -0.20, f'({tail})', transform=ax.transAxes,
            ha='center', fontsize=7, style='italic', color='gray')

fig.suptitle(
    rf'$\beta$-score distributions at SNR $= {TARGET_SNR:+d}$\,dB'
    rf'  ($P_{{\rm fa}} = {TARGET_PFA}$)',
    fontsize=11
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig("snr_distributions_all_-6dB.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved snr_distributions_all_-6dB.png")
