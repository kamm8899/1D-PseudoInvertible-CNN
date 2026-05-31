"""
Verify per-modulation AUC at -10 dB for the 1ch paper models:
  CAE  = compare_A_lowcap-original.pth  (LowCapOriginal, Model A)
  PsiNN = pablos_200epochs.pth          (AE_Pablos1d)

Prints the AUC range at -10 dB for both models so the paper text can be checked.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import torch.nn as nn
import torch.nn.functional as F

from psinn_layer_1d_pablos import AE_Pablos1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CAE model definition (LowCapOriginal = Model A) ───────────────────────────
class LowCapOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  16,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(16),  nn.LeakyReLU(0.2),
            nn.Conv1d(16, 64,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 8,  kernel_size=5, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8,   8,  kernel_size=5, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8,   16, kernel_size=5, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)
        return out


# ── Load data ─────────────────────────────────────────────────────────────────
test_dict   = torch.load("spectrum_data/test_data_full.pt", weights_only=False)
test_data   = test_dict["data"][:, 0:1, :]        # 1ch (N, 1, 1024)
test_labels = test_dict["labels"].numpy()
test_snr    = test_dict["snrs"].numpy()
test_mods   = np.array(test_dict["signals"])

train_noise = torch.load("spectrum_data/train_noise.pt", weights_only=False)
train_1ch   = train_noise[:, 0:1, :]

# ── Load models ───────────────────────────────────────────────────────────────
cae = LowCapOriginal().to(device)
cae.load_state_dict(torch.load("spectrum_data/compare_A_lowcap-original.pth",
                                weights_only=False, map_location=device))
cae.eval()

psinn = AE_Pablos1d(nf=16, k=5, use_dropout=True).to(device)
psinn.load_state_dict(torch.load("spectrum_data/pablos_200epochs.pth",
                                  weights_only=False, map_location=device))
psinn.eval()


# ── Beta computation ──────────────────────────────────────────────────────────
def compute_beta(model, data, use_ae=False):
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 256):
            b = data[i:i+256].to(device)
            recon = model.AE(b) if use_ae else model(b)
            if recon.shape[-1] != b.shape[-1]:
                recon = recon[..., :b.shape[-1]]
            sse   = torch.sum((b - recon) ** 2, dim=[1, 2])
            mean_x = torch.mean(b, dim=[1, 2], keepdim=True)
            sst   = torch.sum((b - mean_x) ** 2, dim=[1, 2])
            betas.append((1.0 - sse / (sst + 1e-8)).cpu())
    return torch.cat(betas).numpy()


# ── Thresholds from training noise ────────────────────────────────────────────
beta_cae_train   = compute_beta(cae,   train_1ch, use_ae=False)
beta_psinn_train = compute_beta(psinn, train_1ch, use_ae=True)

pfa = 0.01
gamma_cae   = np.mean(beta_cae_train)   + norm.ppf(1 - pfa) * np.std(beta_cae_train)
gamma_psinn = np.mean(beta_psinn_train) + norm.ppf(1 - pfa) * np.std(beta_psinn_train)
print(f"CAE   γ = {gamma_cae:.4f}")
print(f"PsiNN γ = {gamma_psinn:.4f}")

# ── Test beta scores ──────────────────────────────────────────────────────────
beta_cae   = compute_beta(cae,   test_data, use_ae=False)
beta_psinn = compute_beta(psinn, test_data, use_ae=True)

# ── Per-modulation AUC at -10 dB ─────────────────────────────────────────────
SNR_TARGET = -10
snr_mask   = (test_snr == SNR_TARGET)

modulations = ['bpsk', 'qpsk', '16qam', '32qam']

print(f"\nPer-modulation AUC at SNR = {SNR_TARGET} dB  (Pfa = {pfa})")
print(f"{'Modulation':<10}  {'CAE AUC':>9}  {'PsiNN AUC':>10}")
print("─" * 35)

cae_aucs, psinn_aucs = [], []

for mod in modulations:
    mask = snr_mask & ((test_mods == mod) | (test_labels == 0))
    if mask.sum() == 0 or len(np.unique(test_labels[mask])) < 2:
        print(f"{mod:<10}   N/A         N/A")
        continue
    fpr_c, tpr_c, _ = roc_curve(test_labels[mask], beta_cae[mask])
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psinn[mask])
    ac = auc(fpr_c, tpr_c)
    ap = auc(fpr_p, tpr_p)
    cae_aucs.append(ac)
    psinn_aucs.append(ap)
    print(f"{mod:<10}  {ac:>9.3f}  {ap:>10.3f}")

print("─" * 35)
print(f"{'Range':<10}  {min(cae_aucs):.3f}--{max(cae_aucs):.3f}  "
      f"{min(psinn_aucs):.3f}--{max(psinn_aucs):.3f}")

# ── Also print pooled modulation AUC (all SNR) for cross-check with Table 2 ──
print(f"\nPooled per-modulation AUC (all SNR) — cross-check with Table 2")
print(f"{'Modulation':<10}  {'CAE AUC':>9}  {'PsiNN AUC':>10}")
print("─" * 35)
for mod in modulations:
    mask = (test_mods == mod) | (test_labels == 0)
    if mask.sum() == 0 or len(np.unique(test_labels[mask])) < 2:
        continue
    fpr_c, tpr_c, _ = roc_curve(test_labels[mask], beta_cae[mask])
    fpr_p, tpr_p, _ = roc_curve(test_labels[mask], beta_psinn[mask])
    print(f"{mod:<10}  {auc(fpr_c, tpr_c):>9.3f}  {auc(fpr_p, tpr_p):>10.3f}")
