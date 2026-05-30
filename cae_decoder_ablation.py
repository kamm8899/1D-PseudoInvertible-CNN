"""
Ablation: LowCap-CAE1ch decoder bottleneck vs symmetric decoder.

Compares three variants — all share the same 3-layer encoder (1->16->64->128):
  original   : decoder 128->8->8->16->1  (extreme bottleneck, Pablos et al.)
  symmetric  : decoder 128->64->16->1   (mirrors encoder channel widths)
  mid        : decoder 128->32->16->1   (intermediate bottleneck)

Run: python cae_decoder_ablation.py
Outputs:
  spectrum_data/cae_dec_original.pth
  spectrum_data/cae_dec_symmetric.pth
  spectrum_data/cae_dec_mid.pth
  spectrum_data/cae_decoder_ablation_report.txt
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm


# ── Shared encoder (same as LowCap-CAE1ch) ───────────────────────────────────
def make_encoder():
    return nn.Sequential(
        nn.Conv1d(1,  16,  kernel_size=5, stride=2, padding=1),
        nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
        nn.Conv1d(16, 64,  kernel_size=5, stride=2, padding=1),
        nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
        nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=1),
        nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
    )


# ── Three decoder variants ────────────────────────────────────────────────────
def make_decoder_original():
    """LowCap-CAE1ch as-is: extreme 128->8 bottleneck."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv1d(128, 8,  kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(8,   8,  kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(8,   16, kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
    )


def make_decoder_symmetric():
    """Mirrors the encoder channel widths: 128->64->16->1."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(64,  16, kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
    )


def make_decoder_mid():
    """Intermediate bottleneck: 128->32->16->1."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv1d(128, 32, kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(32,  16, kernel_size=5, stride=1, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
    )


class CAEVariant(nn.Module):
    def __init__(self, decoder_fn):
        super().__init__()
        self.encoder = make_encoder()
        self.decoder = decoder_fn()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear',
                                align_corners=False)
        return out


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.02):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_val, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        for (x,) in train_loader:
            x = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                val_loss += criterion(model(x.to(device)), x.to(device)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  val_loss={val_loss:.6f}")
    model.load_state_dict(best_state)
    return best_val


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, train_noise, test_data, test_labels, test_snrs, device,
             pfa=0.01):
    def compute_beta(data):
        betas = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(data), 256):
                b = data[i:i+256].to(device)
                r = model(b)
                sse = torch.sum((b - r)**2, dim=[1, 2])
                sst = torch.sum(
                    (b - b.mean(dim=[1, 2], keepdim=True))**2, dim=[1, 2])
                betas.append((1 - sse / (sst + 1e-8)).cpu())
        return torch.cat(betas).numpy()

    b_train = compute_beta(train_noise)
    mu, sigma = np.mean(b_train), np.std(b_train)
    gamma = mu + norm.ppf(1 - pfa) * sigma

    b_test = compute_beta(test_data)
    fpr, tpr, _ = roc_curve(test_labels, b_test)
    auc_score = auc(fpr, tpr)

    snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    pd_arr = []
    for s in snr_points:
        mask = (test_snrs == s) & (test_labels == 1)
        pd_arr.append(
            float(np.mean(b_test[mask] > gamma)) if mask.sum() else np.nan)

    return {
        'mu': mu, 'sigma': sigma, 'gamma': gamma,
        'auc': auc_score, 'pd': pd_arr,
        'params': sum(p.numel() for p in model.parameters()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    train_2ch = torch.load('spectrum_data/train_noise.pt', weights_only=False)
    train_1ch = train_2ch[:, 0:1, :]

    n_val = int(0.1 * len(train_1ch))
    train_set, val_set = random_split(
        TensorDataset(train_1ch), [len(train_1ch) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

    test_dict   = torch.load('spectrum_data/test_data_full.pt',
                             weights_only=False)
    test_data   = test_dict['data'][:, 0:1, :]
    test_labels = test_dict['labels'].numpy()
    test_snrs   = test_dict['snrs'].numpy()

    variants = {
        'original':  make_decoder_original,
        'mid':       make_decoder_mid,
        'symmetric': make_decoder_symmetric,
    }

    results = {}
    for name, dec_fn in variants.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        model = CAEVariant(dec_fn).to(device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        train_model(model, train_loader, val_loader, device, epochs=50, lr=0.02)
        torch.save(model.state_dict(), f'spectrum_data/cae_dec_{name}.pth')
        res = evaluate(model, train_1ch, test_data, test_labels, test_snrs,
                       device)
        results[name] = res
        print(f"  AUC={res['auc']:.4f}  gamma={res['gamma']:.4f}  "
              f"params={res['params']:,}")

    # ── Report ────────────────────────────────────────────────────────────────
    snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    lines = [
        '=' * 70,
        'CAE DECODER BOTTLENECK ABLATION',
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        '=' * 70, '',
        f"{'Variant':<12} {'Params':>8} {'AUC':>7} "
        f"{'gamma':>7} {'mu_H0':>7} {'sigma_H0':>8}",
        '-' * 58,
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<12} {r['params']:>8,} {r['auc']:>7.4f} "
            f"{r['gamma']:>7.4f} {r['mu']:>7.4f} {r['sigma']:>8.4f}")

    lines += ['', 'Pd vs SNR (Pfa=0.01, upper-tail):',
              f"{'SNR':>5}" + ''.join(f" {n:>12}" for n in results),
              '-' * (5 + 13 * len(results))]
    for i, s in enumerate(snr_points):
        row = f"{s:>5}"
        for r in results.values():
            v = r['pd'][i]
            row += f" {v:>12.4f}" if np.isfinite(v) else f" {'N/A':>12}"
        lines.append(row)

    report = '\n'.join(lines)
    print('\n' + report)
    Path('spectrum_data/cae_decoder_ablation_report.txt').write_text(
        report + '\n')
    print('\nSaved spectrum_data/cae_decoder_ablation_report.txt')
