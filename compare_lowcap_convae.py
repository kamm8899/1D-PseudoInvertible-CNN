"""
Direct comparison of LowCap-CAE1ch vs ConvAE-Baseline2ch — isolating each
architectural difference one step at a time.

Five variants tested in a chain, each changing exactly one thing from the
previous:

  A  LowCap-original    1ch  enc:1→16→64→128  dec:bottleneck  SGD/50ep
  B  LowCap-Adam        1ch  enc:1→16→64→128  dec:bottleneck  Adam/200ep
  C  LowCap-SymDec      1ch  enc:1→16→64→128  dec:symmetric   Adam/200ep
  D  ConvAE-1ch         1ch  enc:1→16→32→64→128 dec:symmetric Adam/200ep
  E  ConvAE-2ch         2ch  enc:2→16→32→64→128 dec:symmetric Adam/200ep

A→B : optimizer only
B→C : decoder structure only
C→D : encoder depth + channel progression only
D→E : input channels only (1ch vs 2ch)

Run: python compare_lowcap_convae.py
Outputs:
  spectrum_data/compare_A_lowcap_original.pth
  spectrum_data/compare_B_lowcap_adam.pth
  spectrum_data/compare_C_lowcap_symdec.pth
  spectrum_data/compare_D_convae_1ch.pth
  spectrum_data/compare_E_convae_2ch.pth
  spectrum_data/compare_lowcap_convae_report.txt
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


# ── Model definitions ─────────────────────────────────────────────────────────

class LowCapOriginal(nn.Module):
    """A — LowCap-CAE1ch exactly as in cae_spectrum.py."""
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
            out = F.interpolate(out, size=x.shape[-1], mode='linear',
                                align_corners=False)
        return out


class LowCapSymDec(nn.Module):
    """B/C — same 3-layer encoder as LowCap, symmetric decoder (mirrors encoder)."""
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
        # Symmetric: 128→64→16→1 mirrors 1→16→64→128
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64,  16, kernel_size=5, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16,  1,  kernel_size=5, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear',
                                align_corners=False)
        return out


class ConvAE(nn.Module):
    """D/E — ConvAE architecture with configurable input channels.
       4-layer encoder (gradual: ch→16→32→64→128), symmetric decoder.
       Set in_ch=1 for variant D, in_ch=2 for variant E (original ConvAE).
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 16,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(16),  nn.LeakyReLU(0.2),
            nn.Conv1d(16,   32,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.Conv1d(32,   64,  kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Conv1d(64,   128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        # Symmetric decoder: 128→64→32→16→in_ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64,   kernel_size=5, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64,  32,   kernel_size=5, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32,  16,   kernel_size=5, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm1d(16),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16,  in_ch, kernel_size=5, stride=2, padding=1,
                               output_padding=1),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear',
                                align_corners=False)
        return out


# ── Training helpers ──────────────────────────────────────────────────────────

def train_sgd(model, train_loader, val_loader, device, epochs=50, lr=0.02):
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
                val_loss += criterion(model(x.to(device)),
                                      x.to(device)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  val={val_loss:.6f}")
    model.load_state_dict(best_state)


def train_adam(model, train_loader, val_loader, device, epochs=200, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                 gamma=0.5)
    best_val, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        for (x,) in train_loader:
            x = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                val_loss += criterion(model(x.to(device)),
                                      x.to(device)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  val={val_loss:.6f}")
    model.load_state_dict(best_state)


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
    mu, sigma = float(np.mean(b_train)), float(np.std(b_train))
    gamma = mu + norm.ppf(1 - pfa) * sigma

    b_test = compute_beta(test_data)
    fpr, tpr, _ = roc_curve(test_labels, b_test)
    auc_score = auc(fpr, tpr)

    snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    pd_arr = []
    for s in snr_points:
        mask = (test_snrs == s) & (test_labels == 1)
        pd_arr.append(float(np.mean(b_test[mask] > gamma))
                      if mask.sum() > 0 else np.nan)

    return {
        'mu': mu, 'sigma': sigma, 'gamma': gamma,
        'auc': auc_score, 'pd': pd_arr,
        'params': sum(p.numel() for p in model.parameters()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

import argparse
import matplotlib.pyplot as plt


def _plot_ablation(results, snr_points, out_path='ablation_pd_vs_snr.png'):
    markers = ['o', 's', '^', 'D', 'x']
    plt.figure(figsize=(8, 5.5))
    for (tag, r), m in zip(results.items(), markers):
        short = tag
        plt.plot(snr_points, r['pd'], marker=m, linewidth=2, label=f'Variant {short}')
    plt.xlabel('SNR (dB)')
    plt.ylabel(r'$P_d$  ($\beta > \gamma$,  $H_1$)')
    plt.title(r'Ablation: $P_d$ vs SNR — A→E architectural changes  ($P_{\rm fa}=0.01$)')
    plt.xticks(snr_points)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Load saved weights and regenerate ablation_pd_vs_snr.png without retraining')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    if args.plot_only:
        # Load data and evaluate saved checkpoints without retraining
        train_2ch = torch.load('spectrum_data/train_noise.pt', weights_only=False)
        train_1ch = train_2ch[:, 0:1, :]
        test_dict   = torch.load('spectrum_data/test_data_full.pt', weights_only=False)
        test_2ch    = test_dict['data']
        test_1ch    = test_2ch[:, 0:1, :]
        test_labels = test_dict['labels'].numpy()
        test_snrs   = test_dict['snrs'].numpy()

        ckpt_map = {
            'A': ('spectrum_data/compare_A_lowcap-original.pth', LowCapOriginal(), train_1ch, test_1ch),
            'B': ('spectrum_data/compare_B_lowcap-adam.pth',     LowCapOriginal(), train_1ch, test_1ch),
            'C': ('spectrum_data/compare_C_lowcap-symdec.pth',   LowCapSymDec(),   train_1ch, test_1ch),
            'D': ('spectrum_data/compare_D_convae-1ch.pth',      ConvAE(in_ch=1),  train_1ch, test_1ch),
            'E': ('spectrum_data/compare_E_convae-2ch.pth',      ConvAE(in_ch=2),  train_2ch, test_2ch),
        }
        snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        results_plot = {}
        for tag, (ckpt, model, tr_noise, te_data) in ckpt_map.items():
            model.load_state_dict(torch.load(ckpt, weights_only=False, map_location=device))
            model = model.to(device)
            res = evaluate(model, tr_noise, te_data, test_labels, test_snrs, device)
            results_plot[tag] = res
            print(f"Variant {tag}: AUC={res['auc']:.4f}  gamma={res['gamma']:.4f}")
        _plot_ablation(results_plot, snr_points)
        import sys; sys.exit(0)

    # Data
    train_2ch = torch.load('spectrum_data/train_noise.pt', weights_only=False)
    train_1ch = train_2ch[:, 0:1, :]   # I channel only
    # train_2ch used as-is for variant E

    def make_loaders(data):
        n_val = int(0.1 * len(data))
        tr, va = random_split(TensorDataset(data), [len(data) - n_val, n_val])
        return (DataLoader(tr, batch_size=256, shuffle=True),
                DataLoader(va, batch_size=256, shuffle=False))

    loader_1ch = make_loaders(train_1ch)
    loader_2ch = make_loaders(train_2ch)

    test_dict   = torch.load('spectrum_data/test_data_full.pt',
                             weights_only=False)
    test_2ch    = test_dict['data']
    test_1ch    = test_2ch[:, 0:1, :]
    test_labels = test_dict['labels'].numpy()
    test_snrs   = test_dict['snrs'].numpy()

    # Variants: (label, description, model, train_fn, train_noise, test_data)
    variants = [
        ('A', 'LowCap-original   1ch enc:1→16→64→128 dec:bottleneck SGD/50ep',
         LowCapOriginal(),  train_sgd,  loader_1ch, train_1ch, test_1ch),
        ('B', 'LowCap-Adam       1ch enc:1→16→64→128 dec:bottleneck Adam/200ep',
         LowCapOriginal(),  train_adam, loader_1ch, train_1ch, test_1ch),
        ('C', 'LowCap-SymDec     1ch enc:1→16→64→128 dec:symmetric  Adam/200ep',
         LowCapSymDec(),    train_adam, loader_1ch, train_1ch, test_1ch),
        ('D', 'ConvAE-1ch        1ch enc:1→16→32→64→128 dec:sym     Adam/200ep',
         ConvAE(in_ch=1),  train_adam, loader_1ch, train_1ch, test_1ch),
        ('E', 'ConvAE-2ch        2ch enc:2→16→32→64→128 dec:sym     Adam/200ep',
         ConvAE(in_ch=2),  train_adam, loader_2ch, train_2ch, test_2ch),
    ]

    results = {}
    for tag, desc, model, train_fn, loaders, tr_noise, te_data in variants:
        print(f"\n{'='*60}")
        print(f"Variant {tag}: {desc}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        model = model.to(device)
        train_fn(model, loaders[0], loaders[1], device)
        torch.save(model.state_dict(),
                   f'spectrum_data/compare_{tag}_{desc.split()[0].lower()}.pth')
        res = evaluate(model, tr_noise, te_data, test_labels, test_snrs, device)
        res['desc'] = desc
        results[tag] = res
        print(f"  AUC={res['auc']:.4f}  gamma={res['gamma']:.4f}  "
              f"params={res['params']:,}")

    # ── Report ────────────────────────────────────────────────────────────────
    snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

    lines = [
        '=' * 78,
        'LOWCAP vs CONVAE — STEP-BY-STEP ARCHITECTURAL COMPARISON',
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        '=' * 78,
        '',
        'Each row changes exactly one thing from the row above:',
        '  A→B : optimizer  (SGD/50ep  → Adam/200ep)',
        '  B→C : decoder    (bottleneck → symmetric)',
        '  C→D : encoder    (3-layer 1→16→64→128 → 4-layer 1→16→32→64→128)',
        '  D→E : channels   (1ch → 2ch)',
        '',
        f"{'Tag':<4} {'Params':>8} {'AUC':>7} {'gamma':>7} "
        f"{'mu_H0':>7} {'sigma_H0':>9}  Description",
        '-' * 78,
    ]
    for tag, r in results.items():
        lines.append(
            f"{tag:<4} {r['params']:>8,} {r['auc']:>7.4f} {r['gamma']:>7.4f} "
            f"{r['mu']:>7.4f} {r['sigma']:>9.4f}  {r['desc']}")

    lines += [
        '',
        'Pd vs SNR (Pfa=0.01, upper-tail):',
        f"{'SNR':>5}" + ''.join(f" {t:>8}" for t in results),
        '-' * (5 + 9 * len(results)),
    ]
    for i, s in enumerate(snr_points):
        row = f"{s:>5}"
        for r in results.values():
            v = r['pd'][i]
            row += f" {v:>8.4f}" if np.isfinite(v) else f" {'N/A':>8}"
        lines.append(row)

    report = '\n'.join(lines)
    print('\n' + report)
    Path('spectrum_data/compare_lowcap_convae_report.txt').write_text(
        report + '\n')
    print('\nSaved spectrum_data/compare_lowcap_convae_report.txt')

    _plot_ablation(results, snr_points)
