"""
Channel ablation: compare 1-channel (I only) vs 2-channel (I/Q) for Psl-CNN and baseline AE.

Uses the same scoring convention as evaluate_anomaly_inverted.py:
  1D Psl-CNN: upper-tail NP + ROC on +beta;  Pd = P(beta > gamma | H1)
  Conv AE baseline: lower-tail NP + ROC on -beta; Pd = P(beta < gamma | H1)

Requires 2-channel weights (existing):
  spectrum_data/psl_cnn_{E}epochs.pth
  spectrum_data/baseline_{E}epochs.pth

Requires 1-channel weights (train first):
  python spectrum_data/train_models_1channel.py --epochs 200
  -> spectrum_data/psl_cnn_{E}epochs_ch1.pth
     spectrum_data/baseline_{E}epochs_ch1.pth

Outputs (for side-by-side with spreadsheets / paper):
  spectrum_data/channel_ablation_results.csv
  spectrum_data/channel_ablation_results.png
  spectrum_data/channel_ablation_results_pd_vs_snr.png

Usage:
  python spectrum_data/evaluate_channel_ablation.py
  python spectrum_data/evaluate_channel_ablation.py --epochs 200 --out-prefix spectrum_data/channel_ablation_results
"""

from __future__ import annotations

import os

# macOS / conda: multiple OpenMP runtimes — set before numpy/torch/matplotlib.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from spectrum_paths import assert_psinn_full_channel_metadata, get_psinn_test_data_path

from experiment_labels import ABLATION_CONV_AE, ABLATION_PSL_CNN

from psinn_layer_1d import AE_Classifier1d, AE_Baseline_Classifier1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SNR_POINTS = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype=float)


def slice_channels(x: torch.Tensor, n_ch: int) -> torch.Tensor:
    if n_ch == 2:
        return x
    if n_ch == 1:
        return x[:, 0:1, :].contiguous()
    raise ValueError("n_ch must be 1 or 2")


def compute_beta(model, data: torch.Tensor) -> np.ndarray:
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i : i + 128].to(device)
            recon = model.AE(batch)
            if recon.shape[-1] != batch.shape[-1]:
                recon = recon[..., : batch.shape[-1]]
            sse = torch.sum((batch - recon) ** 2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst = torch.sum((batch - mean_x) ** 2, dim=[1, 2])
            beta = 1.0 - (sse / (sst + 1e-8))
            betas.append(beta.cpu())
    return torch.cat(betas).numpy()


def eval_psinn_upper(
    model, train_noise: torch.Tensor, test_data: torch.Tensor, test_labels: np.ndarray, test_snr: np.ndarray, target_pfa: float
):
    train_b = compute_beta(model, train_noise)
    mu, sig = float(np.mean(train_b)), float(np.std(train_b))
    gamma = mu + norm.ppf(1.0 - target_pfa) * sig
    beta = compute_beta(model, test_data)
    fpr, tpr, _ = roc_curve(test_labels, beta)
    a = auc(fpr, tpr)
    pd_arr = []
    for snr_db in SNR_POINTS:
        sig_mask = (test_snr == snr_db) & (test_labels == 1)
        if sig_mask.sum() == 0:
            pd_arr.append(float("nan"))
        else:
            pd_arr.append(float(np.mean(beta[sig_mask] > gamma)))
    return {
        "mu_h0": mu,
        "sigma_h0": sig,
        "gamma": gamma,
        "auc": a,
        "beta": beta,
        "pd_vs_snr": np.array(pd_arr, dtype=float),
    }


def eval_baseline_lower(
    model, train_noise: torch.Tensor, test_data: torch.Tensor, test_labels: np.ndarray, test_snr: np.ndarray, target_pfa: float
):
    train_b = compute_beta(model, train_noise)
    mu, sig = float(np.mean(train_b)), float(np.std(train_b))
    gamma = mu + norm.ppf(target_pfa) * sig
    beta = compute_beta(model, test_data)
    fpr, tpr, _ = roc_curve(test_labels, -beta)
    a = auc(fpr, tpr)
    pd_arr = []
    for snr_db in SNR_POINTS:
        sig_mask = (test_snr == snr_db) & (test_labels == 1)
        if sig_mask.sum() == 0:
            pd_arr.append(float("nan"))
        else:
            pd_arr.append(float(np.mean(beta[sig_mask] < gamma)))
    return {
        "mu_h0": mu,
        "sigma_h0": sig,
        "gamma": gamma,
        "auc": a,
        "beta": beta,
        "pd_vs_snr": np.array(pd_arr, dtype=float),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200, help="Checkpoint suffix epochs (must match filenames).")
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="spectrum_data/channel_ablation_results",
        help="Prefix for .csv and .png (no extension).",
    )
    ap.add_argument("--pfa", type=float, default=0.01)
    args = ap.parse_args()

    E = args.epochs
    paths_2ch = {
        "psi": f"spectrum_data/psl_cnn_{E}epochs.pth",
        "base": f"spectrum_data/baseline_{E}epochs.pth",
    }
    paths_1ch = {
        "psi": f"spectrum_data/psl_cnn_{E}epochs_ch1.pth",
        "base": f"spectrum_data/baseline_{E}epochs_ch1.pth",
    }

    test_path = get_psinn_test_data_path()
    test_dict = torch.load(test_path, weights_only=False)
    assert_psinn_full_channel_metadata(test_dict)
    test_full = test_dict["data"]
    test_labels = test_dict["labels"].numpy()
    test_snr = test_dict["snrs"].numpy()

    train_full = torch.load("spectrum_data/train_noise.pt", weights_only=False)

    rows = []
    pd_curves: dict[str, np.ndarray] = {}

    csv_fields = [
        "model",
        "n_channels",
        "tail",
        "checkpoint",
        "status",
        "auc",
        "gamma",
        "mu_h0",
        "sigma_h0",
        "Pd_-10dB",
        "Pd_-6dB",
        "Pd_0dB",
        "Pd_+2dB",
        "Pd_+10dB",
    ]

    def run_one(label: str, n_ch: int, kind: str, ckpt: str) -> None:
        if not os.path.isfile(ckpt):
            print(f"SKIP {label} (n_ch={n_ch}): missing checkpoint {ckpt!r}")
            rows.append({k: "" for k in csv_fields} | {"model": label, "n_channels": n_ch, "checkpoint": ckpt, "status": "missing"})
            return
        data_t = slice_channels(test_full, n_ch)
        noise_t = slice_channels(train_full, n_ch)
        if kind == "psi":
            m = AE_Classifier1d(n_channels=n_ch, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
            m.eval()
            r = eval_psinn_upper(m, noise_t, data_t, test_labels, test_snr, args.pfa)
            tail = "upper"
        else:
            m = AE_Baseline_Classifier1d(n_channels=n_ch, n_classes=1, nf=16, k=5, use_dropout=True).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
            m.eval()
            r = eval_baseline_lower(m, noise_t, data_t, test_labels, test_snr, args.pfa)
            tail = "lower"
        key = f"{label}_{n_ch}ch"
        pd_curves[key] = r["pd_vs_snr"]
        rows.append(
            {
                "model": label,
                "n_channels": n_ch,
                "tail": tail,
                "checkpoint": ckpt,
                "status": "ok",
                "auc": f"{r['auc']:.4f}",
                "gamma": f"{r['gamma']:.6f}",
                "mu_h0": f"{r['mu_h0']:.6f}",
                "sigma_h0": f"{r['sigma_h0']:.6f}",
                "Pd_-10dB": f"{r['pd_vs_snr'][0]:.4f}",
                "Pd_-6dB": f"{r['pd_vs_snr'][2]:.4f}",
                "Pd_0dB": f"{r['pd_vs_snr'][5]:.4f}",
                "Pd_+2dB": f"{r['pd_vs_snr'][6]:.4f}",
                "Pd_+10dB": f"{r['pd_vs_snr'][10]:.4f}",
            }
        )
        nparams = sum(p.numel() for p in m.parameters())
        print(f"{label:12s} n_ch={n_ch}  AUC={r['auc']:.4f}  gamma={r['gamma']:.4f}  params={nparams:,}  tail={tail}")

    print(f"Test data: {test_path}  shape={tuple(test_full.shape)}  Pfa={args.pfa}\n")
    run_one(ABLATION_PSL_CNN, 2, "psi", paths_2ch["psi"])
    run_one(ABLATION_PSL_CNN, 1, "psi", paths_1ch["psi"])
    run_one(ABLATION_CONV_AE, 2, "base", paths_2ch["base"])
    run_one(ABLATION_CONV_AE, 1, "base", paths_1ch["base"])

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Side-by-side bar comparison (AUC + key Pd), only ok rows
    ok = [r for r in rows if r.get("status") == "ok"]
    if len(ok) < 2:
        print("Not enough successful runs for comparison figure (need at least two checkpoints).")
        return

    labels = [f"{r['model']}\n{r['n_channels']} ch" for r in ok]
    aucs = [float(r["auc"]) for r in ok]
    pd10 = [float(r["Pd_-10dB"]) for r in ok]
    pd0 = [float(r["Pd_0dB"]) for r in ok]
    pd2 = [float(r["Pd_+2dB"]) for r in ok]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x = np.arange(len(labels))
    w = 0.6
    for ax, vals, title in zip(
        axes,
        [aucs, pd10, pd0, pd2],
        ["AUC", r"$P_d$ @ $-10$ dB", r"$P_d$ @ $0$ dB", r"$P_d$ @ $+2$ dB"],
    ):
        ax.bar(x, vals, width=w, color=["#c44e52", "#dd8452", "#55a868", "#4c72b0"][: len(vals)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        rf"Channel ablation ($P_{{\mathrm{{fa}}}}={args.pfa}$; {ABLATION_PSL_CNN} upper-tail, {ABLATION_CONV_AE} lower-tail)"
    )
    fig.tight_layout()
    png_path = prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")

    # Overlay Pd vs SNR for all successful runs
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, r in enumerate(ok):
        key = f"{r['model']}_{r['n_channels']}ch"
        if key not in pd_curves:
            continue
        ax2.plot(
            SNR_POINTS,
            pd_curves[key],
            marker="o",
            label=key.replace("_", " "),
            color=color_cycle[i % len(color_cycle)],
        )
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel(r"$P_d$")
    ax2.set_title(r"$P_d$ vs SNR — channel ablation")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    pd_path = Path(str(prefix) + "_pd_vs_snr.png")
    fig2.savefig(pd_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {pd_path}")


if __name__ == "__main__":
    main()
