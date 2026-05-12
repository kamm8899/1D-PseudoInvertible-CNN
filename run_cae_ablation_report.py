#!/usr/bin/env python3
"""
Generate four CAE test-channel ablations, evaluate one CAE checkpoint on each,
and write a single text report plus one Pd-vs-SNR figure (four curves).

Cases (professor-style):
  1) ideal   — no pulse shaping, no SNR uncertainty
  2) uncert  — no pulse shaping, SNR uncertainty only
  3) pulse   — pulse shaping, no SNR uncertainty
  4) full    — pulse shaping + SNR uncertainty

Output:
  spectrum_data/cae_ablation_report.txt
  spectrum_data/cae_ablation_pd_vs_snr.png
  spectrum_data/cae_ablation_pd_matrix.npy   # shape (4, 11), same SNR order as snr_points

Usage (from repo root):
  python run_cae_ablation_report.py
  python run_cae_ablation_report.py --snr-uncertainty-db 2 --skip-generate
"""

from __future__ import annotations

import os

# macOS / conda: multiple OpenMP runtimes (PyTorch + NumPy/SciPy) can abort without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

from cae_spectrum import CAE
from generate_spectrum_dataset import generate_iq_dataset, pack_test_tensors_for_save


def _compute_beta(model: CAE, data: torch.Tensor, device: torch.device) -> np.ndarray:
    betas = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i : i + 128].to(device)
            recon = model(batch)
            sse = torch.sum((batch - recon) ** 2, dim=[1, 2])
            mean_x = torch.mean(batch, dim=[1, 2], keepdim=True)
            sst = torch.sum((batch - mean_x) ** 2, dim=[1, 2])
            beta = 1.0 - (sse / (sst + 1e-8))
            betas.append(beta.cpu())
    return torch.cat(betas).numpy()


def _gamma_from_train(
    model: CAE, train_noise_i: torch.Tensor, device: torch.device, target_pfa: float
) -> tuple[float, float, float]:
    train_beta = _compute_beta(model, train_noise_i, device)
    mu_e, sigma_e = float(np.mean(train_beta)), float(np.std(train_beta))
    gamma = mu_e + norm.ppf(1.0 - target_pfa) * sigma_e
    return mu_e, sigma_e, gamma


def _pd_vs_snr(
    beta_cae: np.ndarray,
    test_labels: np.ndarray,
    test_snr: np.ndarray,
    gamma: float,
    snr_points: list[int],
) -> np.ndarray:
    out = []
    for snr_db in snr_points:
        sig_mask = (test_snr == snr_db) & (test_labels == 1)
        if sig_mask.sum() == 0:
            out.append(np.nan)
        else:
            out.append(float(np.mean(beta_cae[sig_mask] > gamma)))
    return np.array(out, dtype=np.float64)


def _generate_all_cases(
    spectrum_dir: Path,
    uncertainty_db: float,
) -> list[tuple[str, str, str]]:
    """Returns list of (tag, norm_path_str, description)."""
    cases: list[tuple[str, bool, float, str]] = [
        ("ideal", False, 0.0, "Ideal: no pulse shaping, no SNR uncertainty."),
        ("uncert", False, uncertainty_db, "SNR uncertainty only (no pulse shaping)."),
        ("pulse", True, 0.0, "Pulse shaping, no SNR uncertainty."),
        ("full", True, uncertainty_db, "Pulse shaping + SNR uncertainty (full channel)."),
    ]
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    paths_out: list[tuple[str, str, str]] = []
    save_train = True

    for tag, use_pulse, u_db, desc in cases:
        train_n, train_raw, td, td_raw, labels, snrs, mods, meta = generate_iq_dataset(
            use_pulse_shaping=use_pulse,
            snr_uncertainty_db=u_db,
        )
        if save_train:
            torch.save(train_raw, spectrum_dir / "train_noise_raw.pt")
            torch.save(train_n, spectrum_dir / "train_noise.pt")
            save_train = False

        d_norm, d_raw = pack_test_tensors_for_save(td, td_raw, labels, snrs, mods, meta)
        p_norm = spectrum_dir / f"test_data_{tag}.pt"
        p_raw = spectrum_dir / f"test_data_raw_{tag}.pt"
        torch.save(d_norm, p_norm)
        torch.save(d_raw, p_raw)
        print(f"Saved {p_norm.name} / {p_raw.name}  ({desc})")
        paths_out.append((tag, str(p_norm), desc))

    return paths_out


def _evaluate_case(
    model: CAE,
    test_path: Path,
    train_noise_i: torch.Tensor,
    device: torch.device,
    target_pfa: float,
    snr_points: list[int],
) -> dict:
    test_dict = torch.load(test_path, weights_only=False)
    test_labels = test_dict["labels"].numpy()
    test_snr = test_dict["snrs"].numpy()
    test_data = test_dict["data"][:, 0:1, :]

    mu_e, sigma_e, gamma = _gamma_from_train(model, train_noise_i, device, target_pfa)
    beta_cae = _compute_beta(model, test_data, device)

    fpr, tpr, _ = roc_curve(test_labels, beta_cae)
    auc_val = auc(fpr, tpr)
    pd_arr = _pd_vs_snr(beta_cae, test_labels, test_snr, gamma, snr_points)

    return {
        "auc": auc_val,
        "gamma": gamma,
        "mu_h0": mu_e,
        "sigma_h0": sigma_e,
        "pd_vs_snr": pd_arr,
        "generation": test_dict.get("generation", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CAE ablation: generate 4 test sets + one report.")
    parser.add_argument(
        "--snr-uncertainty-db",
        type=float,
        default=2.0,
        help="Half-width (dB) for cases with SNR uncertainty (uncert + full).",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Do not regenerate tensors; expect test_data_{ideal,uncert,pulse,full}.pt already.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="spectrum_data/cae_best.pth",
        help="CAE state_dict path (relative to cwd, usually repo root).",
    )
    parser.add_argument(
        "--target-pfa",
        type=float,
        default=0.01,
        help="Neyman-Pearson P_fa for gamma and Pd vs SNR.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    spectrum_dir = repo / "spectrum_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snr_points = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    tags = ["ideal", "uncert", "pulse", "full"]
    case_desc = {
        "ideal": "Case 1 — Ideal (no pulse shaping, no SNR uncertainty).",
        "uncert": "Case 2 — SNR uncertainty only (no pulse shaping).",
        "pulse": "Case 3 — Pulse shaping, no SNR uncertainty.",
        "full": "Case 4 — Full channel (pulse shaping + SNR uncertainty).",
    }

    if not args.skip_generate:
        print(f"Generating four test bundles (SNR uncertainty half-width = {args.snr_uncertainty_db} dB where on)...\n")
        _generate_all_cases(spectrum_dir, args.snr_uncertainty_db)
        print()
    else:
        for tag in tags:
            p = spectrum_dir / f"test_data_{tag}.pt"
            if not p.is_file():
                raise FileNotFoundError(f"--skip-generate but missing {p}")

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"CAE weights not found: {weights_path.resolve()}")

    train_noise = torch.load(spectrum_dir / "train_noise.pt", weights_only=False)
    train_noise_i = train_noise[:, 0:1, :]

    model = CAE().to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))
    model.eval()

    rows_pd: list[np.ndarray] = []
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CAE ABLATION REPORT — one checkpoint, four test-channel conditions")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: {device}")
    lines.append(f"Weights: {weights_path}")
    lines.append(f"Target P_fa (gamma, Pd vs SNR): {args.target_pfa}  (upper tail H0; detect H1 when β > γ)")
    lines.append(f"SNR uncertainty half-width (cases 2 & 4): {args.snr_uncertainty_db} dB")
    lines.append("=" * 78)
    lines.append("")

    for tag in tags:
        test_path = spectrum_dir / f"test_data_{tag}.pt"
        res = _evaluate_case(model, test_path, train_noise_i, device, args.target_pfa, snr_points)
        rows_pd.append(res["pd_vs_snr"])
        gen = res["generation"]
        lines.append(case_desc[tag])
        lines.append(f"  File: {test_path.name}")
        lines.append(f"  generation: {gen}")
        lines.append(f"  H0 β: mean={res['mu_h0']:.6f}  std={res['sigma_h0']:.6f}")
        lines.append(f"  γ (P_fa={args.target_pfa}): {res['gamma']:.6f}")
        lines.append(f"  Test AUC (β, higher = signal, same as PsiNN inverted): {res['auc']:.6f}")
        lines.append("  Pd vs nominal SNR (H1, β > γ):")
        hdr = "    " + "".join(f"{s:>8}" for s in snr_points)
        lines.append(hdr)
        vals = "    " + "".join(f"{v:>8.4f}" if np.isfinite(v) else f"{'N/A':>8}" for v in res["pd_vs_snr"])
        lines.append(vals)
        lines.append("")

    pd_matrix = np.stack(rows_pd, axis=0)
    np.save(spectrum_dir / "cae_ablation_pd_matrix.npy", pd_matrix)
    labels_arr = np.array(tags, dtype=object)
    np.savez(
        spectrum_dir / "cae_ablation_meta.npz",
        tags=labels_arr,
        snr_points=np.array(snr_points, dtype=np.int32),
        snr_uncertainty_db=np.array([args.snr_uncertainty_db], dtype=np.float64),
    )

    report_path = spectrum_dir / "cae_ablation_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {report_path}")
    print(f"Wrote {spectrum_dir / 'cae_ablation_pd_matrix.npy'}  shape {pd_matrix.shape}")

    plt.figure(figsize=(9, 5.5))
    markers = ["o", "s", "^", "D"]
    for i, tag in enumerate(tags):
        plt.plot(
            snr_points,
            pd_matrix[i],
            marker=markers[i],
            linewidth=2,
            label=tag,
        )
    plt.xlabel("SNR (dB) nominal")
    plt.ylabel(r"$P_d$  ($\beta > \gamma$, H1)")
    plt.title("CAE — Pd vs SNR ablation (same model, four test channels)")
    plt.xticks(snr_points)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig_path = spectrum_dir / "cae_ablation_pd_vs_snr.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
