"""Create paper and diagnostic Pd-versus-SNR figures.

The paper Psi-NN is the I-only ``pd_vs_snr_pablos.npy`` result. The similarly
named ``pd_vs_snr_psinn.npy`` file is a separate two-channel I/Q ablation.
"""

import argparse
import os
from pathlib import Path

# Avoid duplicate OpenMP runtime aborts on the project's macOS environment and
# keep Matplotlib's cache inside the writable project directory.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_labels import (
    LEGEND_CONV_AE_BASELINE,
    LEGEND_PABLOS_I_ONLY,
    LEGEND_PSL_CNN,
    LEGEND_SPECTRUM_CAE,
)


def load_optional(path: Path):
    return np.load(path) if path.exists() else None


def plot_if_compatible(ax, snr_points, values, display_mask=None, **kwargs):
    """Plot a saved curve when it matches the selected experiment grid."""
    if values is None:
        print(f"WARNING: missing curve for {kwargs.get('label')}; skipped")
        return False
    if len(values) != len(snr_points):
        print(
            f"WARNING: {kwargs.get('label')} has {len(values)} points, "
            f"but the selected grid has {len(snr_points)}; skipped"
        )
        return False
    # Keep every evaluated SNR in the saved arrays, but allow the paper figure
    # to focus on the low-SNR range used in the advisor's reference figure.
    if display_mask is None:
        display_mask = np.ones(len(snr_points), dtype=bool)
    ax.plot(snr_points[display_mask], values[display_mask], **kwargs)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Tag used by evaluate_baselines.py.")
    parser.add_argument("--L", type=int, default=4, help="CAV/MME lag dimension.")
    parser.add_argument("--noise-unc-db", type=float, default=2.0)
    parser.add_argument("--snr-min", type=float, default=-14.0,
                        help="Lowest SNR displayed in the paper figure.")
    parser.add_argument("--snr-max", type=float, default=-4.0,
                        help="Highest SNR displayed in the paper figure.")
    args = parser.parse_args()

    data_dir = Path("spectrum_data")
    suffix = f"_{args.tag}" if args.tag else ""

    # Prefer the tagged baseline grid to prevent mixing separate experiments.
    tagged_grid = data_dir / f"snr_points{suffix}.npy"
    grid_path = tagged_grid if args.tag and tagged_grid.exists() else data_dir / "snr_points.npy"
    snr_points = np.load(grid_path)
    if len(np.unique(snr_points)) != len(snr_points):
        raise ValueError(f"SNR grid contains duplicates: {grid_path}")
    paper_mask = (snr_points >= args.snr_min) & (snr_points <= args.snr_max)
    if not np.any(paper_mask):
        raise ValueError(
            f"No SNR points fall within [{args.snr_min:g}, {args.snr_max:g}] dB"
        )
    paper_snr = snr_points[paper_mask]

    # Paper models: both use I-only input on the corrected normalized test data.
    pd_cae = load_optional(data_dir / "pd_vs_snr_cae.npy")
    pd_psinn_paper = load_optional(data_dir / "pd_vs_snr_pablos.npy")

    # Classical baselines from the matching tagged raw test-data run.
    pd_cav = load_optional(data_dir / f"pd_vs_snr_cav_L{args.L}{suffix}.npy")
    pd_mme = load_optional(data_dir / f"pd_vs_snr_mme_L{args.L}{suffix}.npy")
    pd_ed = load_optional(data_dir / f"pd_vs_snr_ed{suffix}.npy")
    pd_ed_unc = load_optional(
        data_dir / f"pd_vs_snr_ed_unc{args.noise_unc_db:g}dB{suffix}.npy"
    )

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    common = dict(linewidth=1.8, markersize=5, markerfacecolor="white")
    plot_if_compatible(ax, snr_points, pd_psinn_paper, paper_mask, color="tab:red", marker="D",
                       label="Psi-NN (I-only)", **common)
    plot_if_compatible(ax, snr_points, pd_cae, paper_mask, color="tab:blue", marker="v",
                       label="CAE (I-only)", **common)
    plot_if_compatible(ax, snr_points, pd_cav, paper_mask, color="tab:purple", marker="^",
                       linestyle="--", label=f"CAV (L={args.L})", **common)
    plot_if_compatible(ax, snr_points, pd_mme, paper_mask, color="tab:green", marker="s",
                       linestyle="--", label=f"MME (L={args.L})", **common)
    plot_if_compatible(ax, snr_points, pd_ed, paper_mask, color="black", marker="o",
                       linestyle="-.", label=r"ED (known $\sigma^2$)", **common)
    plot_if_compatible(
        ax, snr_points, pd_ed_unc, paper_mask, color="0.5", marker="o", linestyle=":",
        label=rf"ED ($\pm${args.noise_unc_db:g} dB noise uncertainty)", **common
    )
    ax.set_xlabel("Nominal SNR (dB)")
    ax.set_ylabel(r"Detection probability, $P_d$")
    ax.set_title(r"$P_{\mathrm{fa}} = 0.01$, 4 samples/symbol")
    ax.set_xticks(paper_snr)
    ax.set_ylim(-0.02, 1.03)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    paper_stem = f"pd_vs_snr_paper{suffix}"
    fig.savefig(paper_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(paper_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {paper_stem}.png and {paper_stem}.pdf using {grid_path}")

    # Diagnostic figure. Legacy arrays with a different grid are skipped.
    pd_psinn_iq = load_optional(data_dir / "pd_vs_snr_psinn.npy")
    pd_conv_ae_iq = load_optional(data_dir / "pd_vs_snr_baseline.npy")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_if_compatible(ax, snr_points, pd_cae, marker="s", linewidth=2,
                       label=LEGEND_SPECTRUM_CAE)
    plot_if_compatible(ax, snr_points, pd_psinn_paper, marker="x", linewidth=2,
                       label=LEGEND_PABLOS_I_ONLY)
    plot_if_compatible(ax, snr_points, pd_psinn_iq, marker="^", linewidth=2,
                       label=LEGEND_PSL_CNN)
    plot_if_compatible(ax, snr_points, pd_conv_ae_iq, marker="D", linewidth=2,
                       label=LEGEND_CONV_AE_BASELINE)
    ax.set_xlabel("Nominal SNR (dB)")
    ax.set_ylabel(r"Detection probability, $P_d$")
    ax.set_title(r"Architecture comparison at $P_{\mathrm{fa}}=0.01$")
    ax.set_xticks(snr_points)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    diagnostic_stem = f"pd_vs_snr_architectures{suffix}"
    fig.savefig(diagnostic_stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {diagnostic_stem}.png")


if __name__ == "__main__":
    main()
