"""Plot the paper's 4-sps and 8-sps Pd curves as adjacent panels."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


D = Path("spectrum_data")
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DISPLAY_SNRS = np.array([-14, -12, -10, -8, -6, -4], dtype=float)


def select(curve_file, grid_file):
    """Load a curve and select the six SNR points shown in the paper figure."""
    curve = np.load(D / curve_file)
    grid = np.load(D / grid_file).astype(float)
    if len(curve) != len(grid):
        raise ValueError(f"Length mismatch: {curve_file} ({len(curve)}) vs {grid_file} ({len(grid)})")
    indices = []
    for snr in DISPLAY_SNRS:
        matches = np.flatnonzero(np.isclose(grid, snr))
        if len(matches) != 1:
            raise ValueError(f"{grid_file} does not contain exactly one {snr:g} dB point")
        indices.append(int(matches[0]))
    return curve[indices]


def panel_curves(tag, lag, neural_suffix, grid_file):
    baseline_suffix = f"_{tag}"
    return [
        ("Psi-NN", select(f"pd_vs_snr_pablos{neural_suffix}.npy", grid_file),
         dict(color="tab:red", marker="D", linestyle="-")),
        ("CAE", select(f"pd_vs_snr_cae{neural_suffix}.npy", grid_file),
         dict(color="tab:blue", marker="v", linestyle="-")),
        (f"CAV (L={lag})", select(f"pd_vs_snr_cav_L{lag}{baseline_suffix}.npy", grid_file),
         dict(color="tab:purple", marker="^", linestyle="--")),
        (f"MME (L={lag})", select(f"pd_vs_snr_mme_L{lag}{baseline_suffix}.npy", grid_file),
         dict(color="tab:green", marker="s", linestyle="--")),
        (r"ED, known $\sigma^2$", select(f"pd_vs_snr_ed{baseline_suffix}.npy", grid_file),
         dict(color="black", marker="o", linestyle="-.")),
        (r"ED, $\pm2$ dB noise unc.", select(f"pd_vs_snr_ed_unc2dB{baseline_suffix}.npy", grid_file),
         dict(color="0.55", marker="o", linestyle=":")),
    ]


panels = [
    ("(a) 4 samples/symbol", panel_curves("advisor_sps4", 4, "", "snr_points_advisor_sps4.npy")),
    ("(b) 8 samples/symbol", panel_curves("sps8", 8, "_sps8", "snr_points_sps8.npy")),
]

plt.rcParams.update({"font.family": "serif"})
# Stack the panels vertically so each plot remains legible when Figure 2 is
# placed at IEEE single-column width.
fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.2), sharex=True, sharey=True)
for ax, (title, curves) in zip(axes, panels):
    for label, values, style in curves:
        ax.plot(
            DISPLAY_SNRS, values, label=label, linewidth=1.7,
            markersize=4.5, markerfacecolor="white", **style
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks(DISPLAY_SNRS)
    ax.set_ylim(-0.02, 1.03)
    ax.grid(alpha=0.3, linewidth=0.5)
    # Keep the legend above the low-SNR energy-detector curves so the
    # plotted ED results remain visible in both panels of the paper figure.
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_ylabel(r"$P_d$")
axes[1].set_xlabel("SNR (dB)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pd_vs_snr_4sps_8sps.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "pd_vs_snr_4sps_8sps.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved paper/figures/pd_vs_snr_4sps_8sps.png and .pdf")
