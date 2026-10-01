"""Create the paper's empirical CAE/Psi-NN beta histograms at -6 dB."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


data = torch.load("spectrum_data/test_data_advisor_sps4.pt", weights_only=False)
labels = data["labels"].numpy()
snrs = data["snrs"].numpy()
output_dir = Path("paper/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Use the advisor-confirmed paper checkpoints evaluated on the same corrected
# 4-sps data. Their saved scores are aligned with this test dataset.
models = [
    ("(a) Pablos CAE", np.load("spectrum_data/scores_test_cae.npy"),
     np.load("spectrum_data/scores_calib_cae.npy")),
    ("(b) Psi-NN", np.load("spectrum_data/scores_test_psinn.npy"),
     np.load("spectrum_data/scores_calib_psinn.npy")),
]

selected = np.isclose(snrs, -6)
if not selected.any():
    raise ValueError("The selected test dataset has no observations at -6 dB")

plt.rcParams.update({"font.family": "serif"})
fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.8))
thresholds = []

for ax, (title, beta_test, beta_calib) in zip(axes, models):
    if len(beta_test) != len(labels):
        raise ValueError(f"{title} score count does not match the 4-sps test dataset")

    # Use the independent calibration set for the 1% upper-tail threshold.
    gamma = float(np.quantile(beta_calib, 0.99))
    thresholds.append(gamma)
    noise = beta_test[selected & (labels == 0)]
    signal = beta_test[selected & (labels == 1)]
    if len(noise) == 0 or len(signal) == 0:
        raise ValueError(f"{title} has no H0 or H1 samples at -6 dB")

    lo = min(noise.min(), signal.min())
    hi = max(noise.max(), signal.max())
    pad = 0.04 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 46)

    # Show the empirical samples without KDE or other curve smoothing,
    # matching the histogram presentation in the previous paper.
    ax.hist(noise, bins=bins, density=True, alpha=0.58, color="tab:blue",
            edgecolor="white", linewidth=0.3,
            label=r"Noise ($\mathcal{H}_0$)")
    ax.hist(signal, bins=bins, density=True, alpha=0.58, color="tab:orange",
            edgecolor="white", linewidth=0.3,
            label=r"Signal ($\mathcal{H}_1$)")
    ax.axvline(gamma, color="tab:red", linestyle="--", linewidth=1.4,
               label=rf"$\gamma={gamma:.3f}$")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Reconstruction statistic, $\beta$")
    ax.grid(alpha=0.22, linewidth=0.45)
    ax.legend(fontsize=7, loc="upper right")

axes[0].set_ylabel("Probability density")
axes[1].set_ylabel("Probability density")
fig.tight_layout()
fig.savefig(output_dir / "beta_pdf_minus6db.pdf", bbox_inches="tight")
fig.savefig(output_dir / "beta_pdf_minus6db.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(
    "Saved paper/figures/beta_pdf_minus6db.pdf and .png "
    f"(CAE gamma={thresholds[0]:.6f}, Psi-NN gamma={thresholds[1]:.6f})"
)
