"""
Human-readable names for figures and console output.

Use these constants so legends stay consistent across scripts.

Naming intent
-------------
- **CAE1ch - LowCap** — Standalone spectrum-domain convolutional autoencoder (`cae_spectrum.CAE`),
  evaluated on the **I-only** slice (1×1024); lower-capacity channel schedule vs full 2-ch stacks.
- **Psi-NN (I/Q ablation)** — Pseudo-invertible 1D CNN autoencoder
  (`AE_Classifier1d`), evaluated on two-channel I/Q input. This is not the
  paper's main Psi-NN curve.
- **Conv AE baseline** — Standard 1-D convolutional autoencoder with *separate* encoder and
  decoder weights (`AE_Baseline_Classifier1d`), 2-channel I/Q. Not the `CAE` spectrum class above.
- **Psi-NN (I-only)** — The paper's main Psi-NN model, implemented as
  `AE_Pablos1d` / `evaluate_pablos.py` and evaluated on the in-phase channel
  only (1 x 1024), matching the CAE input.
"""

# Canonical checkpoints for every paper-facing CAE/Psi-NN comparison.
# Confirmed by the advisor for the resubmission on 2026-09-26.
PAPER_CAE_CHECKPOINT = "spectrum_data/cae_best.pth"
PAPER_PSINN_CHECKPOINT = "spectrum_data/pablos_200epochs.pth"

# --- $P_d$ vs SNR combined plot (`plot_pd_vs_snr.py`) ---------------------------------
LEGEND_SPECTRUM_CAE = "CAE1ch - LowCap"
LEGEND_PSL_CNN = "Psi-NN (I/Q ablation)"
LEGEND_CONV_AE_BASELINE = "Conv AE baseline (2-ch I/Q)"
LEGEND_PABLOS_I_ONLY = "Psi-NN (I-only)"

# --- ROC / histogram short tags ------------------------------------------------------
ROC_SPECTRUM_CAE = "CAE1ch - LowCap"
ROC_PSL_CNN = "Psi-NN (I/Q ablation)"
ROC_CONV_AE_BASELINE = "Conv AE baseline"
ROC_PABLOS_STYLE = "Psi-NN (I-only)"

# --- Tables / CSV-friendly -----------------------------------------------------------
TABLE_SPECTRUM_CAE = "CAE1ch - LowCap"
TABLE_PSL_CNN = "Psi-NN (I/Q ablation)"
TABLE_CONV_AE_BASELINE = "Conv AE baseline"
TABLE_PABLOS_STYLE = "Psi-NN (I-only)"

# --- Channel ablation (`evaluate_channel_ablation.py`) -----------------------------
ABLATION_PSL_CNN = "Psi-NN architecture"
ABLATION_CONV_AE = "Conv AE baseline"
