"""
Human-readable names for figures and console output.

Use these constants so legends stay consistent across scripts.

Naming intent
-------------
- **CAE1ch - LowCap** — Standalone spectrum-domain convolutional autoencoder (`cae_spectrum.CAE`),
  evaluated on the **I-only** slice (1×1024); lower-capacity channel schedule vs full 2-ch stacks.
- **1D Psl-CNN** — Pseudo-invertible 1D CNN autoencoder (`AE_Classifier1d`), 2-channel I/Q.
- **Conv AE baseline** — Standard 1-D convolutional autoencoder with *separate* encoder and
  decoder weights (`AE_Baseline_Classifier1d`), 2-channel I/Q. Not the `CAE` spectrum class above.
- **Pablos-style AE (I-only)** — Reference autoencoder aligned with Pablos et al. on the
  *in-phase channel only* (1 x 1024), implemented as `AE_Pablos1d` / `evaluate_pablos.py`.
  Same task family as above but *not* the same input width as the 2-ch models.
"""

# --- $P_d$ vs SNR combined plot (`plot_pd_vs_snr.py`) ---------------------------------
LEGEND_SPECTRUM_CAE = "CAE1ch - LowCap"
LEGEND_PSL_CNN = "1D Psl-CNN (2-ch I/Q)"
LEGEND_CONV_AE_BASELINE = "Conv AE baseline (2-ch I/Q)"
LEGEND_PABLOS_I_ONLY = "Pablos-style AE (I-only, pseudo-inv.)"

# --- ROC / histogram short tags ------------------------------------------------------
ROC_SPECTRUM_CAE = "CAE1ch - LowCap"
ROC_PSL_CNN = "1D Psl-CNN"
ROC_CONV_AE_BASELINE = "Conv AE baseline"
ROC_PABLOS_STYLE = "Pablos-style AE (I-only)"

# --- Tables / CSV-friendly -----------------------------------------------------------
TABLE_SPECTRUM_CAE = "CAE1ch - LowCap"
TABLE_PSL_CNN = "1D Psl-CNN"
TABLE_CONV_AE_BASELINE = "Conv AE baseline"
TABLE_PABLOS_STYLE = "Pablos-style AE"

# --- Channel ablation (`evaluate_channel_ablation.py`) -----------------------------
ABLATION_PSL_CNN = "1D Psl-CNN"
ABLATION_CONV_AE = "Conv AE baseline"
