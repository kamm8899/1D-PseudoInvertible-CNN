# Modulation-Agnostic Spectrum Sensing — Anomaly Detection

Research code for **1D pseudo-invertible** and **baseline** convolutional autoencoders on synthetic I/Q data, following the **β statistic** and **Neyman–Pearson** style calibration from:

> Pablos et al. (2022). *Modulation-agnostic spectrum sensing using deep learning.* ICT Express.  
> https://www.sciencedirect.com/science/article/pii/S2405959522000480

---

## Overview

- **Goal:** Detect primary-user–like **modulated signals** vs **AWGN-only** segments using models trained **only on noise** (unsupervised anomaly detection).
- **Input:** Batches of shape `(N, C, 1024)` with `C ∈ {1, 2}` depending on the experiment (I-only vs I/Q).
- **Score:** Per-sample **coefficient of determination β** from autoencoder reconstruction (same spirit as Pablos et al., Eq. (5)).
- **Threshold:** Scalar **γ** from mean and std of β on **training noise** at a nominal **\(P_{\mathrm{fa}}\)** (e.g. 0.01 and 0.05 in scripts).

**Models compared in this repo**

| Name in docs / `experiment_labels.py` | Code | Role |
|----------------------------------------|------|------|
| **Psi-NN (I/Q ablation)** | `AE_Classifier1d` (`psinn_layer_1d.py`) | Separate two-channel experiment; encoder and decoder share weights via **`PsiNNConv1d`**. |
| **Conv AE baseline** | `AE_Baseline_Classifier1d` | Separate **`Conv1d`** encoder and **`ConvTranspose1d`** decoder; optional small **`C5`** classifier head (reconstruction path `AE()` used for sensing). |
| **CAE1ch - LowCap** | `CAE` (`cae_spectrum.py`) | 1×1024 I-only slice; **upsample + conv** decoder; plots use this display name (`experiment_labels.py`). |
| **Psi-NN (I-only)** | `AE_Pablos1d` (`psinn_layer_1d_pablos.py`) | Main paper Psi-NN; single-channel stack evaluated by `evaluate_pablos.py`. |

**Important:** The **`CAE` class** in code is the spectrum autoencoder; figures label it **CAE1ch - LowCap**. The conv AE baseline is a different architecture; the distinction is **architecture + parameter tying**, not “CAE vs non-CAE” in the taxonomy sense.

**Main inverted evaluation (`evaluate_anomaly_inverted.py`)** uses a **mixed-tail** convention (same nominal \(P_{\mathrm{fa}}\), different inequality):

- **Psi-NN I/Q ablation:** **upper tail** on H₀ — \(P_{\mathrm{fa}} = P(\beta > \gamma \mid H_0)\), ROC on **+β**, \(P_d = P(\beta > \gamma \mid H_1)\).
- **Conv AE baseline:** **lower tail** on H₀ — \(P_{\mathrm{fa}} = P(\beta < \gamma \mid H_0)\), ROC on **−β**, \(P_d = P(\beta < \gamma \mid H_1)\).

Other scripts (spectrum CAE, Pablos, forward PsiNN) document their own tail / ROC orientation in-file.

---

## Repository layout

```
Research_PSNN/
├── psinn_layer_1d.py           # PsiNNConv1d, AE_Classifier1d (I/Q ablation), AE_Baseline_Classifier1d
├── psinn_layer_1d_pablos.py    # AE_Pablos1d (Pablos-style I-only)
├── cae_spectrum.py             # Spectrum CAE class + optional __main__ training
├── spectrum_paths.py           # Test tensor paths, full-channel metadata checks, env overrides
├── experiment_labels.py      # Consistent plot/log names (combined Pd figure, etc.)
├── generate_spectrum_dataset.py
├── train_models.py             # Train 2-ch Psi-NN ablation + conv AE baseline (200 epochs, k=5)
├── train_pablos.py             # Train the paper's I-only Psi-NN weights
├── evaluate_anomaly_inverted.py    # Primary full eval: 2-ch, mixed tails, Pd vs SNR, plots
├── evaluate_pablos.py
├── energy_detector.py          # ED benchmark → spectrum_data/pd_vs_snr_ed.npy
├── plot_pd_vs_snr.py           # Combined Pd vs SNR figure (loads *.npy from other scripts)
├── load_data.py
├── diagnostic_recon_error.py
├── run_cae_ablation_report.py  # Summarize CAE evals across test-channel variants
│
├── spectrum_data/
│   ├── train_noise.pt / train_noise_raw.pt   # Normalized vs raw training noise
│   ├── test_data_full.pt / test_data_raw_full.pt   # Required default for PsiNN / Pablos / ED / CAE
│   ├── psl_cnn_*epochs.pth, baseline_*epochs.pth
│   ├── pablos_200epochs.pth, cae_best.pth (if trained)
│   ├── evaluate_anomalies_forward.py   # Forward PsiNN (100-epoch ckpts, k=3)
│   ├── evaluate_anomalies_cae.py       # Spectrum CAE eval
│   ├── train_models_1channel.py        # Train 1-ch Psl-CNN + baseline
│   ├── evaluate_channel_ablation.py    # 1ch vs 2ch table + figures → channel_ablation_results*.csv/png
│   └── pd_vs_snr_*.npy, snr_points.npy, evaluation_results*.txt
│
├── anomalies_Psi-NN_inverted/      # Plots from evaluate_anomaly_inverted.py
├── anomalies_PSi-NN_forward/
├── anomalies_CAE/
├── anomalies_Pablos/
├── ItalyPowerDemandTest/           # Separate toy demo (not required for spectrum pipeline)
└── Old Version/                    # Legacy 2D PsiNN reference
```

The advisor-confirmed paper checkpoints are `spectrum_data/cae_best.pth` for CAE
and `spectrum_data/pablos_200epochs.pth` for Psi-NN. The similarly shaped
`compare_A_lowcap-original.pth` checkpoint belongs only to the architecture
ablation and is not used for paper-facing comparisons.

There is **no** `evaluate_anomaly.py` or `evaluate_anomalies_forward_cae.py` in the tree anymore; use the scripts listed above.

---

## Dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

**macOS:** If you see `OMP: Error #15` / duplicate `libomp`, set before importing PyTorch/NumPy:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Training scripts in `spectrum_data/` set this internally **before** `import torch`.

---

## Data generation

Default **PsiNN / Pablos / energy detector / CAE (default path)** expect the **full** test channel: **pulse shaping** and **SNR uncertainty** in the saved metadata (`spectrum_paths.assert_psinn_full_channel_metadata`).

Generate the canonical files:

```bash
python generate_spectrum_dataset.py --tag full --snr-uncertainty-db 2
```

This writes (among others):

- `spectrum_data/train_noise.pt`, `train_noise_raw.pt`
- `spectrum_data/test_data_full.pt`, `test_data_raw_full.pt`

**Flags:** `--no-pulse-shaping`, `--tag <name>` for ablation copies (`test_data_<tag>.pt`). See `spectrum_paths.py` for **`SPECTRUM_TEST_DATA_*`** overrides.

**Note:** `spectrum_data/evaluate_anomalies_cae.py` feeds the spectrum **`CAE`** with the **in-phase slice** of the normalized packed tensors, `[:, 0:1, :]` (1×1024), while inverted PsiNN uses full **2×1024** normalized I/Q. The CAE script’s local variable `test_data_raw` is misleading: its default path is `test_data_full.pt`.

---

## Training

| Script | Output (examples) |
|--------|---------------------|
| `python train_models.py` | `spectrum_data/psl_cnn_200epochs.pth`, `baseline_200epochs.pth` — **2 ch**, **k=5**, matches `evaluate_anomaly_inverted.py`. |
| `python spectrum_data/train_models_1channel.py --epochs 200` | `psl_cnn_200epochs_ch1.pth`, `baseline_200epochs_ch1.pth` for **I-only** ablation. |
| `python train_pablos.py` | Weights for `evaluate_pablos.py` (e.g. `pablos_200epochs.pth`). |
| `python cae_spectrum.py` | Can train / save spectrum CAE checkpoints (see file `__main__`). |

**Forward** evaluation (`spectrum_data/evaluate_anomalies_forward.py`) loads **`psl_cnn_100epochs.pth`** / **`baseline_100epochs.pth`** with **k=3** — not the same checkpoint as the inverted 200-epoch run unless you align them.

---

## Evaluation (quick reference)

| Script | Purpose |
|--------|---------|
| `evaluate_anomaly_inverted.py` | Separate **2-ch Psi-NN I/Q ablation** + conv AE baseline: ROC, AUC, Youden, TPR tables, β/MSE plots, **Pd vs SNR** → `spectrum_data/pd_vs_snr_psinn.npy`, `pd_vs_snr_baseline.npy`, `snr_points.npy`. |
| `spectrum_data/evaluate_anomalies_forward.py` | Same scoring family, **100-epoch / k=3** checkpoints. |
| `spectrum_data/evaluate_anomalies_cae.py` | **Spectrum `CAE`**, upper-tail γ, **+β** ROC → `pd_vs_snr_cae.npy`. |
| `evaluate_pablos.py` | Main paper **Psi-NN (I-only)** → `pd_vs_snr_pablos.npy`. |
| `energy_detector.py` | Raw-domain energy → `pd_vs_snr_ed.npy`. |
| `spectrum_data/evaluate_channel_ablation.py` | Side-by-side **1ch vs 2ch** for Psl-CNN + baseline (same mixed-tail rule as inverted). |
| `plot_pd_vs_snr.py` | Reads the `pd_vs_snr_*.npy` files and saves **`pd_vs_snr_combined.png`** at repo root. |

Run **after** the corresponding training and eval steps so all `.npy` files exist (ED line optional in `plot_pd_vs_snr.py`).

---

## β and γ (short)

\[
\beta = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}
\]

with SSE/SST computed over **all** time and channel dimensions in the batch (see `compute_beta` in each eval script).

**γ** is set from **training-noise** β statistics and `scipy.stats.norm.ppf` at the chosen **\(P_{\mathrm{fa}}\)**. The **inequality** used for “signal” vs ROC sign of β depends on the script (see **Overview** for inverted mixed tails).

---

## Human-readable names (`experiment_labels.py`)

Plot legends and log strings are centralized in **`experiment_labels.py`**. The main paper model is **Psi-NN (I-only)**; the two-channel result is labeled **Psi-NN (I/Q ablation)**.

---

## Optional / legacy

- **`ItalyPowerDemandTest/`** — small **univariate** toy dataset for layer smoke tests; not the spectrum pipeline.
- **`Old Version/psinn_layer_and_autoencoder.py`** — 2D PsiNN reference only.

---

## License / attribution

Copyright notice appears in `psinn_layer_1d.py` (Jessica Kamman / Jessica Sinn). Cite Pablos et al. (2022) when using the β / NP methodology in publications.

## Hardware nonlinearity robustness (reviewer comment c)

See [NONLINEARITY.md](NONLINEARITY.md) for the Rapp transmitter/receiver model, paired experiment, checkpoint choices, SNR convention, and statistical reporting. Start the transmitter sweep with:

```bash
.venv/bin/python evaluate_nonlinearity.py --output spectrum_data/nonlinearity_tx
```

The default compares the I-only paper models identified in `verify_mod_auc_10db.py`, without retraining. Outputs include per-modulation Pd, measured Pfa, ROC AUC, paired confidence intervals, and figures. Receiver and combined sweeps are optional. Existing experiment files are preserved.

## Advisor item 2: held-out empirical thresholds

`evaluate_pablos.py`, `spectrum_data/evaluate_anomalies_cae.py`, and
`verify_mod_auc_10db.py` now calibrate on `spectrum_data/calib_noise.pt`
(normalized joint I/Q, then I-only slice), not training noise. Their upper-tail
threshold is `np.quantile(beta_calib, 1 - target_pfa)` with NumPy default
interpolation. Measured Pfa is the fraction of independent test H0 scores above
that threshold. CAE reports both its 0.01 and 0.05 operating points; CAE and Pablos
also record measured Pfa in their existing results text files. TPR tables use the
calibrated threshold rather than selecting a point from the test ROC.

The main paper evaluators save `scores_calib_psinn.npy`, `scores_test_psinn.npy`,
`scores_calib_cae.npy`, and `scores_test_cae.npy` under `spectrum_data/` for
`bootstrap_ci.py`. Test scores retain test-file order. Here `psinn` means the
I-only Pablos-style paper model. The verification script does not overwrite these
score files because it currently uses a different CAE checkpoint.

`evaluate_nonlinearity.py` continues to use independent simulated calibration
noise per seed, with receiver distortion applied before normalization when
recalibrating. It now uses the same default quantile interpolation and continues
to report measured Pfa and intervals in `metrics.csv`. Older two-channel/forward
and energy-detector scripts retain their previous calibration unless separately
updated. Existing saved results are not retroactively changed; rerun evaluators
to obtain results with the new thresholds.

## Advisor item 3: classical baselines on corrected data

The corrected 4-samples/symbol dataset is saved separately as
`test_data_advisor_sps4.pt` / `test_data_raw_advisor_sps4.pt` under `spectrum_data/`.
It contains 1,000 signal windows per modulation/SNR and 1,000 noise windows per
SNR, at -14 through +10 dB in 2 dB steps. It uses random timing, steady-state
filter output, symmetric cross 32-QAM, and ±2 dB SNR uncertainty.

```bash
.venv/bin/python evaluate_baselines.py \
  --test spectrum_data/test_data_raw_advisor_sps4.pt --tag advisor_sps4 \
  --L 4 8 --noise-unc-db 2 --save-scores
```

All six detectors use held-out `calib_noise_raw.pt`. MME/CAV use normalized I;
ED uses raw I to preserve energy information. Thresholds are detector-specific
99th percentiles. The uncertainty ED scales its threshold by `10**(2/10)` and is
then evaluated at nominal noise power, so its measured Pfa can be below 0.01.
The test-set variation is **SNR uncertainty with a fixed noise floor**; only the
special uncertainty-ED curve models **noise-power uncertainty**.
Results are `baselines_results_advisor_sps4.txt` and `.csv`; scores and Pd arrays
carry the same `_advisor_sps4` suffix. No neural-model comparison is implied until
those models are evaluated on the same corrected test set.

For subsequent paper-model runs, point `SPECTRUM_TEST_DATA_PSINN` and
`SPECTRUM_TEST_DATA_CAE` to `spectrum_data/test_data_advisor_sps4.pt`.
For bootstrap comparisons use `--test spectrum_data/test_data_raw_advisor_sps4.pt`
and the tagged baseline score filenames, e.g. `scores_calib_cav_L4_advisor_sps4.npy`
and `scores_test_cav_L4_advisor_sps4.npy`. The paper evaluators currently save
untagged score files: use only scores from the corresponding corrected-data run.
