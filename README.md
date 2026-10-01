# Psi-NN Spectrum Sensing

This repository contains the corrected experiments for the Psi-NN spectrum
sensing paper. The paper studies a noise-trained, I-only pseudo-invertible
convolutional autoencoder and compares it with the Pablos CAE, CAV, MME, and
energy detection at a target false-alarm probability of 0.01.

Start with [PROJECT_MAP.md](PROJECT_MAP.md) for a file-by-file guide. The local
paper files are under [`paper/`](paper/).

## Current implementation

### 1. Corrected signal generator

`generate_spectrum_dataset.py` contains the corrected signal generator with the
optional Rapp nonlinearity support retained.

- Every modulation now uses one receiver sampling rate selected by `--sps`.
  The default and main-paper value is four samples per symbol.
- Blocks use a random symbol-timing offset. `--no-random-timing` restores the
  earlier aligned behavior for a controlled ablation.
- Raised-cosine start-up transients are discarded before blocks are selected.
- Cross 32-QAM uses the symmetric 6-by-6 constellation with the four corners
  removed. This eliminates the earlier constellation-dependent DC bias.
- `--snr-points` accepts an explicit SNR grid.
- Training-noise generation is unchanged, so the generator corrections do not
  require retraining the saved neural models.

The main corrected test data use BPSK, QPSK, 16-QAM, and cross 32-QAM; raised-
cosine roll-off 0.35; random timing; 1024 samples per block; four samples per
symbol; and a uniform +/-2 dB SNR spread around each nominal SNR with a fixed
noise floor.

### 2. Held-out empirical thresholds

Paper-facing thresholds no longer use a Gaussian fit to training-noise scores.
For an upper-tail detector, the threshold is

```python
gamma = np.quantile(beta_calib, 1 - pfa)
```

where `beta_calib` comes from independent held-out calibration noise. Every
detector reports its measured test false-alarm probability. The calibration
files are:

- `spectrum_data/calib_noise.pt` for normalized neural inputs;
- `spectrum_data/calib_noise_raw.pt` for raw-domain calculations.

The current paper evaluators using this protocol are:

- `evaluate_pablos.py`;
- `spectrum_data/evaluate_anomalies_cae.py`;
- `evaluate_baselines.py`;
- `verify_mod_auc_10db.py`;
- `experiments/ablation/evaluate_model_c.py`;
- `experiments/nonlinearity/evaluate_nonlinearity.py`.

### 3. Classical baselines

`evaluate_baselines.py` evaluates the following detectors on the same I-channel
data used by the neural models:

- CAV with lag dimensions 4 and 8;
- MME with lag dimensions 4 and 8;
- energy detection with known noise variance;
- energy detection with a threshold designed for +/-2 dB noise-power
  uncertainty.

CAV and MME use normalized I-channel blocks. Energy detection uses raw I-channel
blocks because received energy is its statistic. The +/-2 dB variation applied
to signal-present test blocks is **SNR uncertainty** with a fixed noise floor.
Only the special uncertainty-aware energy detector models **noise-power
uncertainty**.

### 4. Canonical checkpoints

The canonical paper checkpoints are:

| Paper model | Checkpoint | Result prefix |
| --- | --- | --- |
| Psi-NN, I-only | `spectrum_data/pablos_200epochs.pth` | `pd_vs_snr_pablos` |
| Pablos CAE | `spectrum_data/cae_best.pth` | `pd_vs_snr_cae` |

The file `experiments/ablation/output/compare_A_lowcap-original.pth` is an
architecture-ablation checkpoint and must not be used as the paper CAE.

### 5. Model naming

The paper's Psi-NN is the I-only `AE_Pablos1d` model evaluated by
`evaluate_pablos.py`. Files named `pd_vs_snr_psinn.npy` belong to the older
two-channel I/Q ablation and are not the main paper curve.

### 6. Terminology

- Use **SNR uncertainty** for the per-block +/-2 dB SNR variation with fixed
  noise power.
- Use **noise-power uncertainty** only for the uncertainty-aware energy
  detector.
- Use **measured false-alarm probability** when reporting the operating point.

## Repository layout

```text
Research_PSNN/
├── README.md
├── PROJECT_MAP.md
├── paper/
│   ├── manuscript.tex
│   ├── references.bib
│   └── figures/
├── experiments/
│   ├── ablation/
│   │   ├── README.md
│   │   ├── evaluate_model_c.py
│   │   └── output/
│   └── nonlinearity/
│       ├── README.md
│       ├── rf_nonlinearity.py
│       ├── evaluate_nonlinearity.py
│       ├── test_nonlinearity.py
│       └── output/
├── spectrum_data/              # Current datasets, checkpoints, and scores
├── archive/                    # Superseded code, figures, and evaluator plots
├── generate_spectrum_dataset.py
├── make_calibration_noise.py
├── evaluate_pablos.py
├── evaluate_baselines.py
├── bootstrap_ci.py
├── plot_beta_pdf_minus6db.py
└── plot_sps_comparison.py
```

Active Python modules remain at the repository root when other scripts import
them or the documented commands assume that location. Superseded visual outputs
and unrelated experiments are under `archive/`.

## Environment

The existing virtual environment can be used from the repository root:

```bash
cd /Users/lux/Desktop/Research_PSNN
.venv/bin/python --version
```

Required Python packages include PyTorch, NumPy, SciPy, scikit-learn, and
Matplotlib.

## Main four-sample experiment

### Step 1: calibration noise

```bash
.venv/bin/python make_calibration_noise.py
```

This creates 10,000 held-out noise blocks using a seed independent of training
and testing.

### Step 2: corrected test data

```bash
.venv/bin/python generate_spectrum_dataset.py \
  --tag advisor_sps4 \
  --snr-uncertainty-db 2 \
  --sps 4 \
  --samples-per-cell 1000 \
  --noise-per-snr 1000 \
  --snr-points -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 \
  --skip-training-save
```

Outputs:

- `spectrum_data/test_data_advisor_sps4.pt`;
- `spectrum_data/test_data_raw_advisor_sps4.pt`.

The saved test set contains 1,000 signal blocks per modulation and SNR and
1,000 noise blocks per SNR index.

### Step 3: paper neural models

```bash
SPECTRUM_TEST_DATA_PSINN=spectrum_data/test_data_advisor_sps4.pt \
  .venv/bin/python evaluate_pablos.py

SPECTRUM_TEST_DATA_CAE=spectrum_data/test_data_advisor_sps4.pt \
  .venv/bin/python spectrum_data/evaluate_anomalies_cae.py
```

These commands save per-block calibration and test scores for subsequent
confidence intervals and figures.

### Step 4: classical baselines

```bash
.venv/bin/python evaluate_baselines.py \
  --test spectrum_data/test_data_raw_advisor_sps4.pt \
  --tag advisor_sps4 \
  --L 4 8 \
  --noise-unc-db 2 \
  --save-scores
```

### Step 5: paper figures

```bash
.venv/bin/python plot_beta_pdf_minus6db.py
.venv/bin/python plot_sps_comparison.py
```

The scripts save current PNG and PDF files under `paper/figures/`:

- `beta_pdf_minus6db.*` — stacked empirical CAE and Psi-NN beta histograms at
  -6 dB, without KDE smoothing;
- `pd_vs_snr_4sps_8sps.*` — stacked 4-sps and 8-sps detection curves.

### Step 6: bootstrap confidence intervals

```bash
.venv/bin/python bootstrap_ci.py \
  --a-calib spectrum_data/scores_calib_psinn.npy \
  --a-test spectrum_data/scores_test_psinn.npy \
  --b-calib spectrum_data/scores_calib_cav_L4_advisor_sps4.npy \
  --b-test spectrum_data/scores_test_cav_L4_advisor_sps4.npy \
  --names Psi-NN CAV
```

The current report is
`spectrum_data/bootstrap_ci_psinn_vs_cav_advisor_sps4.txt`.

## Oversampling sweep

The oversampling study repeats dataset generation and model/baseline evaluation
at two and eight samples per symbol. Use tags `sps2` and `sps8`, and point the
neural evaluators to the corresponding normalized test files. Baselines use the
matching raw files.

The paper reports detection probability at -10 dB:

| Detector | 2 sps | 4 sps | 8 sps |
| --- | ---: | ---: | ---: |
| Psi-NN | 0.302 | 0.798 | 0.937 |
| Pablos CAE | 0.086 | 0.396 | 0.818 |
| CAV | 0.425 | 0.769 | 0.914 |
| MME | 0.434 | 0.721 | 0.901 |
| ED, known noise variance | 0.609 | 0.592 | 0.585 |
| ED, +/-2 dB noise uncertainty | 0.000 | 0.000 | 0.000 |

The source table is `spectrum_data/oversampling_sweep_minus10db.md`.

## Current main results

At four samples per symbol and a target false-alarm probability of 0.01:

| Detector | -14 dB | -12 dB | -10 dB | -8 dB |
| --- | ---: | ---: | ---: | ---: |
| Psi-NN | 0.289 | 0.549 | 0.798 | 0.966 |
| CAV, L=4 | 0.219 | 0.487 | 0.769 | 0.960 |
| MME, L=4 | 0.188 | 0.440 | 0.721 | 0.943 |
| ED, known noise variance | 0.150 | 0.333 | 0.592 | 0.847 |
| Pablos CAE | 0.109 | 0.231 | 0.396 | 0.632 |
| ED, +/-2 dB noise uncertainty | 0.000 | 0.000 | 0.000 | 0.000 |

Psi-NN minus CAV bootstrap results:

| SNR | Difference | 95% interval |
| --- | ---: | ---: |
| -14 dB | 0.070 | [0.044, 0.094] |
| -12 dB | 0.062 | [0.033, 0.090] |
| -10 dB | 0.030 | [0.007, 0.050] |
| -8 dB | 0.006 | [-0.002, 0.013] |

## Decoder ablation

All ablation code, checkpoints, and outputs are under
[`experiments/ablation/`](experiments/ablation/). The current paper comparison
uses Model C, which has the same encoder dimensions as Psi-NN and an independently
trained symmetric decoder.

```bash
.venv/bin/python experiments/ablation/evaluate_model_c.py
```

Corrected Model C detection probabilities are 0.073, 0.169, 0.296, and 0.519
at -14, -12, -10, and -8 dB, respectively. Its measured false-alarm probability
is 0.0112. The report is
`experiments/ablation/output/model_c_corrected_results.txt`.

## Complexity

Run:

```bash
.venv/bin/python compute_complexity.py
```

Verified deployment-oriented counts:

- Psi-NN convolutional reconstruction: 13,102,160 MACs per block, assuming
  pseudoinverse matrices are cached after training;
- CAV lag products: approximately 4,090 for L=4 and 8,164 for L=8.

The report is `spectrum_data/complexity_verification.md`.

## Nonlinearity experiment

All nonlinearity code, documentation, tests, and saved outputs are under
[`experiments/nonlinearity/`](experiments/nonlinearity/).

The existing `output/nonlinearity_full/` sweep is retained for provenance but
is **superseded**: it used `compare_A_lowcap-original.pth`, omitted the classical
baselines, and did not include the new -14 and -12 dB points. Do not cite those
numbers as current paper results.

The paper should include a nonlinearity subsection only after rerunning the
current checkpoints and all paper detectors under the corrected protocol.

## Training and reproducibility

The generator corrections do not require retraining because training-noise
generation did not change. The saved paper checkpoints are reused. If additional
time is available, training Psi-NN and CAE with three to five independent seeds can be
reported as a later robustness study using the same checkpoint-selection rule.

## Legacy material

The `archive/` directory contains older figures, evaluator plots, code, LaTeX
fragments, and the unrelated ItalyPowerDemand example. Nothing under `archive/`
is required to reproduce the current paper results.

## Validation

Run the nonlinearity unit tests with:

```bash
.venv/bin/python -m unittest experiments.nonlinearity.test_nonlinearity -v
```

The suite checks Rapp saturation, phase preservation, transmitter/receiver
ordering, paired data generation, normalization, parameter validation, and
confidence-interval helpers.
