# Nonlinearity robustness experiment (reviewer comment c)

The new `evaluate_nonlinearity.py` evaluates frozen CAE and Psi-NN checkpoints on
paired linear, transmitter-compressed, receiver-compressed, and optionally combined
conditions. It generates samples in memory one SNR cell at a time and never rewrites
training data, checkpoints, or existing evaluation outputs. Every run needs a new
output directory.

## Model selection

The default `--model-pair paper` follows `verify_mod_auc_10db.py`:

| Pair | CAE checkpoint | Psi-NN checkpoint | Inputs |
| --- | --- | --- | --- |
| `paper` | `cae_best.pth` | `pablos_200epochs.pth` | Both I-only |
| `legacy` | `cae_best.pth` | `psl_cnn_200epochs.pth` | CAE I-only; Psi-NN I/Q |
| `matched` | `cae_best.pth` | `psl_cnn_200epochs_ch1.pth` | Both I-only |

Checkpoints are under `spectrum_data/`. The advisor confirmed `cae_best.pth` as
the paper CAE checkpoint on 26 September 2026. Override paths with `--cae-checkpoint` and
`--psi-checkpoint`; the architecture must match the selected pair. `CAE` has the
same parameter structure and forward computation as `LowCapOriginal` in the paper
verification script. All pairs use joint I/Q per-window mean/std normalization
followed by channel slicing, consistent with the actual existing evaluators.
The `compare_A_lowcap-original.pth` file remains an architecture-ablation
checkpoint and must not be used for the main paper comparison.

## Run

From the project root, using the existing virtual environment:

```bash
# Main transmitter experiment: linear reference plus IBO = 9, 6, 3, 0 dB.
.venv/bin/python evaluate_nonlinearity.py \
  --output spectrum_data/nonlinearity_tx

# Receiver experiment only: linear reference plus three receiver saturation levels.
.venv/bin/python evaluate_nonlinearity.py \
  --output spectrum_data/nonlinearity_rx \
  --tx-ibo-db --rx-backoff-db 12 6 0

# TX-only, RX-only, and their Cartesian combinations, plus linear reference.
.venv/bin/python evaluate_nonlinearity.py \
  --output spectrum_data/nonlinearity_combined \
  --tx-ibo-db 6 0 --rx-backoff-db 12 6 --combined
```

Defaults: 1,000 signal windows per modulation/SNR, 1,000 independent test H0
windows per SNR, 10,000 independent calibration H0 windows, simulation seeds
101/202/303, 11 SNR points from -10 to +10 dB, existing pulse shaping and ±2 dB
uniform SNR uncertainty. These are simulation seeds, **not training repetitions**.
Models are never retrained. CPU is the default; `--device cuda` is available.

A small integration check (not evidence for a paper conclusion):

```bash
.venv/bin/python evaluate_nonlinearity.py \
  --output spectrum_data/nonlinearity_quick_check \
  --seeds 101 --snrs -6 6 --samples-per-cell 16 --noise-per-snr 32 \
  --calibration-samples 256 --tx-ibo-db 3 --rx-backoff-db 6 --combined \
  --bootstrap 200 --batch-size 32
```

## Physical model and SNR convention

For complex envelope x = I + jQ, the unity-gain Rapp AM/AM model is

```
f(x) = x / (1 + (|x| / A_sat)^(2p))^(1/(2p)), p = 2 by default.
```

I and Q share the same envelope-dependent gain. Phase is preserved. This models
memoryless compression only, not AM/PM distortion, amplifier memory, ADC
quantization, a blocker, or a specific measured radio. See the
[MathWorks Rapp model description](https://www.mathworks.com/help/simrf/ref/idealizedbasebandamplifierold.html).

TX saturation is set per waveform using its pre-amplifier complex mean power:
`IBO = 10 log10(A_sat² / mean(|x|²))`. Lower IBO means stronger compression.
The generator's original component-peak normalization is retained; computing IBO
from mean complex power makes the chosen TX operating point independent of that
arbitrary amplitude scale.

Signal path:

```
symbols -> existing pulse shaping -> TX Rapp -> received-power scaling
        -> add AWGN -> RX Rapp -> joint I/Q normalization -> model input
```

`add_awgn()` scales the distorted signal so the specified SNR is **total post-TX,
pre-RX signal power divided by noise power**. This compensates TX power loss and
isolates waveform distortion at equal received SNR. It is not a fixed-link-budget
experiment or a signal-to-distortion-and-noise ratio. Do not describe the SNR as
post-receiver SNR.

RX saturation is fixed across windows and SNRs:
`A_sat² = 2 * 10^(rx_backoff_db / 10)` because each noise component has variance 1.
The RX acts on both signal-plus-noise H1 and noise-only H0. RX back-off is referenced
to complex noise power; it is not the TX IBO or an input-dependent AGC rule.
TX has no effect on H0. Random draws are identical across conditions for each
seed/SNR cell, and the same waveforms are used for both models.

## Thresholds, metrics, and interpretation

Both models use the existing beta score and upper-tail `beta > gamma` decision.
The script does not pick a score direction from test labels. It calibrates gamma
at the empirical `1-Pfa` quantile (NumPy default linear interpolation) on newly generated noise
independent of training and testing. This differs from the old Gaussian fit to
training-noise scores, so the linear reference must be evaluated in this new
protocol too; it need not reproduce the old paper curve exactly.

Every condition reports:

- **Frozen**: clean calibration thresholds reused under all distortions.
- **Recalibrated**: thresholds from H0 passed through the corresponding receiver.
  TX-only and linear conditions have identical frozen/recalibrated thresholds.
- Per-modulation, per-SNR detection probability and ROC AUC; measured test Pfa.
- 95% Wilson intervals for Pd and Pfa.
- Paired bootstrap intervals for `Pd(PsiNN) - Pd(CAE)`, using the same windows.

Intervals are per simulation seed, conditional on the fitted threshold and fixed
checkpoint. They do not include training variability, threshold-estimation
uncertainty, or simultaneous-comparison corrections. Repeat seeds are reported
separately; figures show their arithmetic means. Small calibration/test sets are
only suitable for checking execution, particularly at Pfa=0.01.

Outputs in each run directory:

| File | Contents |
| --- | --- |
| `config.json` | Arguments, model classes, checkpoint paths/SHA256, protocol |
| `metrics.csv` | Pd/Pfa intervals, AUC, thresholds and sample counts |
| `paired_differences.csv` | Psi-NN minus CAE Pd and paired 95% intervals |
| `scores_*.npz` | Paired beta scores, H0/H1 labels and modulation labels |
| `pd_vs_snr_frozen.png` | Per-modulation curves under fixed clean thresholds |
| `pd_vs_snr_recalibrated.png` | Per-modulation curves after H0 recalibration |
| `COMPLETE.txt` | Present only after all conditions and figures finish |

Report measured Pfa alongside Pd. A detection increase caused by receiver-induced
false alarms is not an improvement at equal false-alarm rate. Compare the
performance gap to the linear reference, especially for 16-QAM/32-QAM; a gap whose
interval covers zero should not be described as demonstrated separation.

The existing pulse-shaping and 32-point constellation definitions are preserved
for comparability. This experiment does not validate or correct those definitions.
A model of AM/PM distortion or a measured amplifier response would be an additional
experiment if required; the current results support claims about Rapp AM/AM only.

## Generate saved datasets separately

The main sweep needs no saved test datasets. For use with other evaluators:

```bash
.venv/bin/python generate_spectrum_dataset.py --tag tx_rapp_3db \
  --snr-uncertainty-db 2 --tx-ibo-db 3 --rapp-p 2 --seed 101 \
  --samples-per-cell 1000 --skip-training-save
```

Generator controls also include `--rx-backoff-db`, `--noise-per-snr`, and `--seed`.
Noise training tensors returned by the Python API remain **clean**: receiver
impairment is a test-domain shift, not automatic retraining. The sweep generates
receiver-impaired calibration noise separately. Nonlinear CLI generation requires
a tag and preserves training files; it refuses to overwrite existing test files.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover complex saturation/phase, IBO and receiver power conventions, linear
bypass, identical paired H0, TX/RX ordering on H1 and H0, normalization, invalid
parameters, and confidence-interval calculations. Run the small integration check
to additionally exercise real checkpoints, output writing, and figures.
