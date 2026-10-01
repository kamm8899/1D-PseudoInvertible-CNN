# Research_PSNN project map

This repository is organized around the current resubmission workflow. Active
Python modules remain at the repository root because they import one another
and because the advisor's run commands assume that location.

## Current paper

- `paper/manuscript.tex` — local working manuscript.
- `paper/figures/beta_pdf_minus6db.pdf` — Figure 1: stacked empirical CAE and
  Psi-NN beta histograms at -6 dB.
- `paper/figures/pd_vs_snr_4sps_8sps.pdf` — Figure 2: stacked 4-sps and 8-sps
  detection curves.
- `paper/references.bib` — bibliography supplied for the resubmission.

Regenerate the paper figures from the repository root with:

```bash
.venv/bin/python plot_beta_pdf_minus6db.py
.venv/bin/python plot_sps_comparison.py
```

## Main experiment workflow

1. `make_calibration_noise.py` — independent held-out H0 calibration data.
2. `generate_spectrum_dataset.py` — corrected signal and noise generator.
3. `evaluate_pablos.py` — paper Psi-NN (`pablos_200epochs.pth`).
4. `spectrum_data/evaluate_anomalies_cae.py` — paper CAE (`cae_best.pth`).
5. `evaluate_baselines.py` — CAV, MME, and energy detectors.
6. `bootstrap_ci.py` — Psi-NN versus CAV confidence intervals.
7. `experiments/ablation/evaluate_model_c.py` — corrected matched-decoder ablation.
8. `compute_complexity.py` — reproducible MAC counts.

## Active model code

- `psinn_layer_1d.py` — pseudo-invertible convolution implementation.
- `psinn_layer_1d_pablos.py` — I-only paper Psi-NN architecture.
- `cae_spectrum.py` — Pablos CAE architecture and training.
- `experiment_labels.py` — canonical paper checkpoint names and labels.
- `spectrum_paths.py` — current dataset path validation.

## Current datasets and results

All large tensors, checkpoints, scores, and numerical result arrays remain in
`spectrum_data/` because the evaluation scripts use those paths directly.

The most important files are:

- `test_data_advisor_sps4.pt` and `test_data_raw_advisor_sps4.pt`
- `test_data_sps2.pt`, `test_data_raw_sps2.pt`
- `test_data_sps8.pt`, `test_data_raw_sps8.pt`
- `calib_noise.pt`, `calib_noise_raw.pt`
- `pablos_200epochs.pth`, `cae_best.pth`
- `experiments/ablation/output/compare_C_lowcap-symdec.pth`
- `baselines_results_advisor_sps4.txt`
- `bootstrap_ci_psinn_vs_cav_advisor_sps4.txt`
- `oversampling_sweep_minus10db.md`
- `experiments/ablation/output/model_c_corrected_results.txt`
- `complexity_verification.md`

## Experiment folders

- `experiments/nonlinearity/` — Rapp model, evaluator, focused plotting,
  tests, protocol, and saved nonlinearity outputs.
- `experiments/ablation/` — Model C and other ablation code, documentation,
  checkpoints, reports, scores, and figures.

The saved `experiments/nonlinearity/output/nonlinearity_full/` results are superseded because
they used the earlier CAE checkpoint and did not include all paper baselines.
Do not cite them without rerunning the experiment under the current protocol.

## Archive

The `archive/` directory contains material that is not part of the current
paper workflow:

- `archive/legacy_figures/` — superseded root-level plots.
- `archive/evaluator_outputs/` — older ROC, MSE, and anomaly plot folders.
- `archive/legacy_code/` — the older implementation.
- `archive/legacy_latex/` — superseded LaTeX fragments.
- `archive/dataset_experiments/` — the unrelated ItalyPowerDemand experiment.

Nothing in `archive/` is required to reproduce the current paper figures.
