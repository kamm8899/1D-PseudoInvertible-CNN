# Focused nonlinearity results

Detection probability (%) averaged across three simulation seeds, with 1,000 signal windows per modulation/SNR/condition per seed. Thresholds are recalibrated using independent noise to target a 1% false-alarm rate. Strong combined compression means TX IBO = 0 dB and RX noise-referenced back-off = 0 dB (Rapp smoothness p = 2).

| Modulation | SNR (dB) | Linear CAE Pd (%) | Linear Psi-NN Pd (%) | Strong combined CAE Pd (%) | Strong combined Psi-NN Pd (%) |
| --- | --- | --- | --- | --- | --- |
| 16-QAM | -10 | 30.7 | 75.6 | 29.0 | 71.7 |
| 32-QAM | -10 | 28.2 | 71.5 | 26.3 | 68.4 |
| 16-QAM | -6 | 78.0 | 99.7 | 75.1 | 99.5 |
| 32-QAM | -6 | 75.1 | 99.7 | 73.7 | 99.5 |

Source: metrics.csv in this directory. This is a selected summary; see paired_differences.csv for per-seed confidence intervals and metrics.csv for all conditions and measured false-alarm rates.

Scope: existing paper checkpoints, memoryless AM/AM compression, and equal total post-TX/pre-RX signal-to-noise ratio with TX power loss compensated. No AM/PM distortion or amplifier memory is modeled.
