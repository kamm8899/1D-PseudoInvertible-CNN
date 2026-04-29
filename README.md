# Modulation-Agnostic Spectrum Sensing Anomaly Detection
### 1D Pseudo-Invertible Neural Network (Psi-NN) vs. Baseline CAE

Implementation of the anomaly detection framework from:
> Pablos et al. (2022). *Modulation-agnostic spectrum sensing using deep learning.* ICT Express.
> https://www.sciencedirect.com/science/article/pii/S2405959522000480

---

## Overview

This project detects the presence of modulated radio signals (anomalies) in a channel that is assumed to contain only AWGN noise under the null hypothesis H₀. Two autoencoder architectures are compared:

- **Psi-CNN** — 1D autoencoder built with Pseudo-Invertible Neural Network (PsiNN) layers, whose encoder and decoder share weights via right-inverse weight tying.
- **Baseline** — Standard 1D convolutional autoencoder (CAE) with separately parameterized encoder and decoder.

Anomaly detection follows **Section 3.3 (Neyman-Pearson criterion)**:
1. Train autoencoders on noise-only data (H₀).
2. Compute the **Coefficient of Determination β** per sample as the reconstruction quality score.
3. Estimate a **CFAR threshold γ** from noise training statistics at a target false-alarm rate P_FA.
4. Declare a signal present if β < γ.
5. Evaluate via ROC curve, AUC, and TPR at γ broken down by SNR range and modulation type.

---

## Project Structure

```
Research_PSNN/
├── psinn_layer_1d.py               # PsiNNConv1d layer + AE_Classifier1d + AE_Baseline_Classifier1d
├── psinn_layer_and_autoencoder.py  # Original 2D PsiNN layer (reference)
├── cae_spectrum.py                 # Standalone CAE architecture (Section 3.2)
├── generate_spectrum_dataset.py    # Synthetic I/Q dataset generator
├── train_models.py                 # Training loop for Psi-CNN and Baseline
├── evaluate_anomaly.py             # Evaluation scaffold (Section 3.3)
├── evaluate_anomaly_inverted.py    # Full evaluation — Psi-NN inverted architecture
├── load_data.py                    # Data loading utilities
├── diagnostic_recon_error.py       # Reconstruction error diagnostics
│
├── spectrum_data/                  # Generated data and saved model weights
│   ├── train_noise.pt              # Pure AWGN training set  (20 000 × 2 × 1024)
│   ├── test_data.pt                # Mixed test set          (5 000 × 2 × 1024)
│   ├── psl_cnn_200epochs.pth       # Trained Psi-CNN weights
│   ├── baseline_200epochs.pth      # Trained Baseline weights
│   ├── evaluate_anomalies_forward.py       # Evaluation — forward architecture
│   ├── evaluate_anomalies_forward_cae.py   # Evaluation — forward CAE
│   └── evaluate_anomalies_inverted_cae.py  # Evaluation — inverted CAE
│
└── anomalies_Psi-NN_inverted/      # Output plots (auto-created on evaluation run)
    ├── roc_comparison.png
    ├── beta_distribution_psl_cnn_inverted.png
    ├── mse_distribution_psl_cnn_inverted.png
    ├── beta_distribution_baseline_inverted.png
    └── mse_distribution_baseline_inverted.png
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install torch numpy scikit-learn scipy matplotlib
```

### 2. Generate the dataset
```bash
python generate_spectrum_dataset.py
```
Produces `spectrum_data/train_noise.pt` (20 000 pure-noise I/Q samples) and `spectrum_data/test_data.pt` (5 000 samples: 50% noise, 50% modulated signals across QPSK, BPSK, 16-QAM, FM).

### 3. Train the models
```bash
python train_models.py
```
Trains both Psi-CNN and Baseline autoencoders for 200 epochs on noise-only data. Saves weights to `spectrum_data/`.

### 4. Run evaluation
```bash
python evaluate_anomaly_inverted.py
```
Outputs results to the terminal and saves plots to `anomalies_Psi-NN_inverted/`.

---

## Section 3.3 — Neyman-Pearson Detection

**β score (Coefficient of Determination):**

$$\beta = 1 - \frac{\sum(X_i - \hat{X}_i)^2}{\sum(X_i - \bar{X})^2}$$

β ≈ 1 → good reconstruction (noise / H₀)  
β << 1 → poor reconstruction (signal present / H₁)

**CFAR threshold γ:**

Under H₀, β ~ Gaussian(μ_e, σ_e). Given target false-alarm rate P_FA:

$$\gamma = \mu_e + \sigma_e \cdot Q^{-1}(P_{FA})$$

Detection rule: **anomaly if β < γ**

Two P_FA values are evaluated: **0.01** and **0.05**, with results broken down by SNR range ([-10, -5), [-5, 0), [0, 5), [5, 10) dB) and modulation type (QPSK, BPSK, 16-QAM, FM).

---

## Data

| Split | Samples | Content |
|-------|---------|---------|
| Train | 20 000 | Pure AWGN (H₀ only) |
| Test  | 5 000  | 50% noise, 50% modulated signals |

Modulations: QPSK, BPSK, 16-QAM, FM  
SNR range: −10 to +10 dB (uniform random)  
Signal format: 2-channel I/Q, length 1024

---

## Models

| Model | Architecture | Parameters | Weight sharing |
|-------|-------------|------------|----------------|
| Psi-CNN | PsiNNConv1d × 4 (encoder = decoder via right-inverse) | ~fewer | Yes |
| Baseline | Conv1d × 4 encoder + ConvTranspose1d × 4 decoder | ~more | No |

Both use `nf=16`, `k=3`, dropout=0.3, trained with MSE loss on noise reconstruction.
