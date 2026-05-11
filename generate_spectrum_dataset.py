'''
Option 1 - Pure Python Synthetic Generator
For 1D Psl-CNN Spectrum Sensing Anomaly Detection
Author: Jessica Kamman
Date: April 2026
'''

import torch
import numpy as np
from pathlib import Path

def add_awgn(signal: torch.Tensor, snr_db: float, noise_floor: float = 1.0) -> torch.Tensor:
    """Add AWGN with fixed noise floor matching training noise (power = noise_floor = 1.0).
    SNR = Psignal / Pnoise per professor guidance — signal rescaled, noise floor fixed."""
    target_signal_power  = noise_floor * (10 ** (snr_db / 10.0))
    current_signal_power = torch.mean(signal ** 2)
    scaled_signal = signal * torch.sqrt(target_signal_power / (current_signal_power + 1e-8))
    noise = torch.sqrt(torch.tensor(noise_floor)) * torch.randn_like(signal)  # power = 1.0, matches training noise
    return scaled_signal + noise

# Original relative-noise version (kept for reference — do not use):
# def add_awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
#     signal_power = torch.mean(signal ** 2)
#     noise_power  = signal_power / (10 ** (snr_db / 10.0))
#     noise = torch.sqrt(noise_power / 2.0) * torch.randn_like(signal)
#     return signal + noise

def _make_signal(mod: str, length: int, _32qam_re, _32qam_im) -> torch.Tensor:
    """Generate one normalized IQ signal for the given modulation."""
    if mod == 'qpsk':
        symbols = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])[torch.randint(0, 4, (length//8,))]
        signal  = torch.repeat_interleave(symbols, 8)
    elif mod == 'bpsk':
        symbols = torch.tensor([1, -1])[torch.randint(0, 2, (length//4,))]
        signal  = torch.repeat_interleave(symbols, 4) + 0j
    elif mod == '16qam':
        re = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
        im = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
        signal = torch.repeat_interleave(re + 1j * im, 4)
    else:  # 32-QAM cross constellation
        idx    = torch.randint(0, 32, (length//4,))
        signal = torch.repeat_interleave(_32qam_re[idx] + 1j * _32qam_im[idx], 4)

    iq = torch.stack([signal.real, signal.imag], dim=0).squeeze(1)
    return iq / torch.abs(iq).max()


def generate_iq_dataset(
    num_train:               int = 20000,
    length:                  int = 1024,
    samples_per_mod_per_snr: int = 200,
    noise_per_snr:           int = 200,
):
    """
    Training set : num_train pure-noise samples (normalized).
    Test set     : fixed grid — 11 SNR points x 4 modulations x samples_per_mod_per_snr
                   signal samples + noise_per_snr noise samples per SNR point.
                   Guarantees exactly samples_per_mod_per_snr samples at every (mod, SNR) cell.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Training set ────────────────────────────────────────────────────────
    train_noise_raw = torch.randn(num_train, 2, length, dtype=torch.float32)
    mean            = train_noise_raw.mean(dim=[1, 2], keepdim=True)
    std             = train_noise_raw.std(dim=[1, 2],  keepdim=True)
    train_noise     = (train_noise_raw - mean) / (std + 1e-8)

    # ── Test set — fixed grid ────────────────────────────────────────────────
    # Professor Ask: "Make sure you have enough samples per modulation for each SNR, e.g. 200 samples each."
    # Uses a fixed grid instead of random uniform SNR draws so every (modulation, SNR) cell
    # has exactly samples_per_mod_per_snr samples — including the edge points -10 and +10 dB.
    snr_points  = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    modulations = ['qpsk', 'bpsk', '16qam', '32qam']

    _32qam_re = torch.tensor(
        [-5,-3,-1,1,3,5, -5,-3,-1,1,3,5, -5,-3,-1,1,3,5,
         -5,-3,-1,1,3,5, -3,-1,1,3,       -3,-1,1,3],
        dtype=torch.float32)
    _32qam_im = torch.tensor(
        [-5,-5,-5,-5,-5,-5, -3,-3,-3,-3,-3,-3, -1,-1,-1,-1,-1,-1,
          1, 1, 1, 1, 1, 1,  3, 3, 3, 3,        5, 5, 5, 5],
        dtype=torch.float32)

    test_data, test_data_raw, test_labels, test_snrs, test_mods = [], [], [], [], []

    for snr_db in snr_points:
        # Exactly samples_per_mod_per_snr signal samples per modulation at this SNR
        for mod in modulations:
            for _ in range(samples_per_mod_per_snr):
                sig        = _make_signal(mod, length, _32qam_re, _32qam_im)
                sample_raw = add_awgn(sig, snr_db)                          # unnormalized
                sample     = (sample_raw - sample_raw.mean()) / sample_raw.std()  # normalized
                test_data.append(sample)
                test_data_raw.append(sample_raw)
                test_labels.append(1)
                test_snrs.append(float(snr_db))
                test_mods.append(mod)

        # noise_per_snr noise (H0) samples at this SNR point
        for _ in range(noise_per_snr):
            sample_raw = torch.randn(2, length, dtype=torch.float32)        # unnormalized
            sample     = (sample_raw - sample_raw.mean()) / sample_raw.std()  # normalized
            test_data.append(sample)
            test_data_raw.append(sample_raw)
            test_labels.append(0)
            test_snrs.append(float(snr_db))
            test_mods.append('none')

    test_data     = torch.stack(test_data)
    test_data_raw = torch.stack(test_data_raw)
    test_labels   = torch.tensor(test_labels, dtype=torch.long)
    test_snrs     = torch.tensor(test_snrs,   dtype=torch.float32)

    return train_noise, train_noise_raw, test_data, test_data_raw, test_labels, test_snrs, test_mods


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating spectrum-sensing dataset (Option 1 - Pure Python)...")

    train_noise, train_noise_raw, test_data, test_data_raw, test_labels, test_snrs, test_mods = \
        generate_iq_dataset()

    Path("spectrum_data").mkdir(exist_ok=True)

    torch.save(train_noise_raw, "spectrum_data/train_noise_raw.pt")   # unnormalized — Energy Detector H0
    torch.save(train_noise,     "spectrum_data/train_noise.pt")        # normalized   — PsiNN / CAE
    torch.save({
        "data":    test_data,
        "labels":  test_labels,
        "snrs":    test_snrs,
        "signals": test_mods,
    }, "spectrum_data/test_data.pt")
    torch.save({
        "data":    test_data_raw,
        "labels":  test_labels,
        "snrs":    test_snrs,
        "signals": test_mods,
    }, "spectrum_data/test_data_raw.pt")                               # unnormalized — Energy Detector

    n_signal = test_labels.sum().item()
    n_noise  = (test_labels == 0).sum().item()
    print("Dataset generation complete!")
    print(f"   Training samples : {train_noise.shape}  (pure noise only)")
    print(f"   Test samples     : {test_data.shape}  ({n_signal} signal, {n_noise} noise)")
    print(f"   Per (mod, SNR)   : 200 signal samples at each of 11 SNR x 4 modulation cells")
    print(f"   Files saved in   : ./spectrum_data/")
