'''
Option 1 - Pure Python Synthetic Generator
For 1D Psl-CNN Spectrum Sensing Anomaly Detection
Author: Jessica Kamman
Date: April 2026
'''

import torch
import numpy as np
from pathlib import Path

def add_awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add real AWGN noise to a 2-channel (I/Q) float signal with exact target SNR."""
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.sqrt(noise_power / 2.0) * torch.randn_like(signal)
    return signal + noise

def generate_iq_dataset(
    num_train: int = 20000,
    num_test: int = 5000,
    length: int = 1024,
    snr_range: tuple = (-10, 10)
):
    """Generate training (pure noise) and test (noise + modulated signals) datasets."""
    np.random.seed(42)
    torch.manual_seed(42)

    # === TRAINING SET: Pure AWGN noise only ===
    
    train_noise = torch.randn(num_train, 2, length, dtype=torch.float32)
    train_noise = (train_noise - train_noise.mean()) / train_noise.std()

    # === TEST SET: Noise + various modulations (modulation-agnostic) ===
    test_data = []
    test_labels = []   # 0 = pure noise, 1 = signal present (anomaly)
    test_snrs = []

    modulations = ['qpsk', 'bpsk', '16qam', 'fm']

    for _ in range(num_test):
        snr_db = np.random.uniform(*snr_range)

        if np.random.rand() < 0.5:  # 50% pure noise
            sample = torch.randn(2, length, dtype=torch.float32)
            label = 0
        else:
            t = torch.arange(length, dtype=torch.float32)
            mod = np.random.choice(modulations)

            if mod == 'qpsk':
                symbols = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])[torch.randint(0, 4, (length//8,))]
                signal = torch.repeat_interleave(symbols, 8)
            elif mod == 'bpsk':
                symbols = torch.tensor([1, -1])[torch.randint(0, 2, (length//4,))]
                signal = torch.repeat_interleave(symbols, 4) + 0j
            elif mod == '16qam':
                re = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
                im = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
                symbols = re + 1j * im
                signal = torch.repeat_interleave(symbols, 4)
            else:  # FM
                freq = 0.1 * torch.sin(2 * np.pi * 0.01 * t)
                signal = torch.cos(2 * np.pi * freq * t + torch.cumsum(0.05 * freq, dim=0)) + 0j

            # Normalize signal and add controlled noise
            signal = torch.stack([signal.real, signal.imag], dim=0).squeeze(1)
            signal = signal / torch.abs(signal).max()
            sample = add_awgn(signal, snr_db)
            label = 1

        # Normalize sample
        sample = (sample - sample.mean()) / sample.std()
        test_data.append(sample)
        test_labels.append(label)
        test_snrs.append(snr_db)

    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_snrs = torch.tensor(test_snrs, dtype=torch.float32)

    return train_noise, test_data, test_labels, test_snrs

# ========================== MAIN ==========================
if __name__ == "__main__":
    print("Generating spectrum-sensing dataset (Option 1 - Pure Python)...")
    
    train_noise, test_data, test_labels, test_snrs = generate_iq_dataset()

    # Create output folder
    Path("spectrum_data").mkdir(exist_ok=True)

    # Save files
    torch.save(train_noise, "spectrum_data/train_noise.pt")
    torch.save({
        "data": test_data,
        "labels": test_labels,      # 0 = noise, 1 = anomaly
        "snrs": test_snrs
    }, "spectrum_data/test_data.pt")

    print("✅ Dataset generation complete!")
    print(f"   Training samples : {train_noise.shape}  (pure noise only)")
    print(f"   Test samples     : {test_data.shape}  ({test_labels.sum().item()} anomalies)")
    print(f"   Files saved in   : ./spectrum_data/")