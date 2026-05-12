'''
Option 1 - Pure Python Synthetic Generator
For 1D Psl-CNN Spectrum Sensing Anomaly Detection
Author: Jessica Kamman
Date: April 2026
'''

import argparse
import os

# macOS / conda: multiple OpenMP runtimes (PyTorch + SciPy) can abort without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from pathlib import Path
from scipy.signal import lfilter

def _actual_snr_db(nominal_snr_db: float, uncertainty_db: float) -> float:
    """If uncertainty_db > 0, draw SNR uniformly in [nominal - u, nominal + u] (noise uncertainty)."""
    if uncertainty_db <= 0:
        return float(nominal_snr_db)
    delta = (torch.rand(()).item() * 2.0 - 1.0) * uncertainty_db
    return float(nominal_snr_db + delta)


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

def _make_signal(
    mod: str,
    length: int,
    _32qam_re,
    _32qam_im,
    use_pulse_shaping: bool = True,
) -> torch.Tensor:
    """Generate one normalized IQ signal for the given modulation."""
    if mod == 'qpsk':
        symbols = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j])[torch.randint(0, 4, (length//8,))]
        signal  = torch.repeat_interleave(symbols, 8)
        samples_per_symbol = 8
    elif mod == 'bpsk':
        symbols = torch.tensor([1, -1])[torch.randint(0, 2, (length//4,))]
        signal  = torch.repeat_interleave(symbols, 4) + 0j
        samples_per_symbol = 4
    elif mod == '16qam':
        re = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
        im = torch.tensor([-3, -1, 1, 3])[torch.randint(0, 4, (length//4,))]
        signal = torch.repeat_interleave(re + 1j * im, 4)
        samples_per_symbol = 4
    else:  # 32-QAM cross constellation
        idx    = torch.randint(0, 32, (length//4,))
        signal = torch.repeat_interleave(_32qam_re[idx] + 1j * _32qam_im[idx], 4)
        samples_per_symbol = 4

    iq = torch.stack([signal.real, signal.imag], dim=0).squeeze(1)

    if use_pulse_shaping:
        # Raised cosine pulse shaping — roll-off 0.35, added per professor guidance
        num_taps = 8 * samples_per_symbol + 1
        t        = np.arange(num_taps) - num_taps // 2
        alpha    = 0.35
        T        = samples_per_symbol
        rc       = np.sinc(t / T) * np.cos(np.pi * alpha * t / T) / (1 - (2 * alpha * t / T) ** 2 + 1e-8)
        rc      /= rc.sum()
        signal_i = lfilter(rc, 1.0, iq[0].numpy())
        signal_q = lfilter(rc, 1.0, iq[1].numpy())
        iq       = torch.tensor(np.stack([signal_i, signal_q]), dtype=torch.float32)

    return iq / torch.abs(iq).max()


def generate_iq_dataset(
    num_train:               int = 20000,
    length:                  int = 1024,
    samples_per_mod_per_snr: int = 200,
    noise_per_snr:           int = 200,
    use_pulse_shaping:       bool = True,
    snr_uncertainty_db:      float = 0.0,
):
    """
    Training set : num_train pure-noise samples (normalized).
    Test set     : fixed grid — 11 SNR points x 4 modulations x samples_per_mod_per_snr
                   signal samples + noise_per_snr noise samples per SNR point.
                   Guarantees exactly samples_per_mod_per_snr samples at every (mod, SNR) cell.

    use_pulse_shaping  : If False, keep rectangular (unshaped) symbol streams before AWGN.
    snr_uncertainty_db : If > 0, each H1 sample uses AWGN at SNR drawn uniformly in
                         [nominal - u, nominal + u]. Labels/snrs in the returned tensors stay
                         at the nominal grid value so Pd vs nominal SNR stays well-defined.
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
                sig        = _make_signal(mod, length, _32qam_re, _32qam_im, use_pulse_shaping)
                snr_eff    = _actual_snr_db(snr_db, snr_uncertainty_db)
                sample_raw = add_awgn(sig, snr_eff)                         # unnormalized
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

    meta = {
        "use_pulse_shaping":  use_pulse_shaping,
        "snr_uncertainty_db": float(snr_uncertainty_db),
    }

    return train_noise, train_noise_raw, test_data, test_data_raw, test_labels, test_snrs, test_mods, meta


def pack_test_tensors_for_save(test_data, test_data_raw, test_labels, test_snrs, test_mods, meta):
    common = {
        "labels":  test_labels,
        "snrs":    test_snrs,
        "signals": test_mods,
        "generation": meta,
    }
    return (
        {**common, "data": test_data},
        {**common, "data": test_data_raw},
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spectrum_data tensors. Use --tag to keep several ablations side by side.",
        epilog=(
            "PsiNN / Pablos / ED **require** spectrum_data/test_data_full.pt (pulse + uncertainty). "
            "Create it with e.g. --tag full --snr-uncertainty-db 2 (omit --no-pulse-shaping). "
            "CAE eval defaults to spectrum_data/test_data_full.pt (same as PsiNN); "
            "set SPECTRUM_TEST_DATA_CAE for ablation files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default="",
        help="If set, writes test_data_<tag>.pt / test_data_raw_<tag>.pt; if empty, default names.",
    )
    parser.add_argument(
        "--no-pulse-shaping",
        action="store_true",
        help="Rectangular symbols only (ablation: turn off raised-cosine shaping).",
    )
    parser.add_argument(
        "--snr-uncertainty-db",
        type=float,
        default=0.0,
        help="Half-width (dB) for uniform SNR jitter around each nominal test SNR on H1 samples.",
    )
    args = parser.parse_args()

    use_pulse = not args.no_pulse_shaping
    print("Generating spectrum-sensing dataset (Option 1 - Pure Python)...")
    print(f"   pulse_shaping={use_pulse}  snr_uncertainty_db={args.snr_uncertainty_db}")

    train_noise, train_noise_raw, test_data, test_data_raw, test_labels, test_snrs, test_mods, meta = \
        generate_iq_dataset(
            use_pulse_shaping=use_pulse,
            snr_uncertainty_db=args.snr_uncertainty_db,
        )

    Path("spectrum_data").mkdir(exist_ok=True)

    torch.save(train_noise_raw, "spectrum_data/train_noise_raw.pt")   # unnormalized — Energy Detector H0
    torch.save(train_noise,     "spectrum_data/train_noise.pt")        # normalized   — PsiNN / CAE

    suffix = f"_{args.tag}" if args.tag else ""
    test_norm_path = f"spectrum_data/test_data{suffix}.pt"
    test_raw_path  = f"spectrum_data/test_data_raw{suffix}.pt"

    d_norm, d_raw = pack_test_tensors_for_save(
        test_data, test_data_raw, test_labels, test_snrs, test_mods, meta,
    )
    torch.save(d_norm, test_norm_path)
    torch.save(d_raw,  test_raw_path)

    n_signal = test_labels.sum().item()
    n_noise  = (test_labels == 0).sum().item()
    print("Dataset generation complete!")
    print(f"   Training samples : {train_noise.shape}  (pure noise only)")
    print(f"   Test samples     : {test_data.shape}  ({n_signal} signal, {n_noise} noise)")
    print(f"   Per (mod, SNR)   : 200 signal samples at each of 11 SNR x 4 modulation cells")
    print(f"   Saved            : {test_norm_path} , {test_raw_path}")
    print("   CAE eval         : defaults to test_data_full.pt (set SPECTRUM_TEST_DATA_CAE for ablations)")
    print("   PsiNN/Pablos/ED  : require test_data_full.pt / test_data_raw_full.pt (see spectrum_paths.py)")
