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
# Reviewer (c): hardware nonlinearity helpers; None settings retain the linear channel.
from rf_nonlinearity import transmitter, receiver

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
    seed: int = 42,
    # Reviewer (c): optional TX/RX compression controls, disabled by default.
    tx_ibo_db: float | None = None,
    rx_backoff_db: float | None = None,
    rapp_p: float = 2.0,
    snr_points=None,
):
    """
    Training set : num_train clean pure-noise samples (normalized); never RX-distorted.
    Nonlinearity : TX Rapp before received-power scaling/AWGN, RX Rapp after AWGN
                   on both hypotheses. SNR refers to post-TX/pre-RX total signal power.
                   No impairment changes RNG draws; seed pairs conditions.
    Test set     : fixed grid — 11 SNR points x 4 modulations x samples_per_mod_per_snr
                   signal samples + noise_per_snr noise samples per SNR point.
                   Guarantees exactly samples_per_mod_per_snr samples at every (mod, SNR) cell.

    use_pulse_shaping  : If False, keep rectangular (unshaped) symbol streams before AWGN.
    snr_uncertainty_db : If > 0, each H1 sample uses AWGN at SNR drawn uniformly in
                         [nominal - u, nominal + u]. Labels/snrs in the returned tensors stay
                         at the nominal grid value so Pd vs nominal SNR stays well-defined.
    """
    if min(num_train, samples_per_mod_per_snr, noise_per_snr) <= 0:
        raise ValueError("Sample counts must be positive")
    if length < 8 or length % 8:
        raise ValueError("length must be a positive multiple of 8")
    if not np.isfinite(snr_uncertainty_db) or snr_uncertainty_db < 0:
        raise ValueError("snr_uncertainty_db must be finite and nonnegative")
    if not np.isfinite(rapp_p) or rapp_p <= 0:
        raise ValueError("rapp_p must be finite and positive")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Training set ────────────────────────────────────────────────────────
    # Reviewer (c): keep training noise clean to test existing frozen models.
    # Receiver-distorted calibration noise is generated in evaluate_nonlinearity.py.
    train_noise_raw = torch.randn(num_train, 2, length, dtype=torch.float32)
    mean            = train_noise_raw.mean(dim=[1, 2], keepdim=True)
    std             = train_noise_raw.std(dim=[1, 2],  keepdim=True)
    train_noise     = (train_noise_raw - mean) / (std + 1e-8)

    # ── Test set — fixed grid ────────────────────────────────────────────────
    # Professor Ask: "Make sure you have enough samples per modulation for each SNR, e.g. 200 samples each."
    # Uses a fixed grid instead of random uniform SNR draws so every (modulation, SNR) cell
    # has exactly samples_per_mod_per_snr samples — including the edge points -10 and +10 dB.
    snr_points = list(snr_points) if snr_points is not None else list(range(-10, 11, 2))
    if not snr_points or not all(np.isfinite(s) for s in snr_points):
        raise ValueError("snr_points must contain finite SNR values")
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
                # Reviewer (c), H1: the PA compresses the pulse-shaped signal
                # before channel noise is added; the RX compresses signal + noise.
                sig = transmitter(sig, tx_ibo_db, rapp_p)
                # Equal post-TX power SNR; received-power scaling compensates PA loss.
                sample_raw = receiver(add_awgn(sig, snr_eff), rx_backoff_db, rapp_p)
                sample     = (sample_raw - sample_raw.mean()) / sample_raw.std()  # normalized
                test_data.append(sample)
                test_data_raw.append(sample_raw)
                test_labels.append(1)
                test_snrs.append(float(snr_db))
                test_mods.append(mod)

        # noise_per_snr noise (H0) samples at this SNR point
        for _ in range(noise_per_snr):
            sample_raw = torch.randn(2, length, dtype=torch.float32)        # unnormalized
            # Reviewer (c), H0: no transmitted signal means no TX distortion,
            # but receiver compression still acts on noise and can change Pfa.
            sample_raw = receiver(sample_raw, rx_backoff_db, rapp_p)
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
        "seed": seed,
        # Reviewer (c): save the operating point and SNR convention for reproducibility.
        "nonlinearity": {"model": "rapp", "tx_ibo_db": tx_ibo_db,
                         "rx_backoff_db": rx_backoff_db, "smoothness": rapp_p,
                         "snr_reference": "post_tx_pre_rx_total_signal_power",
                         "rx_noise_power_per_component": 1.0},
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
    # Reviewer (c): CLI controls for saving separate nonlinear test datasets.
    parser.add_argument("--tx-ibo-db", type=float, default=None)
    parser.add_argument("--rx-backoff-db", type=float, default=None)
    parser.add_argument("--rapp-p", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-cell", type=int, default=200)
    parser.add_argument("--noise-per-snr", type=int, default=200)
    parser.add_argument("--skip-training-save", action="store_true",
                        help="Preserve existing noise training files during robustness generation.")
    args = parser.parse_args()
    if (args.tx_ibo_db is not None or args.rx_backoff_db is not None) and not args.tag:
        parser.error("Nonlinear datasets require --tag to preserve the reference dataset")

    # Keep the original reference datasets available for the linear comparison.
    nonlinear = args.tx_ibo_db is not None or args.rx_backoff_db is not None
    if nonlinear:
        for stem in ("test_data", "test_data_raw"):
            if Path(f"spectrum_data/{stem}_{args.tag}.pt").exists():
                parser.error("Nonlinear generation refuses to overwrite existing test files; choose a new tag")

    use_pulse = not args.no_pulse_shaping
    print("Generating spectrum-sensing dataset (Option 1 - Pure Python)...")
    print(f"   pulse_shaping={use_pulse}  snr_uncertainty_db={args.snr_uncertainty_db}")

    train_noise, train_noise_raw, test_data, test_data_raw, test_labels, test_snrs, test_mods, meta = \
        generate_iq_dataset(
            use_pulse_shaping=use_pulse,
            snr_uncertainty_db=args.snr_uncertainty_db,
            seed=args.seed, tx_ibo_db=args.tx_ibo_db, rx_backoff_db=args.rx_backoff_db,
            rapp_p=args.rapp_p, samples_per_mod_per_snr=args.samples_per_cell,
            noise_per_snr=args.noise_per_snr,
        )

    Path("spectrum_data").mkdir(exist_ok=True)

    # A nonlinear test sweep must not replace the noise used to train checkpoints.
    if not args.skip_training_save and not nonlinear:
        torch.save(train_noise_raw, "spectrum_data/train_noise_raw.pt")
        torch.save(train_noise, "spectrum_data/train_noise.pt")

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
    print(f"   Per (mod, SNR)   : {args.samples_per_cell} signal samples at each of 11 SNR x 4 modulation cells")
    print(f"   Saved            : {test_norm_path} , {test_raw_path}")
    print("   CAE eval         : defaults to test_data_full.pt (set SPECTRUM_TEST_DATA_CAE for ablations)")
    print("   PsiNN/Pablos/ED  : require test_data_full.pt / test_data_raw_full.pt (see spectrum_paths.py)")
