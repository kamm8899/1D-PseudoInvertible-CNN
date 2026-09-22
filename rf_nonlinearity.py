"""Memoryless complex-envelope Rapp compression (unity small-signal gain).

TX saturation is set by input back-off; RX saturation is fixed relative to
complex noise power. No AM/PM distortion or amplifier memory is modeled.
"""
import math

import torch


# Reviewer (c): model hardware AM/AM compression independently of neural-network
# activations. This helper is shared by TX, RX, and receiver-noise calibration.
def rapp(iq: torch.Tensor, saturation: float, smoothness: float = 2.0) -> torch.Tensor:
    """Compress (..., 2, time) I/Q jointly, preserving the complex phase."""
    if iq.ndim < 2 or iq.shape[-2] != 2:
        raise ValueError("Expected (..., 2, time) I/Q")
    if not math.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be positive and finite")
    if not math.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("smoothness must be positive and finite")
    # |I+jQ| = sqrt(I^2 + Q^2); one shared gain preserves the complex phase.
    amplitude = torch.linalg.vector_norm(iq, dim=-2, keepdim=True)
    # log-domain evaluation avoids overflow for large amplitude/back-off ratios.
    log_ratio = torch.log(amplitude / saturation)
    gain = torch.exp(-torch.nn.functional.softplus(2 * smoothness * log_ratio)
                     / (2 * smoothness))
    return iq * gain


def transmitter(iq: torch.Tensor, ibo_db: float | None, smoothness: float = 2.0):
    """Per-waveform IBO = 10 log10(A_sat^2 / mean(|x|^2))."""
    if ibo_db is None:
        return iq
    if not math.isfinite(ibo_db):
        raise ValueError("ibo_db must be finite or None (linear)")
    # Use complex mean power, not separate component peaks, to define TX IBO.
    power = iq.square().sum(dim=-2).mean().item()
    if power == 0:
        return iq
    return rapp(iq, math.sqrt(power * 10 ** (ibo_db / 10)), smoothness)


def receiver(iq: torch.Tensor, backoff_db: float | None, smoothness: float = 2.0,
             noise_power_per_component: float = 1.0):
    """Fixed A_sat relative to E[|n|^2]=2*noise_power_per_component, not sample RMS."""
    if backoff_db is None:
        return iq
    if not math.isfinite(backoff_db) or noise_power_per_component <= 0:
        raise ValueError("Receiver back-off must be finite and noise power positive")
    # Fixed across SNRs and both hypotheses: deriving this from each received
    # window's RMS would change the receiver operating point with the input.
    saturation = math.sqrt(2 * noise_power_per_component * 10 ** (backoff_db / 10))
    return rapp(iq, saturation, smoothness)
