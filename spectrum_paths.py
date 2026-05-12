"""Where each model loads its test tensors.

PsiNN / Pablos / energy detector — **full test channel only** (pulse shaping + SNR uncertainty):
  - Defaults **require** `spectrum_data/test_data_full.pt` and `test_data_raw_full.pt`.
  - Override with `SPECTRUM_TEST_DATA_PSINN` / `SPECTRUM_TEST_DATA_RAW_PSINN` (or legacy
    `SPECTRUM_TEST_DATA` / `SPECTRUM_TEST_DATA_RAW`). Metadata is checked unless
    `SPECTRUM_SKIP_PSINN_CHANNEL_CHECK=1`.
  - Build the full tensors, e.g.:
      python generate_spectrum_dataset.py --tag full --snr-uncertainty-db 2

CAE — **same full test channel** as PsiNN by default (pulse + SNR uncertainty):
  - Defaults **require** `spectrum_data/test_data_full.pt` and `test_data_raw_full.pt`.
  - For **ablations**, set `SPECTRUM_TEST_DATA_CAE` / `SPECTRUM_TEST_DATA_RAW_CAE` to e.g. `test_data_ideal.pt`.
  - Metadata check: `assert_psinn_full_channel_metadata` (shared with PsiNN; skip with `SPECTRUM_SKIP_PSINN_CHANNEL_CHECK=1`).
"""

from __future__ import annotations

import os
import warnings

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _isfile(path: str) -> bool:
    return os.path.isfile(path)


def get_psinn_test_data_path() -> str:
    """Normalized test set for PsiNN, Pablos, forward eval — **full channel** file only by default."""
    for key in ("SPECTRUM_TEST_DATA_PSINN", "SPECTRUM_TEST_DATA"):
        p = os.environ.get(key)
        if p:
            return p
    full = os.path.join(_REPO_ROOT, "spectrum_data", "test_data_full.pt")
    if not _isfile(full):
        raise FileNotFoundError(
            "PsiNN / Pablos require the **full** test set (pulse shaping + SNR uncertainty). "
            f"Missing: {full}\n"
            "Generate it, for example:\n"
            "  python generate_spectrum_dataset.py --tag full --snr-uncertainty-db 2\n"
            "Or point SPECTRUM_TEST_DATA_PSINN to a .pt built with pulse on and snr_uncertainty_db > 0."
        )
    return full


def get_psinn_test_data_raw_path() -> str:
    """Unnormalized test set for energy detector — **full** raw pair of `test_data_full.pt`."""
    for key in ("SPECTRUM_TEST_DATA_RAW_PSINN", "SPECTRUM_TEST_DATA_RAW"):
        p = os.environ.get(key)
        if p:
            return p
    raw_full = os.path.join(_REPO_ROOT, "spectrum_data", "test_data_raw_full.pt")
    if not _isfile(raw_full):
        raise FileNotFoundError(
            "Energy detector raw test must match the full PsiNN channel. "
            f"Missing: {raw_full}\n"
            "Generate with the same command as test_data_full.pt (writes both).\n"
            "Or set SPECTRUM_TEST_DATA_RAW_PSINN."
        )
    return raw_full


def assert_psinn_full_channel_metadata(test_dict: dict) -> None:
    """If `generation` exists, require pulse shaping and SNR uncertainty > 0 (PsiNN, Pablos, CAE full eval)."""
    if os.environ.get("SPECTRUM_SKIP_PSINN_CHANNEL_CHECK", "").lower() in ("1", "true", "yes"):
        return
    meta = test_dict.get("generation")
    if not meta:
        warnings.warn(
            "Test tensor has no 'generation' field — cannot verify pulse shaping + SNR uncertainty "
            "(PsiNN / Pablos / CAE full channel). Regenerate with generate_spectrum_dataset.py.",
            UserWarning,
            stacklevel=2,
        )
        return
    if not meta.get("use_pulse_shaping"):
        raise ValueError(
            "Full-channel check failed (PsiNN / Pablos / CAE): use_pulse_shaping is False in this test file."
        )
    u = float(meta.get("snr_uncertainty_db", 0) or 0)
    if u <= 0:
        raise ValueError(
            "Full-channel check failed (PsiNN / Pablos / CAE): snr_uncertainty_db must be > 0 "
            f"(got {u}). Regenerate with e.g. --snr-uncertainty-db 2."
        )


def get_cae_test_data_path() -> str:
    """Normalized test set for CAE — defaults to **full** channel (same as PsiNN); override for ablations."""
    p = os.environ.get("SPECTRUM_TEST_DATA_CAE")
    if p:
        return p
    full = os.path.join(_REPO_ROOT, "spectrum_data", "test_data_full.pt")
    if not _isfile(full):
        raise FileNotFoundError(
            "CAE eval defaults to the **full** test set (pulse + SNR uncertainty), same as PsiNN. "
            f"Missing: {full}\n"
            "Generate: python generate_spectrum_dataset.py --tag full --snr-uncertainty-db 2\n"
            "For ablations only, set SPECTRUM_TEST_DATA_CAE to e.g. spectrum_data/test_data_ideal.pt"
        )
    return full


def get_cae_test_data_raw_path() -> str:
    """Unnormalized test for CAE raw-domain tools — defaults to **full** raw pair."""
    p = os.environ.get("SPECTRUM_TEST_DATA_RAW_CAE")
    if p:
        return p
    raw_full = os.path.join(_REPO_ROOT, "spectrum_data", "test_data_raw_full.pt")
    if not _isfile(raw_full):
        raise FileNotFoundError(
            "CAE raw test defaults to test_data_raw_full.pt (matches test_data_full.pt). "
            f"Missing: {raw_full}\n"
            "Generate with the same command as the normalized full set, or set SPECTRUM_TEST_DATA_RAW_CAE."
        )
    return raw_full


def get_test_data_path() -> str:
    """Deprecated alias: same as get_psinn_test_data_path(). Do not use for CAE."""
    return get_psinn_test_data_path()


def get_test_data_raw_path() -> str:
    """Deprecated alias: same as get_psinn_test_data_raw_path()."""
    return get_psinn_test_data_raw_path()
