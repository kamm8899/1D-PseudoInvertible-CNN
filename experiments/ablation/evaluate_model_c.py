"""Evaluate decoder-ablation Model C under the corrected paper conditions.

Model C keeps the Pablos-style 1->16->64->128 I-only encoder and uses an
independently parameterized symmetric decoder. This script does not retrain or
replace it with cae_best.pth; it reevaluates the saved Model C checkpoint using
the corrected test set and held-out empirical calibration threshold.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from experiments.ablation.compare_lowcap_convae import LowCapSymDec


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("experiments/ablation/output")
CHECKPOINT = OUTPUT_DIR / "compare_C_lowcap-symdec.pth"
TEST_FILE = Path("spectrum_data/test_data_advisor_sps4.pt")
CALIB_FILE = Path("spectrum_data/calib_noise.pt")
PFA = 0.01


def compute_beta(model, data, batch_size=256):
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = data[start:start + batch_size].to(DEVICE)
            recon = model(x)
            sse = torch.sum((x - recon) ** 2, dim=(1, 2))
            centered = x - x.mean(dim=(1, 2), keepdim=True)
            sst = torch.sum(centered ** 2, dim=(1, 2))
            scores.append((1.0 - sse / (sst + 1e-8)).cpu())
    return torch.cat(scores).numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test = torch.load(TEST_FILE, weights_only=False)
    test_data = test["data"][:, 0:1, :]
    labels = test["labels"].numpy()
    snrs = test["snrs"].numpy().astype(float)
    calib = torch.load(CALIB_FILE, weights_only=False)[:, 0:1, :]

    model = LowCapSymDec().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))

    beta_calib = compute_beta(model, calib)
    beta_test = compute_beta(model, test_data)
    gamma = float(np.quantile(beta_calib, 1.0 - PFA))
    measured_pfa = float(np.mean(beta_test[labels == 0] > gamma))
    snr_points = np.unique(snrs)
    pd = np.array([
        np.mean(beta_test[np.isclose(snrs, s) & (labels == 1)] > gamma)
        for s in snr_points
    ], dtype=float)

    np.save(OUTPUT_DIR / "scores_calib_model_c.npy", beta_calib)
    np.save(OUTPUT_DIR / "scores_test_model_c.npy", beta_test)
    np.save(OUTPUT_DIR / "pd_vs_snr_model_c.npy", pd)
    np.save(OUTPUT_DIR / "snr_points_model_c.npy", snr_points)

    lines = [
        "Model C corrected-condition evaluation",
        f"checkpoint: {CHECKPOINT}",
        f"test: {TEST_FILE}",
        f"calibration: {CALIB_FILE}",
        f"parameters: {sum(p.numel() for p in model.parameters()):,}",
        f"threshold: empirical {1-PFA:.2f} quantile = {gamma:.8f}",
        f"measured test Pfa: {measured_pfa:.6f}",
        "",
        "SNR (dB), Pd",
    ]
    lines += [f"{s:g}, {value:.6f}" for s, value in zip(snr_points, pd)]
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "model_c_corrected_results.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
