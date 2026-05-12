'''
Combined Pd vs SNR plot — ED, CAE, Psi-NN, Baseline AE, Pablos (PsiNN-CAE)

Professor Ask: "Redo experiments for specific SNRs: -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10
so we can plot Pd vs SNR for Pfa = 0.01 — Three lines: ED, baseline CAE, Psi-NN (similar to the paper)."

Run order — all must be run before this file:
    energy_detector.py                        → pd_vs_snr_ed.npy
    spectrum_data/evaluate_anomalies_cae.py   → pd_vs_snr_cae.npy
    evaluate_anomaly_inverted.py              → pd_vs_snr_psinn.npy, pd_vs_snr_baseline.npy
    evaluate_pablos.py                        → pd_vs_snr_pablos.npy
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

snr_points   = np.load("spectrum_data/snr_points.npy")
pd_ed        = np.load("spectrum_data/pd_vs_snr_ed.npy")
pd_cae       = np.load("spectrum_data/pd_vs_snr_cae.npy")
pd_psinn     = np.load("spectrum_data/pd_vs_snr_psinn.npy")
pd_baseline  = np.load("spectrum_data/pd_vs_snr_baseline.npy")

pablos_path = Path("spectrum_data/pd_vs_snr_pablos.npy")
pd_pablos   = np.load(pablos_path) if pablos_path.exists() else None

plt.figure(figsize=(9, 6))
#plt.plot(snr_points, pd_ed,       marker='o', linewidth=2, label='Energy Detector')
plt.plot(snr_points, pd_cae,      marker='s', linewidth=2, label='CAE')
plt.plot(snr_points, pd_psinn,    marker='^', linewidth=2, label='Psi-NN')
plt.plot(snr_points, pd_baseline, marker='D', linewidth=2, label='Baseline AE')
if pd_pablos is not None:
    plt.plot(snr_points, pd_pablos, marker='x', linewidth=2, label='Pablos PsiNN')
plt.xlabel('SNR (dB)')
plt.ylabel('Pd  (Probability of Detection)')
plt.title('Pd vs SNR  —  Pfa = 0.01')
plt.legend()
plt.grid(True)
plt.xticks(snr_points)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("pd_vs_snr_combined.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved pd_vs_snr_combined.png")
