'''
Combined Pd vs SNR plot — Three lines: ED, CAE, Psi-NN
Requires the following files to exist in spectrum_data/:
    snr_points.npy
    pd_vs_snr_ed.npy      (from energy_detector.py)
    pd_vs_snr_cae.npy     (from spectrum_data/evaluate_anomalies_cae.py)
    pd_vs_snr_psinn.npy   (from evaluate_anomaly_inverted.py)
'''

import numpy as np
import matplotlib.pyplot as plt

snr_points = np.load("spectrum_data/snr_points.npy")
pd_ed      = np.load("spectrum_data/pd_vs_snr_ed.npy")
pd_cae     = np.load("spectrum_data/pd_vs_snr_cae.npy")
pd_psinn   = np.load("spectrum_data/pd_vs_snr_psinn.npy")

plt.figure(figsize=(9, 6))
plt.plot(snr_points, pd_ed,    marker='o', linewidth=2, label='Energy Detector')
plt.plot(snr_points, pd_cae,   marker='s', linewidth=2, label='CAE (Baseline)')
plt.plot(snr_points, pd_psinn, marker='^', linewidth=2, label='Psi-NN')
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
