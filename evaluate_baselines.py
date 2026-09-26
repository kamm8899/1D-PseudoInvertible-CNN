"""
Classical blind-sensing baselines on the SAME saved data as the Psi-NN / CAE evaluators.

Detectors (all on the I channel, matching the paper models' input):
  - MME  : maximum-minimum eigenvalue ratio of the L x L lag covariance (Zeng & Liang, 2009)
  - CAV  : covariance absolute value (Zeng & Liang, 2009)
  - ED   : energy detector with known noise power (threshold from held-out calibration noise)
  - ED-u : energy detector whose threshold is designed for +/-u dB noise-power
           uncertainty (worst case), separate from the test set's SNR uncertainty

Thresholds: detector-specific empirical (1 - Pfa) quantiles on held-out noise.
The nominal target is shared with the paper models, not the numerical thresholds.
ED-u raises its threshold for worst-case noise power; its nominal-noise test Pfa
can be substantially below the target. Measured test Pfa is reported for each.

Usage:
  python make_calibration_noise.py                               # once: held-out noise
  python evaluate_baselines.py --save-scores                     # default full test set
  python evaluate_baselines.py --test spectrum_data/test_data_raw_sps8.pt --tag sps8

Outputs (spectrum_data/): pd_vs_snr_<det>[_tag].npy, baselines_results[_tag].txt
"""
import argparse
import csv
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


# ── Helpers usable from the Psi-NN / CAE evaluators too ──────────────────────────
def empirical_threshold(calib_scores, pfa=0.01, upper=True):
    """Empirical held-out quantile; independent test Pfa must still be measured."""
    calib_scores = np.asarray(calib_scores)
    return np.quantile(calib_scores, 1.0 - pfa if upper else pfa)


def joint_normalize(x):
    """Per-block joint I/Q normalization, identical to generate_spectrum_dataset.py."""
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), ddof=1, keepdims=True)
    return (x - m) / (s + 1e-8)


def lag_autocorr(x, L):
    """x: (N, n) real. Returns (N, L) biased autocorrelation estimates r_0..r_{L-1}."""
    n = x.shape[1]
    return np.stack([np.sum(x[:, :n - k] * x[:, k:], axis=1) / n for k in range(L)], axis=1)


def mme_stat(x, L=8):
    r = lag_autocorr(x, L)
    idx = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    ev = np.linalg.eigvalsh(r[:, idx])                  # (N, L), ascending
    return ev[:, -1] / ev[:, 0]


def cav_stat(x, L=8):
    r = lag_autocorr(x, L)
    k = np.arange(1, L)
    total = L * np.abs(r[:, 0]) + 2.0 * np.sum((L - k)[None, :] * np.abs(r[:, 1:]), axis=1)
    return total / (L * r[:, 0])


def energy_stat(x):
    return np.sum(x ** 2, axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="spectrum_data/calib_noise_raw.pt",
                    help="HELD-OUT noise for thresholds (make_calibration_noise.py), not the training noise")
    ap.add_argument("--test", default="spectrum_data/test_data_raw_full.pt")
    ap.add_argument("--pfa", type=float, default=0.01)
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--noise-unc-db", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-scores", action="store_true",
                    help="save per-block calibration/test scores for bootstrap_ci.py")
    args = ap.parse_args()

    calib_raw = torch.load(args.calib, weights_only=False).numpy()          # (N, 2, 1024) held-out noise
    d = torch.load(args.test, weights_only=False)
    test_raw = d["data"].numpy()
    labels = d["labels"].numpy()
    snrs = d["snrs"].numpy()
    mods = np.array(d["signals"])
    snr_points = np.unique(snrs)
    print("generation metadata:", d.get("generation"))

    # Same I-channel input as the paper models (joint I/Q normalization, then slice I).
    tr_i, te_i = joint_normalize(calib_raw)[:, 0], joint_normalize(test_raw)[:, 0]
    # ED uses raw (unnormalized) I channel.
    tr_e, te_e = energy_stat(calib_raw[:, 0]), energy_stat(test_raw[:, 0])

    dets = {}
    for L in args.L:
        dets[f"mme_L{L}"] = (mme_stat(tr_i, L), mme_stat(te_i, L), 1.0)
        dets[f"cav_L{L}"] = (cav_stat(tr_i, L), cav_stat(te_i, L), 1.0)
    dets["ed"] = (tr_e, te_e, 1.0)
    # Worst-case design: threshold set for noise power +u dB above nominal (energy scales
    # linearly with noise power), evaluated at nominal noise.
    dets[f"ed_unc{args.noise_unc_db:g}dB"] = (tr_e, te_e, 10 ** (args.noise_unc_db / 10))

    suffix = f"_{args.tag}" if args.tag else ""
    if args.save_scores:
        for name, (tr, te, scale) in dets.items():
            np.save(f"spectrum_data/scores_calib_{name}{suffix}.npy", tr * scale)
            np.save(f"spectrum_data/scores_test_{name}{suffix}.npy", te)
    rows = []
    lines = [f"test file: {args.test}", f"calibration file: {args.calib}",
             f"generation: {d.get('generation')}", f"target Pfa: {args.pfa}", ""]
    lines.insert(4, "Test-set +/-dB variation: SNR uncertainty with fixed noise floor")
    lines.insert(5, f"ED uncertainty curve: +/-{args.noise_unc_db:g} dB noise-power uncertainty")
    hdr = f"{'detector':16}" + "".join(f"{s:>8.0f}" for s in snr_points) + "   meas.Pfa"
    lines.append(hdr)
    for name, (tr, te, scale) in dets.items():
        g = empirical_threshold(tr, args.pfa) * scale
        pfa_meas = np.mean(te[labels == 0] > g)
        pd = np.array([np.mean(te[(snrs == s) & (labels == 1)] > g) for s in snr_points])
        # Advisor item 3: retain the actual operating point alongside detection rates.
        for snr, value in zip(snr_points, pd):
            rows.append(dict(detector=name, snr_db=float(snr), pd=float(value),
                             measured_pfa=float(pfa_meas), target_pfa=args.pfa,
                             threshold=float(g), n_signal=int(np.sum((snrs == snr) & (labels == 1))),
                             n_noise=int(np.sum(labels == 0))))
        np.save(f"spectrum_data/pd_vs_snr_{name}{suffix}.npy", pd)
        lines.append(f"{name:16}" + "".join(f"{p:8.3f}" for p in pd) + f"   {pfa_meas:.4f}")
    # Per-modulation detail at the lowest SNR
    s0 = snr_points.min()
    lines += ["", f"Per-modulation Pd at {s0:.0f} dB:"]
    for name, (tr, te, scale) in dets.items():
        g = empirical_threshold(tr, args.pfa) * scale
        per = {m: np.mean(te[(snrs == s0) & (mods == m)] > g)
               for m in ["qpsk", "bpsk", "16qam", "32qam"] if np.any(mods == m)}
        lines.append(f"  {name:16}" + "  ".join(f"{m}={v:.3f}" for m, v in per.items()))

    np.save(f"spectrum_data/snr_points{suffix}.npy", snr_points.astype(float))
    with open(f"spectrum_data/baselines_results{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = "\n".join(lines)
    print(out)
    with open(f"spectrum_data/baselines_results{suffix}.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
