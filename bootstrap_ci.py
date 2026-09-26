"""
Paired, modulation-stratified bootstrap confidence intervals for P_d(A) - P_d(B).

Inputs are per-block scores (higher = more signal-like) that each evaluator saves:
  --a-calib / --b-calib : scores on the held-out calibration noise (calib_noise*.pt)
  --a-test  / --b-test  : scores on every block of the test set, in test-file order
  --test                : the test .pt file (for labels, SNRs, modulations)

For Psi-NN / CAE add two lines to the evaluator, e.g.
    np.save("spectrum_data/scores_calib_psinn.npy", beta_calib)
    np.save("spectrum_data/scores_test_psinn.npy",  beta_test)
(For a lower-tail detector, save the NEGATED beta so that higher = signal.)
evaluate_baselines.py --save-scores writes the MME / CAV / ED files.

Example:
  python bootstrap_ci.py --a-calib spectrum_data/scores_calib_psinn.npy --a-test spectrum_data/scores_test_psinn.npy \
                         --b-calib spectrum_data/scores_calib_cav_L4.npy --b-test spectrum_data/scores_test_cav_L4.npy \
                         --names Psi-NN CAV
"""
import argparse
import numpy as np


def paired_bootstrap(cal_a, cal_b, sig_a, sig_b, mods, pfa=0.01, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.where(mods == m)[0] for m in np.unique(mods)]
    pa, pb, d = [], [], []
    for _ in range(B):
        ia = rng.integers(0, len(cal_a), len(cal_a))           # resample calibration noise
        ib = rng.integers(0, len(cal_b), len(cal_b))
        ga, gb = np.quantile(cal_a[ia], 1 - pfa), np.quantile(cal_b[ib], 1 - pfa)
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])   # stratified, paired
        a, b = np.mean(sig_a[idx] > ga), np.mean(sig_b[idx] > gb)
        pa.append(a); pb.append(b); d.append(a - b)
    q = lambda x: np.percentile(x, [2.5, 97.5])
    return q(pa), q(pb), q(d)


def binomial_ci(p, n, z=1.96):
    h = z * np.sqrt(p * (1 - p) / n)
    return p - h, p + h


def main():
    import torch
    ap = argparse.ArgumentParser()
    for k in ["a-calib", "a-test", "b-calib", "b-test"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--test", default="spectrum_data/test_data_raw_full.pt")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--pfa", type=float, default=0.01)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    d = torch.load(args.test, weights_only=False)
    labels, snrs, mods = d["labels"].numpy(), d["snrs"].numpy(), np.array(d["signals"])
    ca, ta = np.load(args.a_calib), np.load(args.a_test)
    cb, tb = np.load(args.b_calib), np.load(args.b_test)
    na, nb = args.names
    ga, gb = np.quantile(ca, 1 - args.pfa), np.quantile(cb, 1 - args.pfa)
    h0 = labels == 0
    for nm, t, g in [(na, ta, ga), (nb, tb, gb)]:
        p = np.mean(t[h0] > g); lo, hi = binomial_ci(p, h0.sum())
        print(f"measured Pfa {nm}: {p:.4f}  [{lo:.4f}, {hi:.4f}]  (n={h0.sum()})")
    print(f"\n{'SNR':>5}  {na:>8} [95% CI]        {nb:>8} [95% CI]        diff [95% CI]          only-{na} only-{nb}")
    for s in np.unique(snrs):
        m = (snrs == s) & (labels == 1)
        sa, sb, mm = ta[m], tb[m], mods[m]
        ci_a, ci_b, ci_d = paired_bootstrap(ca, cb, sa, sb, mm, args.pfa, args.B)
        da, db = sa > ga, sb > gb
        print(f"{s:5.0f}  {da.mean():8.3f} [{ci_a[0]:.3f},{ci_a[1]:.3f}]  {db.mean():8.3f} [{ci_b[0]:.3f},{ci_b[1]:.3f}]"
              f"  {da.mean()-db.mean():+.3f} [{ci_d[0]:+.3f},{ci_d[1]:+.3f}]   {np.sum(da & ~db):6d} {np.sum(db & ~da):6d}")


if __name__ == "__main__":
    main()
