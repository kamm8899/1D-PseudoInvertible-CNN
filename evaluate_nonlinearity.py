"""Paired hardware-nonlinearity sweep using existing, frozen checkpoints.

Added for reviewer comment (c): test whether CAE/Psi-NN differences persist
under transmitter and receiver compression, especially for higher-order QAM.
Default models match verify_mod_auc_10db.py. See NONLINEARITY.md for protocol.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cae_spectrum import CAE
from experiment_labels import PAPER_CAE_CHECKPOINT, PAPER_PSINN_CHECKPOINT
from generate_spectrum_dataset import generate_iq_dataset
from psinn_layer_1d import AE_Classifier1d
from psinn_layer_1d_pablos import AE_Pablos1d
from rf_nonlinearity import receiver

ROOT = Path(__file__).resolve().parent
MODS = ("bpsk", "qpsk", "16qam", "32qam")


def normalize(x):
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True)


def scores(model, data, channels, device, batch_size):
    result = []
    with torch.inference_mode():
        for batch in data.split(batch_size):
            x = batch[:, :channels].to(device)
            recon = model.AE(x) if hasattr(model, "AE") else model(x)
            # The four-layer legacy architecture produces 1038 samples for 1024
            # inputs; retain the crop used by evaluate_anomaly_inverted.py.
            if recon.shape[-1] > x.shape[-1]:
                recon = recon[..., :x.shape[-1]]
            if recon.shape != x.shape:
                raise ValueError(f"Reconstruction shape {recon.shape} != input {x.shape}")
            sse = (x - recon).square().sum(dim=(1, 2))
            sst = (x - x.mean(dim=(1, 2), keepdim=True)).square().sum(dim=(1, 2))
            result.append((1 - sse / (sst + 1e-8)).cpu().numpy())
    result = np.concatenate(result)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite beta scores; inspect checkpoint/numerics")
    return result


def wilson(detected):
    n = len(detected)
    p = float(np.mean(detected))
    z = 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return p, max(0., center-half), min(1., center+half)


def paired_interval(a, b, rng, repeats):
    """Paired percentile bootstrap of Pd(PsiNN)-Pd(CAE), conditional on thresholds."""
    differences = a.astype(float) - b.astype(float)
    # Multinomial counts exactly reproduce resampling of the {-1,0,1} differences.
    probabilities = np.array([(differences == v).mean() for v in (-1, 0, 1)])
    counts = rng.multinomial(len(a), probabilities, size=repeats)
    draws = (counts[:, 2] - counts[:, 0]) / len(a)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(differences.mean()), float(lo), float(hi)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models(args, device):
    # Match existing checkpoint architectures and preprocessing; weights stay
    # frozen so this experiment measures robustness without adaptation training.
    if args.model_pair == "paper":
        psi = AE_Pablos1d(nf=16, k=5, use_dropout=True)
        cae_path = ROOT / PAPER_CAE_CHECKPOINT
        psi_path = ROOT / PAPER_PSINN_CHECKPOINT
        channels = 1
    else:
        channels = 2 if args.model_pair == "legacy" else 1
        psi = AE_Classifier1d(n_channels=channels, n_classes=1, nf=16, k=5, use_dropout=True)
        cae_path = ROOT / "spectrum_data/cae_best.pth"
        suffix = "_ch1" if channels == 1 else ""
        psi_path = ROOT / f"spectrum_data/psl_cnn_200epochs{suffix}.pth"
    specs = {"CAE": (CAE(), 1, args.cae_checkpoint or cae_path),
             "PsiNN": (psi, channels, args.psi_checkpoint or psi_path)}
    models, provenance = {}, {}
    for name, (model, nch, path) in specs.items():
        path = Path(path).resolve()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        models[name] = (model.to(device).eval(), nch)
        provenance[name] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "class": type(model).__name__, "channels": nch, "score": "beta", "tail": "upper"}
    return models, provenance


def plot_results(out, rows, conditions, snrs):
    for mode in ("frozen", "recalibrated"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
        for ax, mod in zip(axes.flat, MODS):
            for ci, (condition, _, _) in enumerate(conditions):
                for model, style in (("CAE", "--"), ("PsiNN", "-")):
                    means = [np.mean([r["pd"] for r in rows if r["mode"] == mode and
                             r["condition"] == condition and r["model"] == model and
                             r["modulation"] == mod and r["snr_db"] == snr]) for snr in snrs]
                    ax.plot(snrs, means, style, color=f"C{ci % 10}", label=f"{condition}: {model}")
            ax.set_title(mod.upper())
            ax.set_xlabel("Nominal SNR (dB), post-TX / pre-RX")
            ax.set_ylabel("Detection probability")
            ax.set_ylim(-.02, 1.02)
            ax.grid(alpha=.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.suptitle(f"{mode.capitalize()} thresholds; mean across simulation seeds")
        fig.tight_layout(rect=(0, min(.42, .04 + .024*len(conditions)), 1, .96))
        fig.savefig(out / f"pd_vs_snr_{mode}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--model-pair", choices=("paper", "legacy", "matched"), default="paper")
    parser.add_argument("--cae-checkpoint", type=Path)
    parser.add_argument("--psi-checkpoint", type=Path)
    parser.add_argument("--tx-ibo-db", type=float, nargs="*", default=[9, 6, 3, 0])
    parser.add_argument("--rx-backoff-db", type=float, nargs="*", default=[])
    parser.add_argument("--combined", action="store_true", help="Also evaluate Cartesian TX/RX combinations")
    parser.add_argument("--rapp-p", type=float, default=2.)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--snrs", type=float, nargs="+", default=list(range(-10, 11, 2)))
    parser.add_argument("--samples-per-cell", type=int, default=1000)
    parser.add_argument("--noise-per-snr", type=int, default=1000)
    parser.add_argument("--calibration-samples", type=int, default=10000)
    parser.add_argument("--pfa", type=float, default=.01)
    parser.add_argument("--snr-uncertainty-db", type=float, default=2.)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.samples_per_cell, args.noise_per_snr, args.calibration_samples,
           args.bootstrap, args.batch_size, args.threads) < 1:
        parser.error("Counts and threads must be positive")
    if not 0 < args.pfa < 1 or not np.isfinite(args.rapp_p) or args.rapp_p <= 0:
        parser.error("Require 0 < pfa < 1 and finite positive rapp-p")
    if not np.isfinite(args.snr_uncertainty_db) or args.snr_uncertainty_db < 0:
        parser.error("SNR uncertainty must be finite and nonnegative")
    for values in (args.snrs, args.tx_ibo_db, args.rx_backoff_db, args.seeds):
        if len(values) != len(set(values)) or not all(np.isfinite(v) for v in values):
            parser.error("Sweep values and seeds must be finite and unique")
    if any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must lie in [0, 2**32)")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    torch.set_num_threads(args.threads)
    models, provenance = load_models(args, torch.device(args.device))
    # Reviewer (c): keep a linear reference, isolate TX and RX effects, and
    # optionally test both together using the same underlying random waveforms.
    conditions = [("linear", None, None)]
    conditions += [(f"tx_{v:g}dB", v, None) for v in args.tx_ibo_db]
    conditions += [(f"rx_{v:g}dB", None, v) for v in args.rx_backoff_db]
    if args.combined:
        conditions += [(f"tx_{t:g}_rx_{r:g}dB", t, r) for t in args.tx_ibo_db for r in args.rx_backoff_db]
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(models=provenance, conditions=conditions, torch_version=torch.__version__,
                  normalization="Joint I/Q sample mean/std, then channel slice",
                  snr_reference="Total distorted signal power after TX, before RX; PA loss compensated",
                  calibration="Independent H0 empirical 1-pfa quantile (linear interpolation); upper tail, strict >",
                  confidence_intervals="95% Wilson Pd/Pfa; paired percentile bootstrap delta Pd, conditional on calibration and fixed weights",
                  limitations="Memoryless AM/AM only; original generator pulse shaping and constellation retained")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, gaps = [], []
    for seed in args.seeds:
        # Independent RNG streams; same underlying H0 across impairment settings.
        cal_rng = torch.Generator().manual_seed(seed + 2**32)
        calibration = torch.randn(args.calibration_samples, 2, 1024, generator=cal_rng)
        # Advisor item 2: use NumPy default quantile interpolation consistently.
        # Independent simulated calibration remains separate from training and test H0.
        thresholds = {}
        # RX compression changes H0, so calibrate it before input normalization.
        # TX does not affect H0 and therefore needs no separate noise threshold.
        for rx in [None] + args.rx_backoff_db:
            cal_data = normalize(receiver(calibration, rx, args.rapp_p))
            thresholds[rx] = {name: float(np.quantile(scores(model, cal_data, nch, args.device, args.batch_size),
                                                       1 - args.pfa))
                              for name, (model, nch) in models.items()}
        for condition, tx, rx in conditions:
            print(f"seed={seed} condition={condition}", flush=True)
            for si, snr in enumerate(args.snrs):
                # Reinitializing with the same seed pairs every condition and model.
                data_seed = int(np.random.SeedSequence([seed, si, 719]).generate_state(1)[0])
                generated = generate_iq_dataset(num_train=1, samples_per_mod_per_snr=args.samples_per_cell,
                            noise_per_snr=args.noise_per_snr, seed=data_seed, snr_points=[snr],
                            tx_ibo_db=tx, rx_backoff_db=rx, rapp_p=args.rapp_p,
                            snr_uncertainty_db=args.snr_uncertainty_db)
                data, labels, mods = generated[2], generated[4].numpy(), np.array(generated[6])
                beta = {name: scores(model, data, nch, args.device, args.batch_size)
                        for name, (model, nch) in models.items()}
                np.savez_compressed(args.output / f"scores_seed{seed}_{condition}_snr{snr:g}.npz",
                                    labels=labels, modulations=mods, **beta)
                h0 = labels == 0
                # Frozen thresholds measure unexpected hardware distortion;
                # recalibration targets the same nominal Pfa under the new RX.
                for mode in ("frozen", "recalibrated"):
                    gamma = thresholds[None if mode == "frozen" else rx]
                    decisions = {name: b > gamma[name] for name, b in beta.items()}
                    # Keep higher-order QAM results separate rather than hiding
                    # a modulation-specific change in an overall average.
                    for mod in MODS:
                        h1 = mods == mod
                        mask = h0 | h1
                        common = dict(seed=seed, condition=condition, tx_ibo_db=tx, rx_backoff_db=rx,
                                      mode=mode, modulation=mod, snr_db=snr)
                        for name in models:
                            pd, pd_lo, pd_hi = wilson(decisions[name][h1])
                            pfa, fa_lo, fa_hi = wilson(decisions[name][h0])
                            rows.append(dict(**common, model=name, gamma=gamma[name], target_pfa=args.pfa,
                                        pd=pd, pd_lo=pd_lo, pd_hi=pd_hi, measured_pfa=pfa,
                                        pfa_lo=fa_lo, pfa_hi=fa_hi, auc=roc_auc_score(labels[mask], beta[name][mask]),
                                        n_signal=int(h1.sum()), n_noise=int(h0.sum())))
                        # Paired resampling compares decisions on identical H1
                        # windows; these intervals condition on fixed thresholds.
                        rng = np.random.default_rng(np.random.SeedSequence([seed, si, MODS.index(mod), 991]))
                        delta, lo, hi = paired_interval(decisions["PsiNN"][h1], decisions["CAE"][h1], rng, args.bootstrap)
                        gaps.append(dict(**common, delta_pd=delta, delta_lo=lo, delta_hi=hi))
                del generated, data
            # Save completed conditions so a long interrupted run retains useful work.
            write_csv(args.output / "metrics.csv", rows)
            write_csv(args.output / "paired_differences.csv", gaps)
    plot_results(args.output, rows, conditions, args.snrs)
    (args.output / "COMPLETE.txt").write_text("Sweep completed. See config.json for protocol and checkpoint hashes.\n")
    print(f"Saved metrics, paired differences, scores, and figures to {args.output}")


if __name__ == "__main__":
    main()
