"""Reviewer (c): focused higher-order-QAM comparison from a completed sweep."""
import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'spectrum_data/nonlinearity_full')
    args = parser.parse_args()
    with (args.results / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    with (args.results / 'paired_differences.csv').open() as f:
        gaps = list(csv.DictReader(f))
    conditions = [('linear', 'Linear', '-'), ('tx_0_rx_0dB', 'Combined compression', '--')]
    snrs = sorted({float(r['snr_db']) for r in metrics})
    colors = {'CAE': '#C65B22', 'PsiNN': '#176B9B'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1]})
    for col, mod in enumerate(('16qam', '32qam')):
        for condition, label, style in conditions:
            for model in ('CAE', 'PsiNN'):
                selected = [r for r in metrics if r['mode'] == 'recalibrated'
                            and r['condition'] == condition and r['modulation'] == mod
                            and r['model'] == model]
                if not selected:
                    raise ValueError(f'Missing results for {condition}, {mod}, {model}')
                means = [np.mean([float(r['pd']) for r in selected if float(r['snr_db']) == snr])
                         for snr in snrs]
                axes[0, col].plot(snrs, means, style, color=colors[model], linewidth=2,
                                  marker='o' if condition == 'linear' else 's', markersize=4,
                                  label=f'{model} · {label}')
            # Paired gaps directly address whether the difference survives distortion.
            # Shading shows variation across simulation seeds, not a confidence interval.
            selected = [r for r in gaps if r['mode'] == 'recalibrated'
                        and r['condition'] == condition and r['modulation'] == mod]
            values = [[float(r['delta_pd']) for r in selected if float(r['snr_db']) == snr]
                      for snr in snrs]
            color = '#454545' if condition == 'linear' else '#7855A4'
            axes[1, col].plot(snrs, [np.mean(v) for v in values], style, color=color,
                              linewidth=2, label=label)
            axes[1, col].fill_between(snrs, [min(v) for v in values], [max(v) for v in values],
                                      color=color, alpha=.14, linewidth=0)
        axes[0, col].set_title(mod.replace('qam', '-QAM'), fontweight='bold')
        axes[0, col].set_ylim(-.025, 1.035)
        axes[0, col].set_ylabel(r'Detection probability, $P_d$')
        axes[1, col].set_ylim(-.02, .51)
        axes[1, col].axhline(0, color='#777777', linewidth=.8)
        axes[1, col].set_ylabel(r'$P_d$(Psi-NN) − $P_d$(CAE)')
        axes[1, col].set_xlabel('Nominal SNR (dB), post-TX / pre-RX')
        for ax in axes[:, col]:
            ax.grid(alpha=.18)
            ax.set_xticks(snrs)
        axes[0, col].legend(loc='lower right', fontsize=8.5, frameon=False)
        axes[1, col].legend(loc='upper right', fontsize=9, frameon=False)
    fig.suptitle('Does the Psi-NN advantage persist under hardware compression?', fontsize=16, y=.98)
    fig.text(.5, .932, 'Linear reference vs. strongest combined Rapp compression tested (TX 0 dB / RX 0 dB)',
             ha='center', fontsize=11)
    fig.text(.5, .027,
             'Means across 3 simulation seeds; shaded gap bands show seed minima–maxima, not confidence intervals.\n'
             'Recalibrated thresholds target Pfa = 0.01. Compression: mean measured Pfa ≈ 0.0091 (CAE), 0.0100 (Psi-NN).',
             ha='center', fontsize=9, linespacing=1.6)
    fig.tight_layout(rect=(0, .09, 1, .91), h_pad=2.2)
    for extension in ('png', 'pdf'):
        fig.savefig(args.results / f'focused_nonlinearity_comparison.{extension}', dpi=220)
    plt.close(fig)
    # Provide probability units for users who prefer the standard sensing notation.
    table = args.results / 'focused_table.csv'
    if table.exists():
        with table.open() as f:
            rows = list(csv.reader(f))
        with (args.results / 'focused_table_probability.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([h.replace(' (%)', ' (0–1)') for h in rows[0]])
            for row in rows[1:]:
                writer.writerow(row[:2] + [f'{float(v)/100:.3f}' for v in row[2:]])
    print(f'Saved focused plot (PNG/PDF) in {args.results}')


if __name__ == '__main__':
    main()
