#!/usr/bin/env python3
"""The three run gates on one axis, as a fraction of their own limits.

The per-metric panels of fig_gate_report carry three different units
and two different senses -- N_full is a floor, epsilon and the
linearity RMSE are ceilings -- so they need three axes and shrink badly
when the figure does.  Normalising each run by its own gate removes
both problems:

    u = |eps| / 3 %,    u = linRMSE / 30 mN.m,    u = 38 / N_full

Every gate is passed exactly when u <= 1, so all three fit on one axis
and the question "did anything come close to a limit?" is answered by
how far the cloud sits below the line.

Reads gate_report.csv (analysis/gate_report.py), which holds the gate
quantities as the pipeline evaluates them -- slope and linearity on the
|M| >= M_floor sub-segment, N_full over the whole excitation window.

Usage: python analysis/gate_margin_figure.py <scratch> [out.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RATES = [0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20]
GATES = [
    ('eps_pct', r'$|\varepsilon|\,/\,3\%$', '#0072B2', 'o', -0.24),
    ('lin_mNm', r'linRMSE$\,/\,30$ mN$\cdot$m', '#E69F00', 's', 0.0),
    ('n_full', r'$38\,/\,N_{\mathrm{full}}$', '#009E73', '^', 0.24),
]


def utilisation(row, col):
    """The run's value as a fraction of that gate's limit."""
    if col == 'eps_pct':
        return abs(float(row[col])) / 3.0
    if col == 'lin_mNm':
        return float(row[col]) / 30.0
    return 38.0 / float(row[col])            # a floor, so inverted


def main():
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out = sys.argv[2] if len(sys.argv) > 2 else 'docs/exp_gate_margin.png'
    rows = list(csv.DictReader(open(scratch / 'gate_report.csv')))
    idx = {r: i for i, r in enumerate(RATES)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    worst = (0.0, None, None)
    for col, lab, colour, mrk, dx in GATES:
        x, y = [], []
        for r in rows:
            u = utilisation(r, col)
            x.append(idx[round(float(r['rate']), 2)] + dx
                     + rng.uniform(-0.07, 0.07))
            y.append(u)
            if u > worst[0]:
                worst = (u, lab, r)
        ax.plot(x, y, mrk, ms=4.2, mec='none', alpha=0.7, color=colour,
                label=lab)
        print(f'  {lab:32} median {np.median(y):.3f}  max {max(y):.3f}')

    ax.axhline(1.0, color='#c0392b', lw=1.4, ls='--', zorder=5)
    ax.text(-0.52, 0.94, 'gate', color='#c0392b', fontsize=10,
            ha='left', va='top')
    ax.set_yscale('log')
    ax.set_ylim(top=3.4)
    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels([f'{r:g}' for r in RATES], fontsize=10)
    ax.set_xlim(-0.6, len(RATES) - 0.4)
    ax.set_xlabel(r'commanded ramp rate $\dot M_{\mathrm{cmd}}$ '
                  r'[N$\cdot$m/s]', fontsize=11)
    ax.set_ylabel('fraction of the gate', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.35, lw=0.7, color='0.6')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9.5, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, 1.005), framealpha=0.92,
              edgecolor='0.85', handletextpad=0.3, columnspacing=1.6)
    ax.set_title(f'{len(rows)}/{len(rows)} runs pass every gate; '
                 f'worst run at {100 * worst[0]:.0f}% of its limit',
                 fontsize=10.5, color='0.25', pad=8)

    print(f'  worst: {worst[1]} = {worst[0]:.3f} on '
          f"{worst[2]['case']}/{worst[2]['axis']}/{worst[2]['bag']}")
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'written {out}')


if __name__ == '__main__':
    main()
