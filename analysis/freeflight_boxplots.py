#!/usr/bin/env python3
"""Box charts of the take-off transient: uncompensated against pivot-free.

One figure per metric, each showing the five cases and then the pooled
set, so the case-by-case behaviour and the campaign statement are read
off the same axis rather than from two separate places.

WHAT IS POOLED, AND WHY IT IS SHOWN.  A case box holds the six trials of
that condition -- two controllers by three repeats -- because the claim
being made is about the compensation, not about HGDO versus L1. Pooling
silently would hide a controller split if one existed, so every trial is
drawn on top of its box with the controller distinguished by marker;
the reader can see that the two interleave.

The metrics are the rotation-invariant magnitudes of
analysis/freeflight_metrics.py: the offset points 14 to 135 degrees off
axis across the five cases, so a per-axis box would split one excursion
between two panels along axes the physics does not prefer.

Only the pivot-free variant is drawn. The pivot-based variant needs a
motion-capture arc fit to recover the lever arm and its take-off
performance is statistically indistinguishable over these trials, so it
is not part of the reported pipeline.

Usage
-----
  python analysis/freeflight_boxplots.py [--csv FILE] [--outdir DIR]

Reads freeflight_metrics_runs.csv (analysis/freeflight_metrics.py) and
writes docs/fig_ff_box_{tilt,rate,drift,speed}.{pdf,png}.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

CASES = ['01', '02', '03', '04', '05']
PAIR = [('wo_ff', 'uncompensated'), ('ff_pivot_free', 'pivot-free')]
COL = {'wo_ff': '#D55E00', 'ff_pivot_free': '#0072B2'}
CTRLS = ['hgdo', 'l1']
CNAME = {'hgdo': 'HGDO', 'l1': r'$\mathcal{L}_1$'}
HATCH = {'hgdo': '', 'l1': '///'}
METRICS = [
    ('tilt',  'peak tilt from vertical [deg]',      '{:.2f}'),
    ('rate',  r'peak $\|(\omega_x,\omega_y)\|$ [deg/s]', '{:.1f}'),
    ('drift', 'peak horizontal drift [m]',          '{:.3f}'),
    ('speed', 'peak horizontal speed [m/s]',        '{:.3f}'),
]


def draw(rows, key, ylabel, fmt, outdir):
    by = defaultdict(list)
    for r in rows:
        by[(r['case'], r['variant'], r['controller'])].append(r)

    groups = CASES + ['all']
    # four boxes per group: each variant split by controller, so the
    # compensation effect and the HGDO/L1 comparison are answered on one
    # axis. The pair of a variant sits together, with a wider gap
    # between variants than within one
    W, INNER, MID, GAP = 0.22, 0.03, 0.16, 0.42
    span = 4 * W + 2 * INNER + MID
    centres = np.arange(len(groups)) * (span + GAP)
    off = {}
    x = -span / 2 + W / 2
    for var, _ in PAIR:
        for ctrl in CTRLS:
            off[(var, ctrl)] = x
            x += W + INNER
        x += MID - INNER

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for (var, _) in PAIR:
        for ctrl in CTRLS:
            pos, data = [], []
            for gi, g in enumerate(groups):
                sel = ([r for r in rows if r['variant'] == var
                        and r['controller'] == ctrl] if g == 'all'
                       else by[(g, var, ctrl)])
                data.append([float(r[key]) for r in sel])
                pos.append(centres[gi] + off[(var, ctrl)])
            bp = ax.boxplot(data, positions=pos, widths=W,
                            patch_artist=True,
                            medianprops=dict(color='0.15', lw=1.5),
                            whiskerprops=dict(color='0.35'),
                            capprops=dict(color='0.35'),
                            flierprops=dict(marker='', ms=0), zorder=2)
            for b in bp['boxes']:
                b.set(facecolor=COL[var], alpha=0.30, edgecolor=COL[var],
                      lw=1.2, hatch=HATCH[ctrl])

    ax.set_xticks(centres)
    # the per-case boxes rest on three repeats, so their whiskers are
    # the range rather than a percentile estimate -- say so on the axis
    ax.set_xticklabels([f'case {g}\n(n=3)' for g in CASES]
                       + [f'all\n(n={len(rows) // 4})'], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    # headroom for the legend and the pooled annotation, so neither
    # sits on a box -- case 02 reaches nearly the top of the data range
    ax.set_ylim(0, max(float(r[key]) for r in rows) * 1.30)
    ax.grid(axis='y', alpha=0.55, lw=0.9, color='0.55')
    ax.set_axisbelow(True)
    # separate the pooled column from the per-case ones
    ax.axvline(0.5 * (centres[-1] + centres[-2]), color='0.6', lw=0.8,
               ls=':')

    # the pooled reduction, stated on the figure rather than in prose
    a = np.array([float(r[key]) for r in rows if r['variant'] == 'wo_ff'])
    b = np.array([float(r[key]) for r in rows
                  if r['variant'] == 'ff_pivot_free'])
    # plain '%': the default renderer is not LaTeX, so an escaped one
    # would print the backslash
    ax.annotate(f'{fmt.format(a.mean())} $\\rightarrow$ '
                f'{fmt.format(b.mean())}   '
                f'({100 * (a.mean() - b.mean()) / a.mean():.0f}%)',
                xy=(0.995, 0.965), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5, color='0.25')

    h, lab = [], []
    for v, name in PAIR:
        for c in CTRLS:
            h.append(plt.Rectangle((0, 0), 1, 1, fc=COL[v], alpha=0.30,
                                   ec=COL[v], hatch=HATCH[c]))
            lab.append(f'{name}, {CNAME[c]}')
    ax.legend(h, lab, fontsize=8, ncol=2, loc='upper left', framealpha=0.9)

    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(outdir / f'fig_ff_box_{key}.{ext}',
                    bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'  fig_ff_box_{key}   {fmt.format(a.mean())} -> '
          f'{fmt.format(b.mean())}  '
          f'({100 * (a.mean() - b.mean()) / a.mean():.0f}%)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path,
                   default=Path('freeflight_metrics_runs.csv'))
    p.add_argument('--outdir', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'docs')
    a = p.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv))
            if r['variant'] in dict(PAIR)]
    if not rows:
        raise SystemExit(f'no uncompensated / pivot-free trials in {a.csv}')
    a.outdir.mkdir(parents=True, exist_ok=True)

    n = defaultdict(int)
    for r in rows:
        n[r['variant']] += 1
    print(f'{n["wo_ff"]} uncompensated, {n["ff_pivot_free"]} pivot-free')
    for key, ylabel, fmt in METRICS:
        draw(rows, key, ylabel, fmt, a.outdir)
    print(f'written to {a.outdir}')


if __name__ == '__main__':
    main()
