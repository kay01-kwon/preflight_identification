#!/usr/bin/env python3
"""The external-baseline comparison, as a table and a figure.

The point the comparison has to carry is not that the proposed estimate
is more accurate than the online identification literature -- it is
not, and saying so would invite a reviewer to check. It is that the
same accuracy is reached before the first take-off, whereas every
baseline needs the vehicle already airborne with the offset
uncompensated, which is the condition the procedure exists to remove.

The figure is built to make exactly that readable:

  (a) parity against the load cell, which establishes that the
      baselines were implemented fairly -- a straw man shows up here as
      a slope far from unity or a scatter with no correlation, and
      neither is present;
  (b) accuracy against the airborne time each method needs before it
      can produce an estimate, where the proposed method sits on the
      zero axis at the same height as the filters.

Inputs are the CSVs written by nls_comparison.py (proposed),
inflight_baselines.py (RLS/EKF/UKF) and hover_trim_baseline.py.

Usage
-----
  python analysis/baseline_comparison.py [--nls DIR] [--ifb DIR]
                                         [--hover DIR] [--outdir DIR]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

TRUTH = {'01': (-11.45, -2.90), '02': (-9.90, -14.29), '03': (3.14, -5.26),
         '04': (2.40, 6.67), '05': (-10.89, 10.91)}
# airborne seconds with the offset uncompensated, before an estimate
# exists. The pre-flight procedure needs none: it runs on the ground.
AIRBORNE = {'proposed': 0.0, 'EKF': 28.0, 'UKF': 28.0, 'RLS': 28.0,
            'hover trim': 13.0}
ORDER = ['proposed', 'EKF', 'UKF', 'RLS', 'hover trim']
COL = {'proposed': '#0072B2', 'EKF': '#D55E00', 'UKF': '#E69F00',
       'RLS': '#009E73', 'hover trim': '#CC79A7'}
MRK = {'proposed': 'o', 'EKF': 's', 'UKF': 'D', 'RLS': '^',
       'hover trim': 'v'}


def load(a):
    """{method: {(case, 'x'|'y'): estimate_mm}} across the three sources."""
    est = defaultdict(dict)

    # proposed: the pivot-free offset of the estimator benchmark
    f = a.nls / 'nls_comparison_summary.csv'
    for r in csv.DictReader(open(f)):
        if r['method'] != 'cosh':
            continue
        # Mx senses the y offset, My the x offset
        comp = 'y' if r['axis'] == 'Mx' else 'x'
        est['proposed'][(r['case'].replace('case_', ''), comp)] = \
            float(r['offset_mm'])

    for src, methods in ((a.ifb / 'inflight_baselines.csv', None),
                         (a.hover / 'hover_trim_baseline.csv',
                          ['hover trim'])):
        if not src.exists():
            print(f'  missing {src}, skipped')
            continue
        rows = list(csv.DictReader(open(src)))
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            m = methods[0] if methods else r['method']
            by[m][r['case']].append((float(r['x_mm']), float(r['y_mm'])))
        for m, cases in by.items():
            for case, v in cases.items():
                v = np.array(v)
                est[m][(case, 'x')] = float(v[:, 0].mean())
                est[m][(case, 'y')] = float(v[:, 1].mean())
    return est


def stats(d):
    t = np.array([TRUTH[c][0 if k == 'x' else 1] for (c, k) in d])
    e = np.array([d[key] for key in d])
    err = np.abs(e - t)
    sl, _ = np.polyfit(t, e, 1)
    return dict(rms=np.sqrt(np.mean(err ** 2)), med=np.median(err),
                mx=err.max(), corr=np.corrcoef(t, e)[0, 1], slope=sl,
                truth=t, est=e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nls', type=Path, default=Path('/tmp/nlscmp'))
    p.add_argument('--ifb', type=Path, default=Path('/tmp/ifb'))
    p.add_argument('--hover', type=Path, default=Path('/tmp/hover'))
    p.add_argument('--outdir', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'docs')
    p.add_argument('--dpi', type=int, default=600)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    est = load(a)
    S = {m: stats(est[m]) for m in ORDER if m in est}

    print(f'{"method":<13}{"RMS":>7}{"med":>7}{"max":>7}'
          f'{"corr":>7}{"slope":>7}{"airborne [s]":>14}')
    for m in ORDER:
        if m not in S:
            continue
        s = S[m]
        print(f'{m:<13}{s["rms"]:>7.2f}{s["med"]:>7.2f}{s["mx"]:>7.2f}'
              f'{s["corr"]:>7.2f}{s["slope"]:>7.2f}'
              f'{AIRBORNE[m]:>14.0f}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) parity
    lim = 17
    ax1.plot([-lim, lim], [-lim, lim], color='0.55', lw=1.0, ls='--',
             zorder=1)
    for m in ORDER:
        if m not in S:
            continue
        ax1.plot(S[m]['truth'], S[m]['est'], MRK[m], ms=5.5, mfc='none',
                 mew=1.4, color=COL[m], ls='', label=m, zorder=3)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect('equal')
    ax1.set_xlabel('load-cell offset [mm]', fontsize=9.5)
    ax1.set_ylabel('identified offset [mm]', fontsize=9.5)
    ax1.grid(alpha=0.45, lw=0.8, color='0.6')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax1.set_title('(a)  every estimator tracks the truth',
                  fontsize=9.5, loc='left')

    # (b) accuracy against what it costs to get there
    for m in ORDER:
        if m not in S:
            continue
        ax2.plot(AIRBORNE[m], S[m]['rms'], MRK[m], ms=9, mfc=COL[m],
                 mec='0.25', mew=1.0, zorder=3)
        if m == 'UKF':
            continue          # it lands on the EKF; one label serves both
        lab = 'EKF / UKF' if m == 'EKF' else m
        ax2.annotate(lab, (AIRBORNE[m], S[m]['rms']), xytext=(0, 11),
                     textcoords='offset points', ha='center',
                     fontsize=8.5, color='0.2')
    ax2.axvline(0, color=COL['proposed'], lw=1.0, ls=':', zorder=1)
    ax2.set_xlim(-4, 34)
    ax2.set_ylim(0, 3.2)
    ax2.set_xlabel('airborne time required, offset uncompensated [s]',
                   fontsize=9.5)
    ax2.set_ylabel('offset error [mm RMS]', fontsize=9.5)
    ax2.grid(alpha=0.45, lw=0.8, color='0.6')
    ax2.set_axisbelow(True)
    ax2.set_title('(b)  the same accuracy, at a different cost',
                  fontsize=9.5, loc='left')

    fig.tight_layout()
    fig.savefig(a.outdir / 'exp_baselines.png', bbox_inches='tight',
                dpi=a.dpi)
    print(f'\nwritten to {a.outdir / "exp_baselines.png"}')

    # ---- the distribution behind the RMS, as a box per method -------
    # Every method is aggregated to the same level first -- the mean
    # over the trials of a configuration -- so the boxes hold the same
    # ten case-axis values and the comparison is like for like. The
    # baselines have thirty trials behind them and the proposed
    # estimate has the excitation campaign; pooling to the
    # configuration removes that asymmetry rather than hiding it.
    present = [m for m in ORDER if m in S]
    fig2, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [np.abs(S[m]['est'] - S[m]['truth']) for m in present]
    bp = ax.boxplot(data, positions=np.arange(len(present)), widths=0.55,
                    patch_artist=True,
                    medianprops=dict(color='0.15', lw=1.6),
                    whiskerprops=dict(color='0.35'),
                    capprops=dict(color='0.35'),
                    flierprops=dict(marker='', ms=0), zorder=2)
    for b, m in zip(bp['boxes'], present):
        b.set(facecolor=COL[m], alpha=0.32, edgecolor=COL[m], lw=1.4)
    # the limit is set before the annotations so they anchor to the
    # final top rather than to the autoscaled one, which put them on
    # the tallest whisker
    ax.set_ylim(0, max(d.max() for d in data) * 1.28)
    for k, m in enumerate(present):
        ax.annotate(f'{S[m]["rms"]:.2f}', (k, ax.get_ylim()[1]),
                    xytext=(0, -3), textcoords='offset points',
                    ha='center', va='top', fontsize=8.5, color='0.25')
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([('proposed\n(pre-flight)' if m == 'proposed'
                         else m) for m in present], fontsize=9)
    ax.set_ylabel('offset error [mm]', fontsize=9.5)
    ax.grid(axis='y', alpha=0.55, lw=0.9, color='0.55')
    ax.set_axisbelow(True)
    ax.axvline(0.5, color='0.6', lw=0.8, ls=':')
    ax.set_title(f'RMS annotated; n = {len(data[0])} configurations each',
                 fontsize=9, loc='right', color='0.35')
    fig2.tight_layout()
    fig2.savefig(a.outdir / 'exp_baselines_box.png',
                 bbox_inches='tight', dpi=a.dpi)
    print(f'written to {a.outdir / "exp_baselines_box.png"}')

    # the same numbers as a LaTeX table body
    tex = a.outdir / 'tab_baselines.tex'
    with open(tex, 'w') as fh:
        for m in ORDER:
            if m not in S:
                continue
            s = S[m]
            name = ('proposed (pre-flight)' if m == 'proposed'
                    else m)
            need = ('none' if AIRBORNE[m] == 0
                    else f'${AIRBORNE[m]:.0f}$')
            fh.write(f'{name} & ${s["rms"]:.2f}$ & ${s["med"]:.2f}$ & '
                     f'${s["mx"]:.2f}$ & ${s["corr"]:.2f}$ & '
                     f'${s["slope"]:.2f}$ & {need} \\\\\n')
    print(f'table body -> {tex}')


if __name__ == '__main__':
    main()
