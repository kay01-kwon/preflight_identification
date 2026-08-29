#!/usr/bin/env python3
"""Run-gate metric report on the floored segment.

For every run, evaluates the three gate quantities in order — (1) slope
tracking error, (2) ramp-linearity RMSE, (3) window size — with the
slope and linearity scored on the |M| >= M_floor sub-segment (0.7 N·m
roll / 0.4 N·m pitch, the |M_crit| lower bound over the ±20 mm
CoM-offset box), and the window size as the full excitation-window
sample count N_full (with the floored count N_floor alongside).

Writes gate_report.csv, docs/fig_gate_report.pdf (three panels vs ramp
rate with the gate thresholds), and prints a per-rate LaTeX summary.

Usage: PYTHONPATH=<stubs> python analysis/gate_report.py [SCRATCH]
                                        [--outdir DIR] [--dpi N]
"""
import argparse
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
ap = argparse.ArgumentParser()
ap.add_argument('scratch', nargs='?', default='.',
                help='directory for gate_report.csv (re-used if present)')
ap.add_argument('--outdir', type=Path,
                default=Path(__file__).resolve().parents[1] / 'docs',
                help='directory the figure is written to')
ap.add_argument('--dpi', type=int, default=600,
                help='raster resolution of the PNG output')
args = ap.parse_args()
OUT = Path(args.scratch)
RATES = [0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20]

rows = []
_csv = OUT / 'gate_report.csv'
if _csv.exists():
    for r in csv.DictReader(open(_csv)):
        r['rate'] = float(r['rate'])
        r['n_full'] = int(r['n_full'])
        r['n_floor'] = int(r['n_floor'])
        rows.append(r)
    print(f're-using {_csv} ({len(rows)} runs)')
for d in (() if rows else sorted(ROOT.glob('case_*/M[xy]'))):
    axis = 'x' if d.name == 'Mx' else 'y'
    floor = cvp.SLOPE_GATE_FLOOR[axis]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    for bag in bags:
        cmd = cvp.commanded_ramp_rate(bag.name)
        if not cmd:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            sig = cvp.prepare_signals(bag, axis)
        i0, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
        win = slice(i0, i1 + 1)
        t, Mw = sig['t'][win], sig['moment'][win]
        mask = np.abs(Mw) >= floor
        tf, Mf = t[mask], Mw[mask]
        a, b = np.polyfit(tf, Mf, 1)
        eps = (abs(a) - cmd) / cmd * 100.0
        lin = float(np.std(Mf - (a * tf + b)))
        rows.append(dict(case=d.parent.name, axis=d.name, bag=bag.name,
                         rate=cmd,
                         dir='pos' if bag.name.startswith('pos') else 'neg',
                         eps_pct=f"{eps:+.2f}",
                         lin_mNm=f"{1e3 * lin:.1f}",
                         n_full=i1 - i0 + 1, n_floor=int(mask.sum())))
    print('done', d.parent.name, d.name, flush=True)

with open(OUT / 'gate_report.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# per-rate LaTeX summary in the requested metric order
print("\n% rate & |eps| med & |eps| max & lin med & lin max & "
      "N_full min & N_floor min")
for r0 in RATES:
    g = [r for r in rows if abs(r['rate'] - r0) < 1e-9]
    e = np.abs([float(r['eps_pct']) for r in g])
    ln = np.array([float(r['lin_mNm']) for r in g])
    nf = np.array([r['n_full'] for r in g])
    nfl = np.array([r['n_floor'] for r in g])
    print(f"${r0:.2f}$ & ${np.median(e):.2f}$ & ${e.max():.2f}$ & "
          f"${np.median(ln):.1f}$ & ${ln.max():.1f}$ & "
          f"${nf.min()}$ & ${nfl.min()}$ \\\\  % n={len(g)}")
e = np.abs([float(r['eps_pct']) for r in rows])
ln = np.array([float(r['lin_mNm']) for r in rows])
print(f"% all runs (n={len(rows)}): |eps| med {np.median(e):.2f} "
      f"max {e.max():.2f} %; lin med {np.median(ln):.1f} "
      f"max {ln.max():.1f} mN·m; N_full min "
      f"{min(r['n_full'] for r in rows)}")

# ── figure: three panels vs ramp rate, gate thresholds drawn ──
AXCOL = {'Mx': '#0072B2', 'My': '#E69F00'}
AXMRK = {'Mx': 'o', 'My': 's'}
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.1))
panels = [('eps_pct', r'slope tracking error $\varepsilon$ [%]',
           [(3.0, '+3%'), (-3.0, '\u22123%')], 'lin'),
          ('lin_mNm', r'ramp-linearity RMSE [mN$\cdot$m]',
           [(30.0, '30 mN$\\cdot$m')], 'lin'),
          ('n_full', r'window samples $N_{\mathrm{full}}$',
           [(38.0, '38')], 'log')]
for ax, (col, ylab, gates, scale) in zip(axes, panels):
    for axname in ('Mx', 'My'):
        g = [r for r in rows if r['axis'] == axname]
        x = np.array([r['rate'] for r in g])
        x = x * (1 + rng.uniform(-0.03, 0.03, len(x)))
        y = np.array([float(r[col]) for r in g])
        ax.plot(x, y, AXMRK[axname], color=AXCOL[axname], ms=3.6,
                alpha=0.75, mec='none',
                label=axname if ax is axes[0] else None)
    for yv, lab in gates:
        ax.axhline(yv, color='0.25', lw=1.0, ls='--')
        ax.annotate(lab, xy=(1.0, yv), xycoords=('axes fraction', 'data'),
                    xytext=(-2, 3), textcoords='offset points',
                    ha='right', fontsize=7, color='0.25')
    if scale == 'log':
        ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{r:g}" for r in RATES], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r'commanded ramp rate [N$\cdot$m/s]')
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
args.outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(args.outdir / 'fig_gate_report.pdf', bbox_inches='tight')
fig.savefig(args.outdir / 'fig_gate_report.png', dpi=args.dpi,
            bbox_inches='tight')
print(f'figure saved to {args.outdir}')
