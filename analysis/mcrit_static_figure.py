#!/usr/bin/env python3
"""Static-threshold check figure and table rows (odom arms, GE on/off).

Reads mcrit_prediction.csv (per-group odom-arm predictions, both thrust
scenarios) and nls_comparison_runs.csv (benchmark estimators) from the
scratch directory; writes fig_mcrit_static.{pdf,png} and prints the
LaTeX rows.

Usage
-----
  python analysis/mcrit_static_figure.py [SCRATCH] [--outdir DIR]
                                         [--dpi N]
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('scratch', nargs='?', default='.',
                help='directory holding mcrit_prediction.csv and '
                     'nls_comparison_runs.csv')
ap.add_argument('--outdir', type=Path,
                default=Path(__file__).resolve().parents[1] / 'docs',
                help='directory the figure is written to')
ap.add_argument('--dpi', type=int, default=600,
                help='raster resolution of the PNG output')
args = ap.parse_args()
SC = args.scratch
M = ['cosh', 'cosh_cad', 'nls', 'pelt_normal', 'pelt_rbf', 'cusum']
LBL = {'cosh': 'COSH', 'cosh_cad': 'COSH-CAD', 'nls': 'NLS',
       'pelt_normal': 'CPD-N', 'pelt_rbf': 'CPD-R', 'cusum': 'CUSUM'}
COL = {'cosh': '#0072B2', 'cosh_cad': '#56B4E9', 'nls': '#E69F00',
       'pelt_normal': '#009E73', 'pelt_rbf': '#D55E00', 'cusum': '#CC79A7'}
MRK = {'cosh': 'o', 'cosh_cad': 'X', 'nls': 's', 'pelt_normal': '^',
       'pelt_rbf': 'v', 'cusum': 'D'}

pred = {(r['case'], r['axis'], r['dir']): r
        for r in csv.DictReader(open(f'{SC}/mcrit_prediction.csv'))}
bench = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f'{SC}/nls_comparison_runs.csv')):
    for m in M:
        if r.get(f'mcrit_{m}'):
            bench[(r['case'], r['axis'], r['dir'])][m].append(
                float(r[f'mcrit_{m}']))
bmean = {k: {m: float(np.mean(v)) for m, v in d.items()}
         for k, d in bench.items()}

print("% rows: case ax dir l_odom M_cosh pred:none r:none r:sgl "
      "pred:interf r:interf bench")
for k in sorted(pred):
    r = pred[k]
    bs = [bmean[k][m] for m in M[1:] if m in bmean[k]]
    c = k[0].replace('case_0', '')
    row = (f"{c} & {k[1][1]} & {'+' if k[2] == 'pos' else '-'} & "
           f"{r['l_odom_mm']} & ${float(r['M_ident']):+.3f}$ & "
           f"${float(r['M_pred']):+.3f}$ & ${float(r['resid_mNm']):+.0f}$ & "
           f"${float(r['resid_single_mNm']):+.0f}$ & "
           f"${float(r['M_pred_interf']):+.3f}$ & "
           f"${float(r['resid_interf_mNm']):+.0f}$ & "
           f"$[{min(bs):+.3f}, {max(bs):+.3f}]$ \\\\")
    print(row)

# magnitude-signed residual: |M_ident| - |M_pred| (negative = the model
# over-predicts the threshold).  This is the statistic that exposes the
# one-sided bias, which the raw signed residual hides because the two tip
# directions carry opposite signs.
print("\n% magnitude deficit |M_ident| - |M_pred| [mN·m]")
for key, col in (('no GE', 'resid_mNm'), ('no interference',
                 'resid_single_mNm'), ('interference', 'resid_interf_mNm')):
    d = np.array([(1.0 if r['dir'] == 'pos' else -1.0) * float(r[col])
                  for r in pred.values()])
    print(f"%   {key:16}: mean {d.mean():+7.1f}, "
          f"{int((d < 0).sum())}/{len(d)} over-predicted")

SLOTS = [('Mx', 'neg'), ('Mx', 'pos'), ('My', 'neg'), ('My', 'pos')]
XL = [r'$M_{x,-}$', r'$M_{x,+}$', r'$M_{y,-}$', r'$M_{y,+}$']
cases = [f'case_0{i}' for i in range(1, 6)]

# One shared x axis: twenty case-slot positions at unit spacing, the six
# estimators dodged inside each, the case names annotated above the panel
# with light separators between cases.  Matches fig_estimator_err.
SLOT, GAP = 1.0, 0.7
pos, x = {}, 0.0
for case in cases:
    for sl in SLOTS:
        pos[(case,) + sl] = x
        x += SLOT
    x += GAP

fig, ax = plt.subplots(figsize=(13.6, 3.6))
ax.axhline(0, color='0.85', lw=0.7, zorder=0)
for case in cases:
    for (axname, dirn) in SLOTS:
        k = (case, axname, dirn)
        xc = pos[k]
        r = pred[k]
        pA, pS, pI = (float(r['M_pred']), float(r['M_pred_single']),
                      float(r['M_pred_interf']))
        ax.plot([xc - 0.34, xc + 0.34], [pA, pA], color='k', lw=1.5,
                zorder=2)
        ax.plot([xc - 0.34, xc + 0.34], [pS, pS], color='0.45', lw=1.2,
                ls=(0, (1.6, 1.6)), zorder=2)
        ax.plot([xc - 0.36, xc + 0.36], [pI, pI], color='#009E73', lw=2.0,
                ls=(0, (4.5, 1.8)), zorder=2)
        first = (k == (cases[0],) + SLOTS[0])
        for mi, m in enumerate(M):
            val = (float(r['M_ident']) if m == 'cosh'
                   else bmean[k].get(m))
            if val is None:
                continue
            ax.plot(xc - 0.24 + mi * 0.12, val, MRK[m], color=COL[m],
                    ms=5.5, zorder=3, label=LBL[m] if first else None)

ax.set_xticks([pos[(c,) + sl] for c in cases for sl in SLOTS])
ax.set_xticklabels(XL * len(cases), fontsize=8)
ax.set_xlim(pos[(cases[0],) + SLOTS[0]] - 0.75,
            pos[(cases[-1],) + SLOTS[-1]] + 0.75)
ax.set_ylabel(r'$M_{\mathrm{crit}}$ [N$\cdot$m]')
# the horizontal grid is what the thresholds are read against
ax.grid(axis='y', alpha=0.55, lw=0.9, color='0.55')
ax.grid(axis='x', alpha=0.20, lw=0.6, color='0.6')
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ylim = ax.get_ylim()
span = ylim[1] - ylim[0]
ax.set_ylim(ylim[0], ylim[1] + 0.13 * span)
ytxt = ylim[1] + 0.045 * span
for ci, case in enumerate(cases):
    xc = 0.5 * (pos[(case,) + SLOTS[0]] + pos[(case,) + SLOTS[-1]])
    ax.text(xc, ytxt, f'Case 0{ci + 1}', ha='center', va='bottom',
            fontsize=10)
    if ci:
        ax.axvline(pos[(case,) + SLOTS[0]] - 0.5 * (SLOT + GAP),
                   color='0.85', lw=0.8, zorder=0)

h, l = ax.get_legend_handles_labels()
import matplotlib.lines as mlines

h += [mlines.Line2D([], [], color='k', lw=1.5),
      mlines.Line2D([], [], color='0.45', lw=1.2, ls=(0, (1.6, 1.6))),
      mlines.Line2D([], [], color='#009E73', lw=2.0, ls=(0, (4.5, 1.8)))]
l += ['no ground effect', 'GE without rotor interference',
      'GE with rotor interference (parameter-free)']
fig.legend(h, l, loc='upper center', ncol=5, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, 1.10))
fig.tight_layout(rect=(0, 0, 1, 0.94))
args.outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(args.outdir / 'fig_mcrit_static.pdf', bbox_inches='tight')
fig.savefig(args.outdir / 'fig_mcrit_static.png', dpi=args.dpi,
            bbox_inches='tight')
print(f'figure saved to {args.outdir}')
