#!/usr/bin/env python3
"""Static-threshold check figure and table rows (odom arms, GE on/off).

Reads mcrit_prediction.csv (per-group odom-arm predictions, both thrust
scenarios) and nls_comparison_runs.csv (benchmark estimators) from
argv[1]; writes docs/fig_mcrit_static.pdf and prints the LaTeX rows.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SC = sys.argv[1] if len(sys.argv) > 1 else '.'
M = ['cosh', 'nls', 'pelt_normal', 'pelt_rbf', 'cusum']
LBL = {'cosh': 'COSH', 'nls': 'NLS', 'pelt_normal': 'CPD-N',
       'pelt_rbf': 'CPD-R', 'cusum': 'CUSUM'}
COL = {'cosh': '#0072B2', 'nls': '#E69F00', 'pelt_normal': '#009E73',
       'pelt_rbf': '#D55E00', 'cusum': '#CC79A7'}
MRK = {'cosh': 'o', 'nls': 's', 'pelt_normal': '^', 'pelt_rbf': 'v',
       'cusum': 'D'}

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

print("% rows: case ax dir l_odom M_cosh pred:none r:none r:sgl r:gar inBr bench")
for k in sorted(pred):
    r = pred[k]
    bs = [bmean[k][m] for m in M[1:] if m in bmean[k]]
    c = k[0].replace('case_0', '')
    chk = '$\\checkmark$' if r['in_bracket'] == 'True' else '--'
    row = (f"{c} & {k[1][1]} & {'+' if k[2] == 'pos' else '-'} & "
           f"{r['l_odom_mm']} & ${float(r['M_ident']):+.3f}$ & "
           f"${float(r['M_pred']):+.3f}$ & ${float(r['resid_mNm']):+.0f}$ & "
           f"${float(r['resid_single_mNm']):+.0f}$ & "
           f"${float(r['resid_interf_mNm']):+.0f}$ & "
           f"${float(r['resid_garofano_mNm']):+.0f}$ & " + chk +
           f" & $[{min(bs):+.3f}, {max(bs):+.3f}]$ \\\\")
    print(row)

SLOTS = [('Mx', 'neg'), ('Mx', 'pos'), ('My', 'neg'), ('My', 'pos')]
XL = [r'$M_{x,-}$', r'$M_{x,+}$', r'$M_{y,-}$', r'$M_{y,+}$']
cases = [f'case_0{i}' for i in range(1, 6)]
fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2), sharey=True)
for ci, case in enumerate(cases):
    ax = axes[ci]
    ax.axhline(0, color='0.85', lw=0.7, zorder=0)
    for xi, (axname, dirn) in enumerate(SLOTS):
        r = pred[(case, axname, dirn)]
        lo, hi = sorted((float(r['M_pred_single']),
                         float(r['M_pred_garofano'])))
        ax.add_patch(plt.Rectangle((xi - 0.34, lo), 0.68, hi - lo,
                                   color='0.85', zorder=1))
        pA = float(r['M_pred'])
        ax.plot([xi - 0.34, xi + 0.34], [pA, pA], color='k', lw=1.6,
                zorder=2)
        pI = float(r['M_pred_interf'])
        ax.plot([xi - 0.34, xi + 0.34], [pI, pI], color='0.25', lw=1.3,
                ls=(0, (4, 2)), zorder=2)
        k = (case, axname, dirn)
        for mi, m in enumerate(M):
            val = (float(r['M_ident']) if m == 'cosh'
                   else bmean[k].get(m))
            if val is None:
                continue
            ax.plot(xi - 0.24 + mi * 0.12, val, MRK[m], color=COL[m],
                    ms=4.2, zorder=3,
                    label=LBL[m] if (ci == 0 and xi == 0) else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(XL, fontsize=8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_title(f'Case 0{ci + 1}', fontsize=10)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel(r'$M_{\mathrm{crit}}$ [N$\cdot$m]')
h, l = axes[0].get_legend_handles_labels()
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
h += [mlines.Line2D([], [], color='k', lw=1.6),
      mlines.Line2D([], [], color='0.25', lw=1.3, ls=(0, (4, 2))),
      mpatches.Patch(color='0.85')]
l += ['rigid no-GE prediction (odometry-fitted arms)',
      'parameter-free interference model',
      'GE bracket [single, Garofano]']
fig.legend(h, l, loc='upper center', ncols=7, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('/home/user/preflight_identification/docs/fig_mcrit_static.pdf',
            bbox_inches='tight')
print('figure saved')
