#!/usr/bin/env python3
"""E2 showcase: the critical moments and what they deliver.

Two presentation-oriented artefacts for the single worst-offset case
(E2, the largest true offset of the campaign):

  exp_e2_mcrit.png    the detected critical moment of all six
                      detectors for the four direction groups, with
                      the static thresholds of (7)/(14) under the
                      three ground-effect treatments
  tab_e2_estimates    the CoM offset and balanced moment each
                      detector delivers, against the load-cell truth

Both read nls_comparison_runs.csv (analysis/nls_comparison.py) and
mcrit_prediction.csv (analysis/mcrit_prediction.py).

Usage
-----
  python analysis/e2_showcase.py <scratch> [--case case_02]
         [--outdir DIR] [--dpi N]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

G = 9.81
MASS = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
        'case_04': 3.220, 'case_05': 3.220}
TRUTH = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
         ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
         ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
         ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
         ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
# Mx senses the y offset, My the x offset (with a sign)
SIGN = {'Mx': 1.0, 'My': -1.0}
M = ['cosh', 'cosh_cad', 'nls', 'pelt_normal', 'pelt_rbf', 'cusum']
LBL = {'cosh': 'COSH', 'cosh_cad': 'COSH-CAD', 'nls': 'PNLS',
       'pelt_normal': 'CPD-N', 'pelt_rbf': 'CPD-R', 'cusum': 'CUSUM'}
COL = {'cosh': '#0072B2', 'cosh_cad': '#56B4E9', 'nls': '#E69F00',
       'pelt_normal': '#009E73', 'pelt_rbf': '#D55E00',
       'cusum': '#CC79A7'}
MRK = {'cosh': 'o', 'cosh_cad': 'X', 'nls': 's', 'pelt_normal': '^',
       'pelt_rbf': 'v', 'cusum': 'D'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scratch', type=Path)
    ap.add_argument('--case', default='case_02')
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    ap.add_argument('--dpi', type=int, default=600)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    case, W = a.case, MASS[a.case] * G
    tag = 'E' + case[-1]

    runs = defaultdict(list)          # (axis, dir, method) -> [M]
    for r in csv.DictReader(
            open(a.scratch / 'nls_comparison_runs.csv')):
        if r['case'] != case:
            continue
        for m in M:
            if r.get(f'mcrit_{m}'):
                runs[(r['axis'], r['dir'], m)].append(
                    float(r[f'mcrit_{m}']))

    th = {}                            # (axis, dir) -> (none, sgl, int)
    for r in csv.DictReader(
            open(a.scratch / 'mcrit_prediction.csv')):
        if r['case'] != case:
            continue
        th[(r['axis'], r['dir'])] = (float(r['M_pred']),
                                     float(r['M_pred_single']),
                                     float(r['M_pred_interf']))

    # ── figure: the four direction groups side by side ───────────────
    slots = [('Mx', 'neg'), ('Mx', 'pos'), ('My', 'neg'), ('My', 'pos')]
    lab = [r'$M_{x,-}$', r'$M_{x,+}$', r'$M_{y,-}$', r'$M_{y,+}$']
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, key in enumerate(slots):
        if key in th:
            for v, ls, c, nm in zip(
                    th[key], ('-', ':', '--'),
                    ('0.15', '0.45', '#009E73'),
                    ('No ground effect', 'Ground effect (Single rotor)',
                     'Ground effect (Rotor interference)')):
                ax.plot([i - 0.42, i + 0.42], [v, v], ls, color=c,
                        lw=2.0, zorder=2,
                        label=nm if i == 0 else None)
        for k, m in enumerate(M):
            v = np.array(runs[(key[0], key[1], m)])
            if not v.size:
                continue
            lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
            x = i + (k - 2.5) * 0.115
            ax.errorbar(x, med, yerr=[[med - lo], [hi - med]],
                        fmt=MRK[m], ms=5.0, color=COL[m],
                        elinewidth=1.2, capsize=2.2, zorder=4,
                        label=LBL[m] if i == 0 else None)
    ax.axhline(0, color='0.7', lw=0.9, zorder=1)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels(lab, fontsize=12)
    ax.set_xlim(-0.6, len(slots) - 0.4)
    ax.set_ylabel(r'$M_{\mathrm{crit}}$ [N$\cdot$m]', fontsize=12)
    ax.set_title(f'{tag}', loc='left', fontsize=12)
    ax.grid(axis='y', alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = a.outdir / f'exp_{tag.lower()}_mcrit.png'
    fig.savefig(out, dpi=a.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'written {out}')

    # ── table: what each detector delivers on this case ──────────────
    rows = []
    for m in M:
        mb, off = {}, {}
        for axis in ('Mx', 'My'):
            p = np.mean(runs[(axis, 'pos', m)])
            n = np.mean(runs[(axis, 'neg', m)])
            mb[axis] = 0.5 * (p + n)
            off[axis] = SIGN[axis] * 1e3 * mb[axis] / W
        rows.append((m, mb['Mx'], mb['My'], off['My'], off['Mx']))

    tx, ty = TRUTH[(case, 'My')], TRUTH[(case, 'Mx')]
    out = a.outdir / f'tab_{tag.lower()}_estimates.tex'
    with open(out, 'w') as fh:
        fh.write(
            f'% {tag}: balanced moment and CoM offset per detector.\n'
            '% Generated by analysis/e2_showcase.py.\n'
            '\\begin{table}[t]\n'
            f'\\caption{{Balanced moment and CoM offset delivered for '
            f'case {tag} by each detector, against the load-cell '
            f'truth. The balanced moment is the direction-pair average '
            f'$\\tfrac12(M_{{\\mathrm{{crit}},+}}+'
            f'M_{{\\mathrm{{crit}},-}})$ per axis, and the offset '
            f'follows from it through the vehicle weight.}}\n'
            f'\\label{{tab:{tag.lower()}_estimates}}\n'
            '\\centering\\small\n'
            '\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n'
            '& \\multicolumn{2}{c}{balanced moment [N$\\cdot$m]}'
            ' & \\multicolumn{2}{c}{CoM offset [mm]}\\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n'
            'method & $M_{x,\\mathrm{bal}}$ & $M_{y,\\mathrm{bal}}$'
            ' & $x_{\\mathrm{off}}$ & $y_{\\mathrm{off}}$\\\\\n'
            '\\midrule\n')
        print(f'{"method":<10}{"Mx_bal":>9}{"My_bal":>9}'
              f'{"x_off":>9}{"y_off":>9}{"err":>8}')
        for m, mx, my, xo, yo in rows:
            e = float(np.hypot(xo - tx, yo - ty))
            fh.write(f'{LBL[m]} & ${mx:+.3f}$ & ${my:+.3f}$ & '
                     f'${xo:+.2f}$ & ${yo:+.2f}$\\\\\n')
            print(f'{LBL[m]:<10}{mx:+9.3f}{my:+9.3f}{xo:+9.2f}'
                  f'{yo:+9.2f}{e:8.2f}')
        fh.write('\\midrule\n'
                 f'load-cell truth & --- & --- & ${tx:+.2f}$ & '
                 f'${ty:+.2f}$\\\\\n'
                 '\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(f'{"truth":<10}{"":>9}{"":>9}{tx:+9.2f}{ty:+9.2f}')
    print(f'written {out}')


if __name__ == '__main__':
    main()
