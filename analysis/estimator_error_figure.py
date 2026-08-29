#!/usr/bin/env python3
"""Per-case CoM-offset error figures.

exp_estimator_err: the marker is the error of the delivered estimate
(pair average of the two directional group means) and the whisker its
Welch-t 95% confidence interval.

exp_estimator_indiv: a parity view, estimated against true offset
with the y = x diagonal: the individual per-rate paired estimates as
small translucent dots and the delivered mean estimate as the large
marker, so the spread the group averaging removes is read directly
against the diagonal.

Reads nls_comparison_runs.csv (analysis/nls_comparison.py output).

Usage
-----
  python analysis/estimator_error_figure.py <scratch>
         [--outdir DIR] [--dpi N]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

G = 9.81
MASS = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
        'case_04': 3.220, 'case_05': 3.220}
TRUTH = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
         ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
         ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
         ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
         ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
SIGN = {'Mx': 1.0, 'My': -1.0}
M = ['cosh', 'cosh_cad', 'nls', 'pelt_normal', 'pelt_rbf', 'cusum']
LBL = {'cosh': 'COSH', 'cosh_cad': 'COSH-CAD', 'nls': 'PNLS',
       'pelt_normal': 'CPD-N', 'pelt_rbf': 'CPD-R', 'cusum': 'CUSUM'}
COL = {'cosh': '#0072B2', 'cosh_cad': '#56B4E9', 'nls': '#E69F00',
       'pelt_normal': '#009E73', 'pelt_rbf': '#D55E00',
       'cusum': '#CC79A7'}
MRK = {'cosh': 'o', 'cosh_cad': 'X', 'nls': 's', 'pelt_normal': '^',
       'pelt_rbf': 'v', 'cusum': 'D'}
CASES = [f'case_0{i}' for i in range(1, 6)]
SLOT, GAP = 1.0, 0.55


# the parity view carries the individual-vs-mean message for the
# deployed readout alone; the six-method comparison lives in
# exp_estimator_err
PARITY_M = ['cosh']


def draw_parity(est, out, dpi):
    """Estimated against true offset, individuals and means."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lim = 20
    ax.plot([-lim, lim], [-lim, lim], color='0.55', lw=1.0, ls='--',
            zorder=1)
    for key in TRUTH:
        truth = TRUTH[key]
        for m in PARITY_M:
            me, half, indiv = est(key, m)
            first = key == ('case_01', 'Mx')
            ax.plot([truth] * len(indiv), indiv + truth, '.', ms=4.2,
                    color=COL[m], alpha=0.45, mec='none', zorder=2,
                    label='individual estimate (per ramp rate)'
                    if first else None)
            ax.plot(truth, me + truth, MRK[m], color=COL[m], ms=6.0,
                    zorder=3, label='delivered estimate (group mean)'
                    if first else None)
    ax.set_xlabel('true CoM offset [mm]', fontsize=10)
    ax.set_ylabel('estimated CoM offset [mm]', fontsize=10)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def draw(est, mode, out, dpi):
    """One slot-dodged panel; mode 'ci' or 'indiv'."""
    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    pos, x = {}, 0.0
    for case in CASES:
        for axname in ('My', 'Mx'):     # x_off from pitch, y_off roll
            pos[(case, axname)] = x
            x += SLOT
        x += GAP

    ax.axhline(0, color='0.35', lw=0.9, zorder=1)
    for case in CASES:
        for axname in ('My', 'Mx'):
            key = (case, axname)
            xc = pos[key]
            for k, m in enumerate(M):
                me, half, indiv = est(key, m)
                xm = xc + (k - 2.5) * 0.115
                lab = LBL[m] if key == (CASES[0], 'My') else None
                if mode == 'ci':
                    ax.errorbar(xm, me, yerr=half, fmt=MRK[m],
                                color=COL[m], ms=5.5, elinewidth=1.2,
                                capsize=2.0, zorder=3, label=lab)
                else:
                    ax.plot([xm] * len(indiv), indiv, '.', ms=3.4,
                            color=COL[m], alpha=0.45, mec='none',
                            zorder=2)
                    ax.plot(xm, me, MRK[m], color=COL[m], ms=5.5,
                            zorder=3, label=lab)

    ax.set_xticks([pos[(c, a)] for c in CASES for a in ('My', 'Mx')])
    ax.set_xticklabels([r'$x_{\mathrm{off}}$', r'$y_{\mathrm{off}}$']
                       * len(CASES), fontsize=9)
    ax.set_xlim(pos[(CASES[0], 'My')] - 0.75,
                pos[(CASES[-1], 'Mx')] + 0.75)
    ax.set_ylabel('CoM offset error [mm]')
    ax.grid(axis='y', alpha=0.55, lw=0.9, color='0.55')
    ax.grid(axis='x', alpha=0.20, lw=0.6, color='0.6')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + 0.13 * span)
    ytxt = ylim[1] + 0.045 * span
    for ci_, case in enumerate(CASES):
        xc = 0.5 * (pos[(case, 'My')] + pos[(case, 'Mx')])
        ax.text(xc, ytxt, f'E{ci_ + 1}', ha='center', va='bottom',
                fontsize=10)
        if ci_:
            ax.axvline(pos[(case, 'My')] - 0.5 * (SLOT + GAP),
                       color='0.85', lw=0.8, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scratch', nargs='?', default='.',
                    help='directory holding nls_comparison_runs.csv')
    ap.add_argument('--outdir', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'docs')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(
        open(Path(args.scratch) / 'nls_comparison_runs.csv')))
    # (case, axis) -> method -> dir -> {rate: [moments]}
    agg = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        for m in M:
            if r.get(f'mcrit_{m}'):
                agg[(r['case'], r['axis'])][m][r['dir']][
                    float(r['rate'])].append(float(r[f'mcrit_{m}']))

    def est(key, m):
        """Mean error, its Welch-t 95% half-width, and the
        individual per-rate paired estimate errors."""
        g = agg[key][m]
        W = MASS[key[0]] * G
        s = SIGN[key[1]] * 1e3 / W
        p = np.concatenate([np.array(v) for v in g['pos'].values()])
        n = np.concatenate([np.array(v) for v in g['neg'].values()])
        me = s * 0.5 * (p.mean() + n.mean()) - TRUTH[key]
        var = .25 * (p.var(ddof=1) / len(p) + n.var(ddof=1) / len(n))
        num = (p.var(ddof=1) / len(p) + n.var(ddof=1) / len(n)) ** 2
        den = ((p.var(ddof=1) / len(p)) ** 2 / (len(p) - 1)
               + (n.var(ddof=1) / len(n)) ** 2 / (len(n) - 1))
        half = stats.t.ppf(.975, num / den) * abs(s) * W * np.sqrt(var) / W
        indiv = np.array(
            [s * 0.5 * (np.mean(g['pos'][rt]) + np.mean(g['neg'][rt]))
             - TRUTH[key]
             for rt in sorted(set(g['pos']) & set(g['neg']))])
        return me, half, indiv

    draw(est, 'ci', args.outdir / 'exp_estimator_err.png', args.dpi)
    draw_parity(est, args.outdir / 'exp_estimator_indiv.png',
                args.dpi)
    print(f'saved exp_estimator_err and exp_estimator_indiv '
          f'to {args.outdir}')


if __name__ == '__main__':
    main()
