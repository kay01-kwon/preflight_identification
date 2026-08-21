#!/usr/bin/env python3
"""The static GE result, reported: parity plot + physics ladder.

(a) Per-run parity: identified M_crit against the thrust-matched
    interference prediction M_theory/(1 + alpha_M); the diagonal is
    perfect agreement, the shaded band the static repeatability.

(b) The physics ladder: |residual| distributions as each ingredient is
    added -- no GE, single-rotor GE, interference GE without the
    thrust-channel matching, and the full thrust-matched interference
    prediction.  Every step is one physical ingredient, and the last
    (matching the reference's total thrust) is what this figure argues
    for: median 147 -> 59 mN.m.

Usage: python analysis/mcrit_report_figure.py <dir with csvs> [out.png]
"""
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = 0.04314


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out = sys.argv[2] if len(sys.argv) > 2 else 'mcrit_report.png'
    rows = list(csv.DictReader(open(d / 'mcrit_per_run.csv')))
    mi = np.array([float(r['M_ident']) for r in rows])
    rn = np.array([float(r['resid_none_mNm']) for r in rows]) * 1e-3
    rs = np.array([float(r['resid_single_mNm']) for r in rows]) * 1e-3
    ri = np.array([float(r['resid_interf_mNm']) for r in rows]) * 1e-3
    ax_is_x = np.array([r['axis'] == 'Mx' for r in rows])
    T = mi - rn                                   # rigid threshold per run
    r_unm = mi - T / (1.0 + B)                    # matching dropped
    pred = mi - ri                                # matched interference

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.835, bottom=0.115,
                        wspace=0.26)

    lim = 1.15 * max(np.abs(mi).max(), np.abs(pred).max())
    a1.plot([-lim, lim], [-lim, lim], 'k-', lw=1.0, label='$y = x$')
    a1.fill_between([-lim, lim], [-lim - .088, lim - .088],
                    [-lim + .088, lim + .088], color='0.88', lw=0,
                    zorder=0, label=r'$\pm 88$ mN$\cdot$m repeatability')
    a1.scatter(pred[ax_is_x], mi[ax_is_x], s=26, c='#1a5276', alpha=0.75,
               lw=0, label=f'roll runs (n={int(ax_is_x.sum())})')
    a1.scatter(pred[~ax_is_x], mi[~ax_is_x], s=30, c='#c0392b', alpha=0.75,
               lw=0, marker='^',
               label=f'pitch runs (n={int((~ax_is_x).sum())})')
    a1.set_xlabel(r'thrust-matched prediction '
                  r'$M_{\rm theory}/(1+\alpha_M)$ [N$\,$m]', fontsize=10)
    a1.set_ylabel(r'identified $M_{\rm crit}$ [N$\,$m]', fontsize=10)
    rmsi = 1e3 * float(np.sqrt(np.mean(ri ** 2)))
    a1.set_title(f'(a) per-run parity, {len(rows)} runs:\n'
                 f'RMS residual {rmsi:.0f} mN$\\cdot$m about the diagonal',
                 fontsize=11)
    a1.legend(fontsize=8.5, loc='upper left')
    a1.grid(alpha=0.22, lw=0.4)
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)

    lad = [('no ground effect', np.abs(rn)),
           ('single-rotor GE', np.abs(rs)),
           ('interference GE,\nthrust unmatched', np.abs(r_unm)),
           ('interference GE,\nthrust matched', np.abs(ri))]
    data = [1e3 * v for _, v in lad]
    bp = a2.boxplot(data, vert=False, widths=0.55, showfliers=True,
                    patch_artist=True,
                    flierprops=dict(marker='.', ms=4, alpha=0.5),
                    medianprops=dict(color='k', lw=1.4))
    for patch, c in zip(bp['boxes'],
                        ['0.80', '#a9cce3', '#f5b041', '#148f77']):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    for i, v in enumerate(data):
        a2.text(np.median(v) + 8, i + 1 + 0.32, f'{np.median(v):.0f}',
                fontsize=9, va='bottom', fontweight='bold')
    a2.set_yticks(range(1, 5))
    a2.set_yticklabels([n for n, _ in lad], fontsize=9)
    a2.set_xlabel(r'$|M_{\rm ident} - M_{\rm pred}|$ [mN$\cdot$m]',
                  fontsize=10)
    a2.set_title('(b) the physics ladder: each ingredient, one step\n'
                 'medians annotated; the last step is the matching',
                 fontsize=11)
    a2.grid(alpha=0.22, lw=0.4, axis='x')
    a2.invert_yaxis()

    fig.suptitle('Static critical moment against the thrust-matched '
                 r'interference prediction $M_{\rm crit} = '
                 r'M_{\rm theory}/(1+\alpha_M)$', fontsize=12.5, y=0.975)
    fig.savefig(out, dpi=150)
    for n, v in lad:
        print(f"  {n.replace(chr(10), ' '):34s} median "
              f"{1e3*np.median(v):6.1f}  p90 {1e3*np.percentile(v,90):6.1f}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
