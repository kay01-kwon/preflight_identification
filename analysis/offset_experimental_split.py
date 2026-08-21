#!/usr/bin/env python3
"""The measured counterpart of the offset-error theory sweep: what the
per-direction CoM-offset estimates do, and what the pair average does.

For each of the ten configurations the no-ground-effect estimator is
run on each tip direction separately,

    p_hat_s = sgn_axis [ M_s - s (W - f_s) l_s ] / W ,

and on the pair average, and each is compared with the load-cell
truth.  Two facts come out, and they are the experimental version of
what analysis/matlab/offset_ge_theory.m derives:

  * the single-direction estimates are off by 6-8 mm RMS, worst 12 mm,
    while the pair average is off by 1.8 mm RMS, worst 3.0 mm;
  * the two directions err in OPPOSITE senses on every one of the ten
    configurations -- the antisymmetric signature the pairing cancels.

The ground effect predicted by the interference model accounts for
roughly half of the single-direction spread (dashed lines); the rest
is other antisymmetric error -- contact lever, arm fit -- which the
same pairing removes just as effectively, which is why the average
lands inside the load-cell validation RMS.

Usage: python analysis/offset_experimental_split.py <dir with csv> [out.png]
"""
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = 0.0431                       # interference gains at phi* = 0
OFF = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
       ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
       ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
       ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
       ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
SGN = {'Mx': +1.0, 'My': -1.0}       # S_off = +W y_off / -W x_off
C = {'p': '#c0392b', 'n': '#1a5276', 'a': '#148f77'}


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    out = sys.argv[2] if len(sys.argv) > 2 else 'offset_experimental.png'
    rows = list(csv.DictReader(open(d / 'mcrit_prediction.csv')))
    by = {}
    for r in rows:
        by.setdefault((r['case'], r['axis']), {})[r['dir']] = r

    tru, ep, en, ea, tp, tn, ta = ([] for _ in range(7))
    for key, grp in sorted(by.items()):
        W = float(grp['pos']['W'])
        sa = SGN[key[1]]
        p = OFF[key] * 1e-3
        est, the = {}, {}
        for dn, s in (('pos', +1.0), ('neg', -1.0)):
            r = grp[dn]
            M = float(r['M_ident'])
            f = float(r['f_onset'])
            l = float(r['l_odom_mm']) * 1e-3
            est[dn] = sa * (M - s * (W - f) * l) / W
            # same estimator on GE-biased moments generated from truth
            m_ge = (s * (W - f) * l + sa * W * p - s * ALPHA * f * l) \
                / (1.0 + ALPHA)
            the[dn] = sa * (m_ge - s * (W - f) * l) / W
        tru.append(1e3 * p)
        ep.append(1e3 * (est['pos'] - p))
        en.append(1e3 * (est['neg'] - p))
        ea.append(1e3 * (0.5 * (est['pos'] + est['neg']) - p))
        tp.append(1e3 * (the['pos'] - p))
        tn.append(1e3 * (the['neg'] - p))
        ta.append(1e3 * (0.5 * (the['pos'] + the['neg']) - p))
    tru, ep, en, ea = map(np.array, (tru, ep, en, ea))
    tp, tn, ta = map(np.array, (tp, tn, ta))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.13,
                        wspace=0.26)

    # ---- (a) errors against the truth offset ----
    a1.axhspan(-1.64, 1.64, color='0.90', lw=0, zorder=0,
               label='load-cell validation RMS $\\pm 1.64$ mm')
    a1.axhline(0, color='0.5', lw=0.8)
    # ground-effect-only prediction, per configuration: drawn as open
    # markers, not a line -- the two axes carry different arms, so a
    # curve through them sorted by truth would zigzag meaninglessly
    for v, c, mk in ((tp, C['p'], '^'), (tn, C['n'], 'v'),
                     (ta, C['a'], 'o')):
        a1.scatter(tru, v, s=46, facecolors='none', edgecolors=c,
                   marker=mk, lw=1.0, alpha=0.85)
    a1.scatter([], [], s=46, facecolors='none', edgecolors='0.35',
               marker='s', lw=1.0,
               label='open: ground effect alone (interference model)')
    a1.scatter(tru, ep, s=44, c=C['p'], marker='^', lw=0,
               label=f'$\\hat{{p}}_{{\\rm off,+}}$   RMS '
                     f'{np.sqrt(np.mean(ep**2)):.1f} mm')
    a1.scatter(tru, en, s=44, c=C['n'], marker='v', lw=0,
               label=f'$\\hat{{p}}_{{\\rm off,-}}$   RMS '
                     f'{np.sqrt(np.mean(en**2)):.1f} mm')
    a1.scatter(tru, ea, s=52, c=C['a'], marker='o', lw=0,
               label=f'$\\hat{{p}}_{{\\rm off,avg}}$  RMS '
                     f'{np.sqrt(np.mean(ea**2)):.1f} mm')
    a1.set_xlabel('load-cell truth $p_{\\rm off}$ [mm]', fontsize=10)
    a1.set_ylabel('$\\hat{p}_{\\rm off} - p_{\\rm off}$ [mm]', fontsize=10)
    a1.set_title('(a) measured offset error, ten configurations\n'
                 'single directions 6-8 mm RMS, the pair average 1.8',
                 fontsize=11)
    a1.legend(fontsize=7.5, loc='upper right', framealpha=0.95,
              ncol=1)
    a1.grid(alpha=0.25, lw=0.4)

    # ---- (b) the antisymmetry the pairing exploits ----
    lim = 1.15 * max(np.abs(ep).max(), np.abs(en).max())
    a2.plot([-lim, lim], [lim, -lim], 'k-', lw=1.0,
            label='$e_- = -e_+$ (perfect cancellation)')
    a2.axhline(0, color='0.6', lw=0.7)
    a2.axvline(0, color='0.6', lw=0.7)
    a2.scatter(ep, en, s=52, c=C['a'], lw=0)
    for x, y, k in zip(ep, en, sorted(by)):
        a2.annotate(k[0][-2:] + '/' + k[1][-1], (x, y), fontsize=6.5,
                    xytext=(4, 3), textcoords='offset points',
                    color='0.35')
    r = float(np.corrcoef(ep, en)[0, 1])
    a2.set_xlabel('$+$ direction error $e_+$ [mm]', fontsize=10)
    a2.set_ylabel('$-$ direction error $e_-$ [mm]', fontsize=10)
    a2.set_title('(b) why the average works: the two directions err\n'
                 f'in opposite senses on all ten ($r = {r:+.2f}$)',
                 fontsize=11)
    a2.legend(fontsize=8.5, loc='upper right')
    a2.grid(alpha=0.25, lw=0.4)
    a2.set_xlim(-lim, lim)
    a2.set_ylim(-lim, lim)

    fig.suptitle('Per-direction versus paired CoM-offset estimates, '
                 'measured (filled) against the ground-effect '
                 'prediction (open)', fontsize=12.5, y=0.975)
    fig.savefig(out, dpi=150)

    for nm, v, t in (('+ direction', ep, tp), ('- direction', en, tn),
                     ('pair average', ea, ta)):
        print(f"  {nm:<13} RMS {np.sqrt(np.mean(v**2)):5.2f} mm, "
              f"worst {np.abs(v).max():5.2f};  GE alone predicts RMS "
              f"{np.sqrt(np.mean(t**2)):4.2f}")
    print(f"  corr(e+, e-) = {r:+.3f};  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
