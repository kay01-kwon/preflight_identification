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

    D = {'Mx': {k: [] for k in 'tru ep en ea tp tn ta lab'.split()},
         'My': {k: [] for k in 'tru ep en ea tp tn ta lab'.split()}}
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
        q = D[key[1]]
        q['tru'].append(1e3 * p)
        q['ep'].append(1e3 * (est['pos'] - p))
        q['en'].append(1e3 * (est['neg'] - p))
        q['ea'].append(1e3 * (0.5 * (est['pos'] + est['neg']) - p))
        q['tp'].append(1e3 * (the['pos'] - p))
        q['tn'].append(1e3 * (the['neg'] - p))
        q['ta'].append(1e3 * (0.5 * (the['pos'] + the['neg']) - p))
        q['lab'].append(key[0][-2:])
        q.setdefault('geom', []).append(
            (W, sa, float(grp['pos']['f_onset']),
             float(grp['pos']['l_odom_mm']) * 1e-3,
             float(grp['neg']['f_onset']),
             float(grp['neg']['l_odom_mm']) * 1e-3))
    for q in D.values():
        for k in 'tru ep en ea tp tn ta'.split():
            q[k] = np.array(q[k])

    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.12,
                        wspace=0.07)

    ttl = {'Mx': '(a) roll ($M_x$): $y_{\\rm off}$, arm $\\approx 140$ mm',
           'My': '(b) pitch ($M_y$): $x_{\\rm off}$, arm $\\approx 113$ mm'}
    grp = [('$\\hat{p}_{\\rm off,+}$', 'ep', 'tp'),
           ('$\\hat{p}_{\\rm off,-}$', 'en', 'tn'),
           ('$\\hat{p}_{\\rm off,avg}$', 'ea', 'ta')]
    for ax, an in zip(axs, ('Mx', 'My')):
        q = D[an]
        ax.axhspan(-1.64, 1.64, color='0.90', lw=0, zorder=0)
        ax.axhline(0, color='0.5', lw=0.8)
        for i, (nm, mk, tk) in enumerate(grp):
            mv, tv = q[mk], q[tk]
            ax.errorbar(i - 0.16, mv.mean(), yerr=mv.std(ddof=1),
                        fmt='o', ms=8, color=C['a'], capsize=5, lw=1.6,
                        label='measured' if i == 0 else None)
            ax.errorbar(i + 0.16, tv.mean(), yerr=tv.std(ddof=1),
                        fmt='s', ms=7, color='#e08214', capsize=5,
                        lw=1.6,
                        label='ground effect alone' if i == 0 else None)
            ax.text(i, 13.0, f'RMS {np.sqrt(np.mean(mv**2)):.1f}',
                    ha='center', fontsize=8.5, color=C['a'])
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[0] for g in grp], fontsize=11)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-14, 15.5)
        ax.set_title(ttl[an], fontsize=11)
        ax.grid(alpha=0.25, lw=0.4, axis='y')
        if an == 'Mx':
            ax.legend(fontsize=9, loc='lower left', framealpha=0.95)
    axs[0].set_ylabel('$\\hat{p}_{\\rm off} - p_{\\rm off}$ [mm]',
                      fontsize=10.5)

    fig.suptitle('CoM-offset error by estimator and axis: mean $\\pm$ '
                 's.d. over five configurations\n'
                 'grey band: load-cell validation RMS $\\pm 1.64$ mm',
                 fontsize=11.5, y=0.975)
    fig.savefig(out, dpi=150)

    for an in ('Mx', 'My'):
        q = D[an]
        print(f"  --- {an} ---")
        for nm, k, tk in (('+ direction', 'ep', 'tp'),
                          ('- direction', 'en', 'tn'),
                          ('pair average', 'ea', 'ta')):
            v, t = q[k], q[tk]
            print(f"    {nm:<13} RMS {np.sqrt(np.mean(v**2)):5.2f} mm, "
                  f"worst {np.abs(v).max():5.2f};  GE alone "
                  f"{np.sqrt(np.mean(t**2)):4.2f}")
    ep = np.concatenate([D[a]['ep'] for a in D])
    en = np.concatenate([D[a]['en'] for a in D])
    ea = np.concatenate([D[a]['ea'] for a in D])
    tp = np.concatenate([D[a]['tp'] for a in D])
    tn = np.concatenate([D[a]['tn'] for a in D])
    ta = np.concatenate([D[a]['ta'] for a in D])
    r = float(np.corrcoef(ep, en)[0, 1])
    print("  --- pooled ---")
    for nm, v, t in (('+ direction', ep, tp), ('- direction', en, tn),
                     ('pair average', ea, ta)):
        print(f"    {nm:<13} RMS {np.sqrt(np.mean(v**2)):5.2f} mm, "
              f"worst {np.abs(v).max():5.2f};  GE alone predicts RMS "
              f"{np.sqrt(np.mean(t**2)):4.2f}")
    print(f"    corr(e+, e-) = {r:+.3f}")
    print(f"  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
