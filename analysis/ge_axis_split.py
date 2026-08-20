#!/usr/bin/env python3
"""The trusted-span GE level test, split by axis (and tip direction).

Roll (Mx) pivots about a gear line; pitch (My) rides a different
contact geometry, so the agreement with the rotor-interference model
need not be shared.  Same construction as ge_trusted_span.py
(deployed SG 9/order 2, trim = half-width, relative attitude via
q_rest); only the reporting is split.

Usage: PYTHONPATH=<stubs> python analysis/ge_axis_split.py [out.png]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.ge_trusted_span import collect, K_TRIM, W_SG, SG_P


def agg(rows):
    PH, GD, GM, d = [], [], [], []
    for r in rows:
        ph, gd, gm = r['trace']
        e = len(ph) - K_TRIM
        if e < 10:
            continue
        PH.append(ph[:e]); GD.append(gd[:e]); GM.append(gm[:e])
        d.append(float(np.mean(gd[:e] - gm[:e])))
    PH, GD, GM = map(np.concatenate, (PH, GD, GM))
    return PH, GD, GM, np.array(d)


def band(PH, V, bins):
    cx, md, q1, q3 = [], [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (PH >= b0) & (PH < b1)
        if m.sum() < 80:
            continue
        cx.append(0.5 * (b0 + b1))
        md.append(np.median(V[m]))
        q1.append(np.percentile(V[m], 25)); q3.append(np.percentile(V[m], 75))
    return cx, md, q1, q3


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_axis_split.png'
    rows = collect()
    print(f"\n  {len(rows)} runs;  w = {W_SG}, order {SG_P}, "
          f"trim k = {K_TRIM}\n")

    print(f"  {'group':<12}{'n':>4}{'model lvl':>11}{'median d':>10}"
          f"{'IQR':>19}{'RMS':>7}{'|d|<100':>9}")
    groups = [('roll  (Mx)', lambda r: r['axisname'] == 'Mx'),
              ('pitch (My)', lambda r: r['axisname'] == 'My'),
              ('Mx pos', lambda r: r['axisname'] == 'Mx' and r['dir'] == 'pos'),
              ('Mx neg', lambda r: r['axisname'] == 'Mx' and r['dir'] == 'neg'),
              ('My pos', lambda r: r['axisname'] == 'My' and r['dir'] == 'pos'),
              ('My neg', lambda r: r['axisname'] == 'My' and r['dir'] == 'neg')]
    packs = {}
    for lab, sel in groups:
        sub = [r for r in rows if sel(r)]
        PH, GD, GM, d = agg(sub)
        lvl = np.median([np.mean(r['trace'][2]) for r in sub])
        packs[lab] = (PH, GD, GM, d)
        print(f"  {lab:<12}{len(d):>4}{lvl:>11.1f}{np.median(d):>+10.1f}"
              f"   [{np.percentile(d,25):+6.1f},{np.percentile(d,75):+6.1f}]"
              f"{np.sqrt(np.mean(d**2)):>7.0f}"
              f"{int(np.sum(np.abs(d)<100)):>6d}/{len(d)}")

    fig, axs2 = plt.subplots(2, 2, figsize=(12.4, 9.4))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.885, bottom=0.065,
                        wspace=0.24, hspace=0.33)
    axs = axs2[0]
    for ax, lab, ttl in [(axs[0], 'roll  (Mx)', '(a) roll (Mx): pivot on a gear line'),
                         (axs[1], 'pitch (My)', '(b) pitch (My): the other contact geometry')]:
        PH, GD, GM, d = packs[lab]
        bins = np.arange(0.0, PH.max() + 0.25, 0.25)
        cx, md, q1, q3 = band(PH, GD, bins)
        ax.fill_between(cx, q1, q3, color='#c0392b', alpha=0.22, lw=0)
        ax.plot(cx, md, '-', color='#c0392b', lw=2.0,
                label='dynamic inversion: median, IQR')
        cxm, mdm, q1m, q3m = band(PH, GM, bins)
        ax.fill_between(cxm, q1m, q3m, color='#e08214', alpha=0.35, lw=0)
        ax.plot(cxm, mdm, '-', color='#e08214', lw=2.0,
                label='rotor-interference model: median, IQR')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xlim(left=0.0)
        ax.set_ylim(-220, 560)
        ax.set_xlabel(r'excursion $\delta\varphi$ [deg]', fontsize=10)
        ax.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
        ax.set_title(f'{ttl}\nmedian $d$ {np.median(d):+.0f} mN$\\cdot$m, '
                     f'$|d|{{<}}100$: {int(np.sum(np.abs(d)<100))}/{len(d)}',
                     fontsize=11)
        ax.legend(fontsize=8.5, loc='upper left')
        ax.grid(alpha=0.22, lw=0.4)
    # ---- bottom row: per-axis histograms of the per-run mean diff ----
    hbins = np.linspace(-400, 400, 33)
    for ax, axn, lab, ttl in [
            (axs2[1][0], 'Mx', 'roll  (Mx)', '(c) roll: per-run mean difference'),
            (axs2[1][1], 'My', 'pitch (My)', '(d) pitch: per-run mean difference')]:
        _, _, _, d_all = packs[lab]
        dp = packs[f'{axn} pos'][3]
        dn = packs[f'{axn} neg'][3]
        ax.hist(np.clip(dp, hbins[0], hbins[-1]), bins=hbins,
                color='#1a5276', alpha=0.75,
                label=f'pos tip (median {np.median(dp):+.0f})')
        ax.hist(np.clip(dn, hbins[0], hbins[-1]), bins=hbins,
                color='#c0392b', alpha=0.55,
                label=f'neg tip (median {np.median(dn):+.0f})')
        ax.axvline(0, color='#e08214', lw=2.0, label='model')
        ax.axvline(float(np.median(d_all)), color='k', lw=1.2, ls='--',
                   label=f'axis median {np.median(d_all):+.0f}')
        ax.set_xlabel(r'per-run mean of $\Delta M_{GE}^{dyn} - '
                      r'\Delta M_{GE}^{model}$ [mN$\cdot$m]', fontsize=10)
        ax.set_ylabel('runs', fontsize=10)
        ax.set_title(f'{ttl}\n'
                     f'RMS {np.sqrt(np.mean(d_all**2)):.0f} mN$\\cdot$m, '
                     f'$|d|{{<}}100$: {int(np.sum(np.abs(d_all)<100))}'
                     f'/{len(d_all)}', fontsize=11)
        ax.legend(fontsize=8.5, loc='upper left')
        ax.grid(alpha=0.22, lw=0.4, axis='y')

    fig.suptitle('Trusted-span GE level test split by axis '
                 f'($w={W_SG}$, order {SG_P}, trim {K_TRIM}, '
                 'relative attitude)',
                 fontsize=12, y=0.975)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
