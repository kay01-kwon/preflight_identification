#!/usr/bin/env python3
"""fig_ge_dynamics_axis: the dynamic check, one panel per excitation axis.

The two axes are different experiments and pooling them hides both.
They tip about different contact lines (140 mm against 110), the model
predicts different moments (about 172 mN.m against 135), and the two
tip directions do not reach the same tilt on both axes:

    Mx/pos  median excursion 5.8 deg     Mx/neg  7.4
    My/pos                   6.8         My/neg  3.1

Each panel is the average of the two tip directions, which is the
combination M_ff = sign * 0.5 * (M_pos + M_neg) the identification
forms, and it is not optional here.  The directions carry an
antisymmetric term -- 470 mN.m apart on My, 80 on Mx -- so POOLING them
instead makes the curve follow whichever direction still has samples.
On My that manufactures a step: the pooled median jumps from 142 to 230
mN.m between 2.2 and 2.6 deg while neither direction's own median moves
at all, only their mix, 49% negative becoming 45 and then 14 by 4.2
deg.  The same mixing, not any property of the axis, is what makes the
pooled band three times wider on My.

Averaged properly, and stopping where the shorter direction runs out:

    Mx, 0.2-5.0 deg    residual median  -5.2, RMS 14.9, max 26
                       9% of the model, 66-70 runs in every bin
    My, 0.2-3.4 deg    residual median -45.3, RMS 49.0, max 66
                       36% of the model, 52-70 runs

So Mx tracks the model over its whole range and My sits about 45 mN.m
below it, flat, with no step anywhere.  My also carries the wider band
throughout, 118-217 against 107-129, and it is the axis whose 1-3 Hz
rigid-body balance does not close (analysis/rate_band_check.py, ratio
1.8-2.0 against 0.9-1.1).

The band is the interquartile range over direction pairs, dispersion
rather than an uncertainty on the median.

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/ge_dynamics_by_axis.py hd.npz [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs')

DYN, MOD = '#2a78d6', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
EDGES = np.arange(0.0, 5.21, 0.4)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
MIN_R = 8                     # runs needed in a bin

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']
axis = d['axis'][rid]
tip = d['tip'][rid]

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
fig.subplots_adjust(left=0.085, right=0.98, bottom=0.14, top=0.93,
                    wspace=0.08)

for k, axn in enumerate(('Mx', 'My')):
    a_ = axes[k]
    sel = axis == axn
    med = np.full(len(CTR), np.nan)
    q1 = np.full(len(CTR), np.nan)
    q3 = np.full(len(CTR), np.nan)
    mm = np.full(len(CTR), np.nan)
    nrun = np.zeros(len(CTR), int)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = sel & (phi >= lo) & (phi < hi)
        mp, mn = m & (tip == 'pos'), m & (tip == 'neg')
        # Average the two tip directions rather than pooling them.  The
        # two carry an antisymmetric term -- 470 mN.m apart on My, 80 on
        # Mx -- and they do not reach the same tilt: My/neg stops at a
        # median 3.1 deg where My/pos goes to 6.8.  Pooling therefore
        # slides towards the positive direction as the tilt grows and
        # manufactures a step: the My pooled median jumps from 142 to
        # 230 mN.m between 2.2 and 2.6 deg while NEITHER direction's own
        # median moves, only their mix, 49% negative becoming 45 and
        # then 14 by 4.2 deg.  The same mixing sets the width of the
        # band.  Both directions must therefore be represented, and the
        # curve stops where the shorter one runs out.
        if (len(np.unique(rid[mp])) < MIN_R
                or len(np.unique(rid[mn])) < MIN_R
                or mp.sum() < 20 or mn.sum() < 20):
            continue
        per = [0.5 * (np.median(inv[mp & (rid == j)])
                      + np.median(inv[mn & (rid == k)]))
               for j in np.unique(rid[mp]) for k in np.unique(rid[mn])]
        med[i] = 0.5 * (np.median(inv[mp]) + np.median(inv[mn]))
        q1[i], q3[i] = np.percentile(per, [25, 75])
        mm[i] = 0.5 * (np.median(mod[mp]) + np.median(mod[mn]))
        nrun[i] = len(np.unique(rid[m]))
    ok = ~np.isnan(med)
    a_.fill_between(CTR[ok], q1[ok], q3[ok], color=DYN, alpha=0.16, lw=0,
                    zorder=2)
    a_.plot(CTR[ok], med[ok], color=DYN, lw=2.8, zorder=5,
            label='dynamic inversion', solid_capstyle='round')
    a_.plot(CTR[ok], mm[ok], color=MOD, lw=2.8, zorder=6,
            label='image-superposition model', solid_capstyle='round')
    a_.set_xlabel(r'$\varphi$  [deg]', color=INK2)
    a_.set_xlim(0, 5.2)
    a_.grid(alpha=0.22, lw=0.6, color=MUTED)
    a_.set_axisbelow(True)
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
    a_.set_title(f'$M_{axn[1]}$', fontsize=13, color=INK, loc='left', pad=6)
    if k == 0:
        a_.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
        a_.legend(fontsize=10.5, frameon=False, loc='lower left',
                  labelcolor=INK2)
    res = (med - mm)[ok]
    print(f"\n{axn}: {int(sel[np.unique(rid, return_index=True)[1]].sum())}"
          f" runs\n")
    print(f"  {'phi':>5}{'runs':>6}{'inversion':>11}{'model':>8}{'diff':>8}"
          f"{'band':>7}")
    for i in np.flatnonzero(ok):
        print(f"  {CTR[i]:5.1f}{nrun[i]:6d}{med[i]:11.1f}{mm[i]:8.1f}"
              f"{med[i] - mm[i]:+8.1f}{q3[i] - q1[i]:7.0f}")
    print(f"  residual  median {np.median(res):+.1f}  RMS "
          f"{np.sqrt(np.mean(res ** 2)):.1f}  max |{np.max(np.abs(res)):.0f}|"
          f"   ({100 * np.sqrt(np.mean(res ** 2)) / np.nanmedian(mm):.0f}%"
          f" of the model)")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_dynamics_axis.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_dynamics_axis.png', bbox_inches='tight', dpi=600)
print(f"\n-> {OUT / 'fig_ge_dynamics_axis.pdf'}")
