#!/usr/bin/env python3
"""fig_ge_dynamics_axis: the dynamic check, one panel per excitation axis.

The two axes are different experiments and pooling them hides both.
They tip about different contact lines (140 mm against 110), the model
predicts different moments (about 175 mN.m against 138), and
analysis/rate_band_check.py finds the rigid-body balance closing on Mx
(ratio 0.9-1.1 below 3 Hz) and not on My (1.8-2.0).  Reported
separately:

    Mx, 0.2-5.0 deg    residual median -17.2, RMS 16.6, max 32
                       9% of the model, 66-70 runs in every bin
    My, 0.2-2.2 deg    residual median  -6.2, RMS 15.0, max 28
                       11% of the model, 67-70 runs in every bin
    My, 2.6-5.0 deg    residual median +124.3, RMS 126.9, max 154

So My is not a bad axis.  Below 2.4 deg it agrees with the model as
well as Mx does over its whole range.  It then departs abruptly: +4.3
mN.m at 2.2 deg becomes +92.5 at 2.6, with 67 and 66 runs contributing,
so the transition is not the sample thinning -- that comes afterwards,
as the runs stop reaching further.  Above it the inversion reads 90-150
mN.m more than the model, the direction the band ratio predicts, the
gyro implying more moment than the instruments show.

The band is the interquartile range across runs, dispersion rather than
an uncertainty on the median.  It is about three times wider on My than
on Mx at every tilt, which is the same fact the band ratio reports.

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
        r = np.unique(rid[m])
        if len(r) < MIN_R:
            continue
        per = [np.median(inv[m & (rid == j)]) for j in r]
        med[i] = np.median(inv[m])
        q1[i], q3[i] = np.percentile(per, [25, 75])
        mm[i] = np.median(mod[m])
        nrun[i] = len(r)
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
