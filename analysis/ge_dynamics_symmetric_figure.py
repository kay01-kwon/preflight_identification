#!/usr/bin/env python3
"""fig_ge_dynamics_sym: the ground-effect moment against tilt.

The dynamic inversion, averaged over the two tip directions, against
the parameter-free image-superposition model.  Averaging the directions
is not a choice made for this figure: it is the combination
M_ff = sign * 0.5 * (M_pos + M_neg) the identification is built from,
and it removes an antisymmetric term worth +7.7 mm on My and -1.4 mm on
Mx as a length (see analysis/ge_dynamic_symmetry.py and fig_ge_symmetry).

Line and band are the same estimator -- median and interquartile range
across the ten case/axis groups -- so the line always lies inside its
band.  Both are dispersion across groups, not an uncertainty on the
median.

Range.  A group can only form an average in a bin where both its
directions are represented, and the neg direction does not tip as far,
so the set of contributing groups shrinks with tilt and shrinks
unevenly: all ten reach 2.6 deg, but by 4.6 only the five Mx groups do.
The curve is drawn wherever four groups remain, the count is printed
per bin, and the residual is quoted both over the balanced range and
over everything drawn.

The derivative is the filtered one, HD_DERIV=bwk:3 -- centred
difference, zero-phase Butterworth at 3 Hz over [0, window end], phi
from the integral of that same filtered rate.  3 Hz is where
analysis/rate_band_check.py puts the boundary between rigid rotation
about the contact line and airframe structure: over 140 runs the
inertial moment the gyro implies matches the moment actually present to
within a factor 1.2 below 3 Hz and misses it by 8.7 in 3-6 Hz.

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/ge_dynamics_symmetric_figure.py hd.npz [outdir]
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
GRP_S, GRP_R = 5, 3          # samples and runs per group, per direction
MIN_G = 4                    # groups needed before a bin is drawn at all

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']
tip = d['tip'][rid]
gs = np.array([f"{c}/{a}" for c, a in zip(d['case'], d['axis'])])[rid]
GROUPS = sorted(set(gs))
BINS = [(phi >= a) & (phi < b) for a, b in zip(EDGES[:-1], EDGES[1:])]


def per_bin(sel, y):
    out = np.full(len(CTR), np.nan)
    for i, b in enumerate(BINS):
        m = sel & b
        if m.sum() >= GRP_S and len(np.unique(rid[m])) >= GRP_R:
            out[i] = np.median(y[m])
    return out


sym_g = np.array([0.5 * (per_bin((gs == g) & (tip == 'pos'), inv)
                         + per_bin((gs == g) & (tip == 'neg'), inv))
                  for g in GROUPS])
mod_g = np.array([0.5 * (per_bin((gs == g) & (tip == 'pos'), mod)
                         + per_bin((gs == g) & (tip == 'neg'), mod))
                  for g in GROUPS])

# A group only forms an average where both its directions reach the
# bin, and neg does not tip as far, so the contributing set shrinks with
# tilt and shrinks unevenly -- by 4.6 deg only the five Mx groups are
# left.  The curve is drawn wherever MIN_G groups remain and the count
# is printed per bin, so what each part is a statistic of stays on the
# record; the range over which all ten contribute is reported
# separately as the balanced one.
ngrp = (~np.isnan(sym_g)).sum(axis=0)
ok = ngrp >= MIN_G
full = ngrp == len(GROUPS)
x = CTR[ok]
med = np.nanmedian(sym_g[:, ok], axis=0)
q1 = np.nanpercentile(sym_g[:, ok], 25, axis=0)
q3 = np.nanpercentile(sym_g[:, ok], 75, axis=0)
mm = np.nanmedian(mod_g[:, ok], axis=0)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.4, 4.6))
fig.subplots_adjust(left=0.145, right=0.975, bottom=0.135, top=0.90)

ax.fill_between(x, q1, q3, color=DYN, alpha=0.16, lw=0, zorder=2)
ax.plot(x, med, color=DYN, lw=2.8, zorder=5, label='dynamic inversion',
        solid_capstyle='round')
ax.plot(x, mm, color=MOD, lw=2.8, zorder=6,
        label='image-superposition model', solid_capstyle='round')
ax.set_xlabel(r'$\varphi$  [deg]', color=INK2)
ax.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
ax.set_xlim(0, EDGES[np.flatnonzero(ok)[-1] + 1])
ax.grid(alpha=0.22, lw=0.6, color=MUTED)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=11, frameon=False, loc='lower left', labelcolor=INK2)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_dynamics_sym.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_dynamics_sym.png', bbox_inches='tight', dpi=600)
print(f"-> {OUT / 'fig_ge_dynamics_sym.pdf'}")

res = med - mm
w = q3 - q1
print(f"\n  drawn over {x[0]:.1f}-{x[-1]:.1f} deg;  all {len(GROUPS)} groups"
      f" to {CTR[full][-1]:.1f} deg\n")
print(f"  {'phi':>5}{'inversion':>11}{'model':>8}{'diff':>8}"
      f"{'q25':>8}{'q75':>8}{'band':>7}{'groups':>8}   composition")
for i, k in enumerate(np.flatnonzero(ok)):
    got = [GROUPS[j].replace('case_0', 'c') for j in range(len(GROUPS))
           if not np.isnan(sym_g[j, k])]
    print(f"  {x[i]:5.1f}{med[i]:11.1f}{mm[i]:8.1f}{res[i]:+8.1f}"
          f"{q1[i]:8.1f}{q3[i]:8.1f}{w[i]:7.0f}{len(got):8d}   "
          f"{'all ten' if len(got) == len(GROUPS) else ','.join(got)}")
rb = (med - mm)[full[ok]]
print(f"\n  residual, all {len(GROUPS)} groups (0.2-{CTR[full][-1]:.1f} deg):"
      f"  median {np.median(rb):+.1f}  RMS {np.sqrt(np.mean(rb ** 2)):.1f}"
      f"  max |{np.max(np.abs(rb)):.0f}|")
print(f"  residual, everything drawn:            "
      f"  median {np.median(res):+.1f}  RMS {np.sqrt(np.mean(res ** 2)):.1f}"
      f"  max |{np.max(np.abs(res)):.0f}|")
print(f"  band width max {w.max():.0f} mNm at phi = {x[np.argmax(w)]:.1f} deg,"
      f"  min {w.min():.0f},  median {np.median(w):.0f}")
