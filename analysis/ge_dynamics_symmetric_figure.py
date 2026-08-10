#!/usr/bin/env python3
"""fig_ge_dynamics_sym: the moment against tilt, before and after the average.

The same comparison as fig_ge_dynamics -- the ground-effect moment from
the parameter-free image-superposition model against the dynamic
inversion, plotted against the tilt excursion -- but split by what the
identification actually forms.

(a) Per tip direction.  The two directions straddle the model by
    hundreds of mN.m and neither follows it.  Read this way the check
    fails, and that is what earlier versions of the figure showed
    (pooling the two directions hid it behind a median).

(b) After the pivot-free average.  For each case/axis group the pos and
    neg medians are averaged at each tilt, which is the combination
    M_ff = sign * 0.5 * (M_pos + M_neg) the deliverable is built from,
    so the antisymmetric term -- +7.7 mm on My, -1.4 mm on Mx expressed
    as a length -- is removed by construction rather than by choice.
    What is left tracks the model.

The averaged curve stops where the shorter direction stops: My/neg
reaches about 4.8 deg while My/pos reaches 9.2, and an average is only
formed where both exist.  The line is drawn only over that range.

Usage:
  HD_DERIV=polyk:6 HD_GAIN=0.890 HD_DUMP=hd.npz \
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

POS, NEG, DYN, MOD = '#b4451f', '#2a78d6', '#2a78d6', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
EDGES = np.arange(0.0, 6.01, 0.4)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
MIN_N = 12                    # samples needed in a bin for one group/direction
MIN_G = 4                     # groups needed before a bin's median is drawn


def enough(v):
    """mask of bins backed by at least MIN_G groups -- without it the
    tail spikes, since past 4 deg only one or two groups still have
    data and the median is then a single run"""
    return (~np.isnan(v)).sum(axis=0) >= MIN_G

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']
grp = np.array([f"{c}/{a}/{t}"
                for c, a, t in zip(d['case'], d['axis'], d['tip'])])
gs = np.array([f"{c}/{a}" for c, a in zip(d['case'], d['axis'])])
GROUPS = sorted(set(gs))


def binned(mask_sample, y):
    """median of y per tilt bin over the selected samples"""
    out = np.full(len(CTR), np.nan)
    for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = mask_sample & (phi >= a) & (phi < b)
        if m.sum() >= MIN_N:
            out[i] = np.median(y[m])
    return out


# per group and direction, then the average of the two directions
sym_inv, sym_mod = [], []
for g in GROUPS:
    p = binned((gs[rid] == g) & (grp[rid].astype(str) == g + '/pos'), inv)
    n = binned((gs[rid] == g) & (grp[rid].astype(str) == g + '/neg'), inv)
    pm = binned((gs[rid] == g) & (grp[rid].astype(str) == g + '/pos'), mod)
    nm = binned((gs[rid] == g) & (grp[rid].astype(str) == g + '/neg'), mod)
    sym_inv.append(0.5 * (p + n))
    sym_mod.append(0.5 * (pm + nm))
sym_inv = np.array(sym_inv)
sym_mod = np.array(sym_mod)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.80,
                    wspace=0.24)


def dress(ax, title):
    ax.axhline(0, color=MUTED, lw=0.9, zorder=0)
    ax.set_xlabel(r'$\varphi$  [deg]', color=INK2)
    ax.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
    ax.set_xlim(-0.1, 6.05)
    ax.grid(alpha=0.22, lw=0.6, color=MUTED)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=11.5, color=INK, loc='left', pad=8)


# ---- (a) per direction ----------------------------------------------
ax = axes[0]
for tip, col, lab in (('pos', POS, 'pos tip'), ('neg', NEG, 'neg tip')):
    v = np.array([binned(grp[rid].astype(str) == g + '/' + tip, inv)
                  for g in GROUPS])
    med = np.nanmedian(v, axis=0)
    ok = ~np.isnan(med) & enough(v)
    ax.fill_between(CTR[ok], np.nanpercentile(v, 25, axis=0)[ok],
                    np.nanpercentile(v, 75, axis=0)[ok], color=col,
                    alpha=0.15, lw=0, zorder=2)
    ax.plot(CTR[ok], med[ok], color=col, lw=2.6, zorder=5, label=lab,
            solid_capstyle='round')
vm = np.array([binned(gs[rid] == g, mod) for g in GROUPS])
mm = np.nanmedian(vm, axis=0)
ok = ~np.isnan(mm) & enough(vm)
ax.plot(CTR[ok], mm[ok], color=MOD, lw=2.6, zorder=6, label='model',
        solid_capstyle='round')
dress(ax, '(a)  per tip direction')
ax.legend(fontsize=10, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.17), ncol=3, labelcolor=INK2)
YL = ax.get_ylim()

# ---- (b) after the pivot-free average -------------------------------
ax = axes[1]
med = np.nanmedian(sym_inv, axis=0)
ok = ~np.isnan(med) & enough(sym_inv)
ax.fill_between(CTR[ok], np.nanpercentile(sym_inv, 25, axis=0)[ok],
                np.nanpercentile(sym_inv, 75, axis=0)[ok], color=DYN,
                alpha=0.16, lw=0, zorder=2)
ax.plot(CTR[ok], med[ok], color=DYN, lw=2.8, zorder=5,
        label='dynamic inversion, direction-averaged',
        solid_capstyle='round')
mm = np.nanmedian(sym_mod, axis=0)
okm = ~np.isnan(mm) & enough(sym_mod)
ax.plot(CTR[okm], mm[okm], color=MOD, lw=2.8, zorder=6,
        label='image-superposition model', solid_capstyle='round')
dress(ax, '(b)  after the pivot-free average')
ax.set_ylim(YL)
ax.legend(fontsize=10, frameon=False, loc='upper left', ncol=1,
          labelcolor=INK2)
res = med[ok] - mm[ok]
ax.text(0.03, 0.05,
        f'residual over {CTR[ok][0]:.1f}–{CTR[ok][-1]:.1f}°:  '
        f'median {np.median(res):+.0f}, RMS {np.sqrt(np.mean(res**2)):.0f} mNm\n'
        'band: interquartile range across the 10 case/axis groups',
        transform=ax.transAxes, fontsize=9.5, color=INK, linespacing=1.4,
        bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=4, alpha=0.93))

fig.suptitle('the same comparison, before and after the combination the '
             'identification forms — 140 runs',
             fontsize=12, color=INK, x=0.075, ha='left', y=0.965)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_dynamics_sym.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_dynamics_sym.png', bbox_inches='tight', dpi=190)
print(f"-> {OUT / 'fig_ge_dynamics_sym.pdf'}")
print(f"   averaged curve spans {CTR[ok][0]:.1f}-{CTR[ok][-1]:.1f} deg")
print(f"   residual  median {np.median(res):+.1f}  RMS "
      f"{np.sqrt(np.mean(res ** 2)):.1f}  max |{np.max(np.abs(res)):.0f}| mNm")
for c, v in zip(CTR[ok], res):
    print(f"     phi {c:4.1f}   inversion - model {v:+7.1f}")
