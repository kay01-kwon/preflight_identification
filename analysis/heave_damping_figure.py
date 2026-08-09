#!/usr/bin/env python3
"""Figure for the heave-damping explanation of the inversion residual.

Three panels, sharing the story in the order a reader needs it:

  (a) the residual falls with attitude at -41.5 mN.m/deg -- an order of
      magnitude steeper than the ground-effect model's own -2.7 to -0.1;
  (b) subtracting the a-priori heave-damping term (momentum theory, no
      fitted constant) flattens it to -0.2;
  (c) and what that agreement is worth: the ratio of the damping the
      data asks for, run by run, to the damping momentum theory
      predicts.  Its median is 0.96, but its interquartile range spans
      a factor of three.

Panels (a) and (b) share a y-axis so the flattening is read off
directly.  Each run's own mean is removed first, because the quoted
slope is a WITHIN-run statistic: pooling runs of different offset would
show a trend that is not the one being measured.  Binned medians
rather than per-run regression lines -- R^2 is 0.21, so the trend is an
ensemble property and the figure should not suggest otherwise.

The figure is deliberately built so the weak part shows.  Subtracting
the a-priori term moves the median slope from -41.5 to -0.2 mN.m/deg
but does NOT narrow the run-to-run scatter (IQR 47.4 -> 49.4), and the
predicted damping is nearly a constant across runs (10% spread) while
the required damping varies by 90%, the two being negatively correlated
(-0.44).  So the agreement is between medians, not run by run: the
mechanism accounts for the level of the residual, not for its
structure.

Usage: python analysis/heave_damping_figure.py hd.npz out.pdf
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'hd.npz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'heave_damping.pdf'

# categorical slots 1-3 of the reference palette (validated all-pairs,
# light mode); aqua sits below 3:1 on the surface, so every band is
# direct-labelled instead of relying on a legend swatch
BAND = [(0.0, 0.2, 'slow', '#2a78d6'),
        (0.2, 0.6, 'mid', '#eb6834'),
        (0.6, 9.9, 'fast', '#1baf7a')]
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b2'
SURF = '#fcfcfb'

d = np.load(SRC)
rid, phi, resid, reg = d['rid'], d['phi'], d['resid'], d['reg']
mdot, d_fit, d_ideal = d['mdot'], d['d_fit'], d['d_ideal']
band_of_run = np.full(len(mdot), -1)
for b, (lo, hi, _, _) in enumerate(BAND):
    band_of_run[(mdot >= lo) & (mdot < hi)] = b
band = band_of_run[rid]

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42,
})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
fig.subplots_adjust(left=0.093, right=0.985, bottom=0.19, top=0.855,
                    wspace=0.33)


def binned(x, y, edges):
    """Median of y in each x bin, with the interquartile band."""
    c, m, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b)
        if s.sum() < 25:
            continue
        c.append(0.5 * (a + b))
        m.append(np.median(y[s]))
        lo.append(np.percentile(y[s], 25))
        hi.append(np.percentile(y[s], 75))
    return map(np.asarray, (c, m, lo, hi))


EDGES = np.linspace(0.0, 6.0, 13)
YLIM = (-260, 260)


def centre(y):
    """Remove each run's own mean: the quoted slope is within-run."""
    out = y.astype(float).copy()
    for i in range(rid.max() + 1):
        s_ = rid == i
        if s_.any():
            out[s_] -= out[s_].mean()
    return out

for k, (ax, y, ttl, slope) in enumerate([
        (axes[0], centre(resid), 'residual, as inverted', -41.5),
        (axes[1], centre(resid - reg),
         'minus a-priori heave damping', -0.2)]):
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.scatter(phi, y, s=1.4, c=MUTED, alpha=0.30, lw=0, zorder=2,
               rasterized=True)
    for b, (lo_, hi_, lab, col) in enumerate(BAND):
        s = band == b
        cx, cm, q1, q3 = binned(phi[s], y[s], EDGES)
        if not len(cx):
            continue
        ax.fill_between(cx, q1, q3, color=col, alpha=0.13, lw=0, zorder=3)
        ax.plot(cx, cm, color=col, lw=2.0, zorder=4,
                solid_capstyle='round')
        # stagger the anchor along the curve so the three labels
        # cannot collide where the bands converge
        p_ = max(0, len(cx) - 1 - 2 * b)
        ax.plot(cx[p_], cm[p_], 'o', ms=4.5, color=col, zorder=5,
                mec=SURF, mew=1.2)
        ax.annotate(lab, (cx[p_], cm[p_]), textcoords='offset points',
                    xytext=(1, -13 if b == 1 else 9), color=col,
                    fontsize=7.5, fontweight='bold', ha='center')
    ax.set_xlim(-0.15, 6.3)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r'tip angle $\delta\varphi$  [deg]', color=INK2)
    ax.set_title(f'({"ab"[k]})  {ttl}', color=INK, loc='left', pad=6)
    iqr = '−63.6 … −16.2' if k == 0 else '−21.9 … +27.5'
    ax.text(0.035, 0.045,
            f'within-run slope  {slope:+.1f}\nIQR  {iqr}   mN·m/deg',
            transform=ax.transAxes, fontsize=7, color=INK, linespacing=1.35,
            bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=2.6, alpha=0.92))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel('inversion − GE model,\nrun mean removed  [mN·m]',
                   color=INK2)
axes[1].tick_params(labelleft=False)

# ---- (c) how far the ratio scatters --------------------------------
ax = axes[2]
eta = np.abs(d_fit) / np.abs(d_ideal)
bins = np.linspace(0, 3, 25)
ax.hist(eta, bins=bins, color=MUTED, alpha=0.55, lw=0)
q1, med, q3 = np.percentile(eta, [25, 50, 75])
ax.axvspan(q1, q3, color='#2a78d6', alpha=0.13, lw=0, zorder=1)
ax.axvline(med, color='#2a78d6', lw=2.0, zorder=4)
ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=3)
ax.annotate('momentum\ntheory', (1.0, ax.get_ylim()[1] * 0.97),
            textcoords='offset points', xytext=(5, -2), color=INK,
            fontsize=7, va='top', linespacing=1.3)
ax.annotate(f'median {med:.2f}', (med, ax.get_ylim()[1] * 0.55),
            textcoords='offset points', xytext=(-6, 0), color='#2a78d6',
            fontsize=7.5, fontweight='bold', ha='right', va='center')
ax.set_xlim(0, 3)
ax.set_xlabel(r'$\eta$ = required |D| / predicted |D|', color=INK2)
ax.set_ylabel('runs', color=INK2)
ax.set_title('(c)  medians agree, runs do not', color=INK,
             loc='left', pad=6)
ax.text(0.97, 0.045,
        f'IQR {q1:.2f}–{q3:.2f}\ncorr(pred, req) = '
        f'{np.corrcoef(np.abs(d_ideal), np.abs(d_fit))[0, 1]:+.2f}',
        transform=ax.transAxes, fontsize=7, color=INK, ha='right',
        linespacing=1.35,
        bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=2.6, alpha=0.92))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

fig.savefig(OUT, dpi=300)
fig.savefig(OUT.rsplit('.', 1)[0] + '.png', dpi=200)
print(f"wrote {OUT}")
print(f"  slope before {np.median(d['resid']):+.1f} / bands "
      + ', '.join(f"{lab} n={int((band_of_run == b).sum())}"
                  for b, (_, _, lab, _) in enumerate(BAND)))
