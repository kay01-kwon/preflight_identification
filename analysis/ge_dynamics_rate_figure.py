#!/usr/bin/env python3
"""fig_ge_dynamics, restricted to the slow end of the ramp-rate range.

Same figure as analysis/ge_dynamics_figure.py -- the ground-effect
moment from the parameter-free image-superposition model against the
dynamic inversion, both plotted against the tilt excursion measured
from the onset -- but built from the current pipeline dump
(z_CoM = 0.261 m, PNLS constants, mocap-measured pivot arm) and with a
ramp-rate ceiling applied.

Why a rate ceiling is worth looking at: the inversion differentiates
the rate gyro, and the faster the ramp the shorter the window over
which that derivative is taken, so the slow runs are where the
inversion is least noisy.  If the model and the inversion are going to
agree anywhere, it is here.

NO heave-damping correction is applied.  This is the inversion as it
comes out of the balance.

Two panels, because one would mislead.  Panel (a) is the levels: the
pooled binned median of each series against the excursion.  At the
onset the balance is static, so the agreement there is the LEVEL of the
ground-effect moment -- the same quantity the static check already
reports, not independent evidence.

The band in panel (a) is the interquartile range across runs at each
tilt bin.  It is the run-to-run DISPERSION and must not be read as a
confidence interval on the median: with 116 runs the standard error of
the median is smaller by roughly sqrt(n), so two medians can be
distinguished far more finely than the overlap of the bands suggests.

Panel (b) is the attitude dependence, and it needs its own panel
because the pooled trend in (a) is NOT the within-run trend.  Runs
differ in offset and in how far they tip, so pooling them flattens the
line, so the panel fits one straight line PER RUN and histograms the
slopes.  Its three summary numbers are different in kind and must not
be confused:

  IQR                the 25th-75th percentile of the per-run slopes.
                     The run-to-run DISPERSION.  Half the runs lie
                     inside it; it does not shrink with n.
  SE of the median   the uncertainty on where the centre is, smaller
                     than the IQR by roughly sqrt(n).
  model - inversion  the gap to be closed, to be read against the SE.

With the corrected pipeline (HD_DERIV=polyk:6 HD_GAIN=0.890) the gap
is of order one SE, so the two are statistically consistent AT THAT
POLYNOMIAL ORDER -- but the order systematic across K = 5, 6, 7 is
about +-10 mN.m/deg, larger than both the SE and the model slope.  The
honest reading is therefore that this apparatus resolves the LEVEL of
the ground-effect moment (1.01-1.04 of the model) and does NOT resolve
its attitude gradient.

Usage:
  python analysis/ge_dynamics_rate_figure.py hd.npz [outdir] [max_rate]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('hd.npz')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs')
MAX_RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

# categorical slots 1-2 of the reference palette (validated)
DYN, MOD = '#2a78d6', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'

d = np.load(SRC)
rid, phi, resid, model, mdot = (d['rid'], d['phi'], d['resid'],
                                d['model'], d['mdot'])
keep = mdot[rid] <= MAX_RATE
n_run = len(np.unique(rid[keep]))
phi, inv, mod = phi[keep], (resid + model)[keep], model[keep]
print(f"{n_run} of {len(mdot)} runs at Mdot <= {MAX_RATE} N.m/s"
      f"  ({keep.sum()} samples)")

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42,
})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9),
                         gridspec_kw=dict(width_ratios=[1.55, 1]))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.155, top=0.80,
                    wspace=0.30)
ax = axes[0]

EDGES = np.arange(0.0, 7.01, 0.4)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])


def band(series):
    med, lo, hi = [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        v = series[(phi >= a) & (phi < b)]
        if len(v) > 20:
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        else:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
    return map(np.asarray, (med, lo, hi))


ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
onset = {}
for series, col, lab, short in (
        (inv, DYN, 'dynamic inversion', 'inversion'),
        (mod, MOD, 'image-superposition model', 'model')):
    med, lo, hi = band(series)
    ok = ~np.isnan(med)
    ax.fill_between(CTR[ok], lo[ok], hi[ok], color=col, alpha=0.16, lw=0,
                    zorder=2)
    ax.plot(CTR[ok], med[ok], color=col, lw=2.6, zorder=5, label=lab,
            solid_capstyle='round')
    i_end = np.flatnonzero(ok)[-1]
    ax.plot(CTR[i_end], med[i_end], 'o', ms=6, color=col, zorder=6,
            mec=SURF, mew=1.4)
    ax.annotate(short, (CTR[i_end], med[i_end]),
                textcoords='offset points', xytext=(8, 0), color=col,
                fontsize=8.5, fontweight='bold', va='center')
    onset[short] = med[np.flatnonzero(ok)[0]]
    print(f"  {short:10} onset level {onset[short]:7.1f} mN.m")

ax.annotate('at the onset the balance is static:\n'
            f'{onset["inversion"]:.0f} vs {onset["model"]:.0f} mN·m — but this is\n'
            'the quantity the static check already gives',
            xy=(0.30, 0.5 * (onset['inversion'] + onset['model'])),
            xytext=(1.05, 395), fontsize=8, color=INK2, linespacing=1.35,
            arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.9))
ax.text(0.03, 0.045,
        'lines: median over runs, binned in tilt\n'
        'band: interquartile range ACROSS RUNS —\n'
        'dispersion, not an uncertainty on the median',
        transform=ax.transAxes, fontsize=7.5, color=INK, linespacing=1.35,
        bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=3, alpha=0.93))
ax.set_xlabel(r'tilt excursion from the onset, $\delta\varphi$  [deg]',
              color=INK2)
ax.set_ylabel(r'ground-effect moment  $\Delta M_{\mathrm{GE}}$  [mN$\cdot$m]',
              color=INK2)
ax.set_xlim(-0.1, 7.05)
ax.set_ylim(-430, 500)
ax.grid(alpha=0.22, lw=0.6, color=MUTED)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title('(a)  levels', fontsize=9, color=INK, loc='left', pad=6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2,
          fontsize=8, frameon=False, labelcolor=INK2)

# ---- (b) within-run attitude slope ----------------------------------
ax = axes[1]
si, sm = [], []
flat = []
for i in np.unique(rid[keep]):
    s_ = rid == i
    # A slope needs an abscissa to fit over.  A run that tips less than
    # 0.2 deg is a single point in phi and polyfit's slope runs away, so
    # panel (b) always has fewer runs than panel (a).  Say by how many.
    if np.ptp(d['phi'][s_]) < 0.2:
        flat.append(i)
        continue
    si.append(np.polyfit(d['phi'][s_], (d['resid'] + d['model'])[s_], 1)[0])
    sm.append(np.polyfit(d['phi'][s_], d['model'][s_], 1)[0])
si, sm = np.array(si), np.array(sm)
if flat:
    print(f"  panel (b) drops {len(flat)} of {n_run} runs that tip less"
          f" than 0.2 deg (no abscissa to fit a slope over)")
bins = np.arange(-140, 41, 10)
ax.hist(si, bins=bins, color=DYN, alpha=0.55, lw=0, label='inversion')
ax.axvline(np.median(si), color=DYN, lw=2.2, zorder=5)
ax.axvline(np.median(sm), color=MOD, lw=2.2, zorder=5)
ax.annotate(f'inversion\n{np.median(si):.1f}', (np.median(si), 0),
            xycoords=('data', 'axes fraction'), textcoords='offset points',
            xytext=(-5, 118), color=DYN, fontsize=8, fontweight='bold',
            ha='right', linespacing=1.3)
ax.annotate(f'model\n{np.median(sm):.1f}', (np.median(sm), 0),
            xycoords=('data', 'axes fraction'), textcoords='offset points',
            xytext=(6, 118), color=MOD, fontsize=8, fontweight='bold',
            ha='left', linespacing=1.3)
ax.set_xlim(-140, 40)
ax.set_xlabel(r'within-run slope  d$\Delta M_{\mathrm{GE}}$/d$\varphi$'
              '  [mN$\cdot$m/deg]', color=INK2)
ax.set_ylabel('runs', color=INK2)
ax.grid(alpha=0.22, lw=0.6, color=MUTED, axis='y')
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title('(b)  attitude dependence', fontsize=9, color=INK, loc='left',
             pad=6)
# sigma/sqrt(n) is the standard error of the MEAN.  The bars here mark
# medians, and for a normal sample the median's standard error is larger
# by sqrt(pi/2) = 1.253, so quoting sigma/sqrt(n) against a median gap
# understates it by a quarter.  Carry the factor.
se = 1.2533 * np.std(si, ddof=1) / np.sqrt(len(si))
ax.text(0.03, 0.78,
        f'IQR {np.percentile(si, 25):.0f} … {np.percentile(si, 75):.0f}\n'
        f'SE of the median  {se:.1f}\n'
        f'model − inversion  {np.median(sm) - np.median(si):+.1f}',
        transform=ax.transAxes, fontsize=7.5, color=INK, linespacing=1.35,
        bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=3, alpha=0.93))
gap = np.median(sm) - np.median(si)
print(f"  within-run slope: inversion {np.median(si):+.1f}, "
      f"model {np.median(sm):+.1f}  ({len(si)} runs)")
print(f"  IQR {np.percentile(si, 25):+.1f} .. {np.percentile(si, 75):+.1f}"
      f"  (dispersion across runs, does not shrink with n)")
print(f"  SE of the median {se:.1f};  model - inversion {gap:+.1f}"
      f"  = {abs(gap)/se:.1f} SE")
print("  NB the polynomial-order systematic across K = 5, 6, 7 is about"
      " +-10,\n     which exceeds both this SE and the model slope itself:"
      " the\n     gradient is NOT resolved by this apparatus.  The level is.")

cap_txt = (rf'all {n_run} runs, $\dot M$ = {mdot.min():.2f}–{mdot.max():.2f}'
           rf' N$\cdot$m/s'
           if MAX_RATE >= mdot.max() else
           rf'ramp rates $\dot M \leq {MAX_RATE:g}$ N$\cdot$m/s '
           rf'({n_run} runs)')
fig.suptitle(cap_txt + ' — no heave-damping correction',
             fontsize=9.5, color=INK, x=0.085, ha='left', y=0.975)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_dynamics.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_dynamics.png', bbox_inches='tight', dpi=200)
print(f"-> {OUT / 'fig_ge_dynamics.pdf'}")
