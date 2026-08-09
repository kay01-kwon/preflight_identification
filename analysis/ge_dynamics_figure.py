#!/usr/bin/env python3
"""SUPERSEDED.  The UNCORRECTED dynamic inversion, kept as the record.

This draws the ground-effect moment from the model against the dynamic
inversion as the raw balance gives it, with none of the three measures
that were later found to be necessary:

  (i)   the Savitzky-Golay derivative is taken on the SLICED window, so
        the onset sits on the filter's extrapolated edge -- worth 1.23
        N.m of fabricated offset on one measured run;
  (ii)  phi comes from the attitude while omega_dot comes from a
        smoothed rate, so the two are not one consistent motion;
  (iii) the reported rate is used as reported, though it carries a
        scale of 0.890 and a bias, both measurable
        (omega_true = (omega_meas - b) / g).

The result is a model that misses badly: gradient -43.5 mN.m/deg
against the model's -2.0, and level 0.77 of it.  That is what this
script is FOR -- it is the "before" panel of that story -- but it is
not the manuscript figure.

FOR THE MANUSCRIPT FIGURE USE INSTEAD:

    HD_DERIV=polyk:6 HD_GAIN=0.890 HD_DUMP=hd.npz \
        python analysis/heave_damping.py
    python analysis/ge_dynamics_rate_figure.py hd.npz docs 9.9

which gives gradient -9.1 and level 1.04 of the model.  See
analysis/rate_derivative.py for (i) and (ii) and the heave_damping
docstring for (iii).

This script writes fig_ge_dynamics_UNCORRECTED.pdf.  It used to write
fig_ge_dynamics.pdf, the same name the corrected script uses, so
running it silently replaced the manuscript figure with this one.

Usage: PYTHONPATH=<stubs> python analysis/ge_dynamics_figure.py [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import ge_dynamics_check as gd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('docs')
Z = 0.261

print('=' * 70)
print('NOTE: this is the SUPERSEDED, UNCORRECTED inversion.')
print('      For the manuscript figure run instead:')
print('        HD_DERIV=polyk:6 HD_GAIN=0.890 HD_DUMP=hd.npz \\')
print('            python analysis/heave_damping.py')
print('        python analysis/ge_dynamics_rate_figure.py hd.npz docs 9.9')
print('      Output here goes to fig_ge_dynamics_UNCORRECTED.pdf')
print('=' * 70)

# The cache holds traces computed with the settings below.  It is
# reused blindly if present, so delete it after changing anything --
# z_CoM, the Savitzky-Golay width, the J_P mode -- or the figure will
# silently keep showing the old ones.
CACHE = Path(__file__).resolve().parent / '.ge_dynamics_traces.npz'
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    ph_all, gd_all, gm_all = (list(z['ph']), list(z['gd']), list(z['gm']))
    print(f"{len(ph_all)} runs from cache ({CACHE.name})")
else:
    gd.analyse.keep_traces = True
    rows = gd.batch([Z], savgol=9, jp_mode='parallel')[Z]
    ph_all = [r['trace'][0] for r in rows if 'trace' in r]
    gd_all = [r['trace'][1] for r in rows if 'trace' in r]
    gm_all = [r['trace'][2] for r in rows if 'trace' in r]
    np.savez(CACHE, ph=np.array(ph_all, dtype=object),
             gd=np.array(gd_all, dtype=object),
             gm=np.array(gm_all, dtype=object))
    print(f"{len(ph_all)} runs with traces -> cached")

fig, ax = plt.subplots(figsize=(6.6, 4.2))
DYN, MOD = '#0072B2', '#D55E00'
P = np.concatenate(ph_all)

# Median with an interquartile band: the run-to-run scatter of the
# inversion is large (it differentiates the gyro), so plotting every
# trace hides the trend that is the point of the figure.
edges = np.arange(0, 7.01, 0.4)
ctr = 0.5 * (edges[:-1] + edges[1:])
for series, col, lab in (
        (np.concatenate(gd_all), DYN,
         r'dynamic inversion:  $J_P\dot\omega - m - f\,l_p'
         r' + Wa\cos\varphi - Wz_{\mathrm{CoM}}\sin\varphi$'),
        (np.concatenate(gm_all), MOD,
         'image-superposition model (parameter-free)')):
    med, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = series[(P >= a) & (P < b)]
        if len(v) > 20:
            med.append(np.median(v)); lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    ax.fill_between(ctr, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
    ax.plot(ctr, med, color=col, lw=2.6, zorder=5, label=lab,
            solid_capstyle='round')

ax.axhline(0, color='0.75', lw=0.8, zorder=0)
ax.annotate('at the onset the balance is static:' '\n' 'the two levels agree',
            xy=(0.35, 205), xytext=(1.05, 400), fontsize=8.5, color='0.2',
            arrowprops=dict(arrowstyle='->', color='0.45', lw=0.9))
ax.annotate('the inversion drifts by the residual of the' '\n'
            r'$Wz_{\mathrm{CoM}}\sin\varphi$ cancellation '
            r'($-141$ mN$\cdot$m/deg, closed to $70\%$),' '\n'
            r'which swamps the model slope of $-2.7$ to $-0.1$',
            xy=(6.2, -35), xytext=(0.30, -350), fontsize=8.5, color='0.2',
            arrowprops=dict(arrowstyle='->', color='0.45', lw=0.9))
ax.set_xlabel(r'tilt excursion from the onset, $\delta\varphi$  [deg]')
ax.set_ylabel(r'ground-effect moment  $\Delta M_{\mathrm{GE}}$  [mN$\cdot$m]')
ax.set_xlim(-0.1, 7.05)
ax.set_ylim(-430, 500)
ax.grid(alpha=0.22, lw=0.6)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=1,
          fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_ge_dynamics_UNCORRECTED.pdf',
            bbox_inches='tight')
print(f"-> {OUT / 'fig_ge_dynamics_UNCORRECTED.pdf'}")
