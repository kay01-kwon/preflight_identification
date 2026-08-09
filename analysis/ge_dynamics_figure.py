#!/usr/bin/env python3
"""One figure: the ground-effect moment from the model against the
dynamic inversion, over the onset-to-peak window.

Both curves are plotted against the tilt excursion measured from the
onset, so the two questions separate visually: the intercept is the
LEVEL of the ground-effect moment (which the static check already
gives, since the balance at the onset is static) and the slope is its
ATTITUDE DEPENDENCE.  The levels agree; the slopes do not, because the
model's own slope is a per-cent-level feature of the
W z_CoM sin(phi) term the inversion has to cancel.

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
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncols=1,
          fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_ge_dynamics.pdf', bbox_inches='tight')
print(f"-> {OUT / 'fig_ge_dynamics.pdf'}")
