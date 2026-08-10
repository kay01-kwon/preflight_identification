#!/usr/bin/env python3
"""Why the runs must not be admitted on the lever their balance demands.

The 1-3 Hz balance can be solved for the contact lever that would close
it (analysis/lever_solve.py), and a lever outside the landing-gear
footprint -- 140 mm on Mx, 125 on My -- is not a contact line, so it is
tempting to drop those runs.  Doing it is a mistake, and this shows
why.

Split the Mx runs on that test and the two halves land on opposite
sides of the model, both further from it than the whole set:

    all 70 Mx runs                 residual median -17.2, RMS 16.6
    lever within the footprint     -70.9, RMS 73.4   (34 runs)
    lever outside it               +27.6, RMS 26.8   (36 runs)

The criterion is not independent of the outcome.  Both the demanded
lever and the residual are built from the same gyro, load cell and
collective, and the two correlate run by run at +0.35 on Mx and -0.61
on My, so selecting on the lever selects on the residual.  It is the
selection bias the criterion was meant to avoid, and it manufactures a
disagreement in whichever direction the cut is placed.

The axis-level statement survives: the band ratio is a property of a
whole axis, computed over 140 runs, and does not select runs.  The
per-run version does not.

Usage:
  python analysis/admit_effect.py SCRATCH OUTDIR
    (SCRATCH holds hd_bwk3.npz and admitted.txt)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ALL, IN, OUTC, MOD = '#2a78d6', '#1baf7a', '#b4451f', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'

keep = set((SP / 'admitted.txt').read_text().split())
d = np.load(SP / 'hd_bwk3.npz')
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']
bk = np.array([f"{c}/{a}/{b}"
               for c, a, b in zip(d['case'], d['axis'], d['bag'])])
kr = np.array([k in keep for k in bk])
isMx = d['axis'] == 'Mx'
E = np.arange(0.0, 5.21, 0.4)
C = 0.5 * (E[:-1] + E[1:])


def curve(sel_run):
    y = np.full(len(C), np.nan)
    for i, (a, b) in enumerate(zip(E[:-1], E[1:])):
        m = sel_run[rid] & (phi >= a) & (phi < b)
        if len(np.unique(rid[m])) >= 8:
            y[i] = np.median(inv[m])
    return y


mm = np.full(len(C), np.nan)
for i, (a, b) in enumerate(zip(E[:-1], E[1:])):
    m = isMx[rid] & (phi >= a) & (phi < b)
    if m.sum():
        mm[i] = np.median(mod[m])

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.0, 4.8))
fig.subplots_adjust(left=0.135, right=0.975, bottom=0.135, top=0.95)

for sel, col, lab in (
        (isMx, ALL, f'all {int(isMx.sum())} $M_x$ runs'),
        (isMx & kr, IN, f'lever within the footprint ({int((isMx&kr).sum())})'),
        (isMx & ~kr, OUTC,
         f'lever outside it ({int((isMx&~kr).sum())})')):
    y = curve(sel)
    ok = ~np.isnan(y)
    ax.plot(C[ok], y[ok], color=col, lw=2.8, zorder=5, label=lab,
            solid_capstyle='round')
ok = ~np.isnan(mm)
ax.plot(C[ok], mm[ok], color=MOD, lw=2.8, zorder=6,
        label='image-superposition model', solid_capstyle='round')
ax.set_xlabel(r'$\varphi$  [deg]', color=INK2)
ax.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
ax.set_xlim(0, 5.2)
ax.grid(alpha=0.22, lw=0.6, color=MUTED)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=10.5, frameon=False, loc='lower left', labelcolor=INK2)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_admit_effect.png', bbox_inches='tight', dpi=600)
fig.savefig(OUT / 'fig_admit_effect.pdf', bbox_inches='tight')
print(f"-> {OUT / 'fig_admit_effect.png'}")
