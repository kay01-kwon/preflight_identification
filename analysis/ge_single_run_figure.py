#!/usr/bin/env python3
"""fig_ge_single_run: one representative run pair per axis, unaggregated.

Everything else in this set is a median over 70 runs.  This is what a
single measurement looks like: the ground-effect moment from the
dynamic inversion against the parameter-free model, for one positive
and one negative tip run of one case, on each axis, plus their average
-- the combination M_ff = sign * 0.5 * (M_pos + M_neg) the
identification forms.

The cases shown are the ones whose direction-averaged onset residual
sits closest to their axis's median over the five cases, so neither
panel is a best case: case_05 on Mx (-32.5 mN.m against an axis median
of -32.5) and case_02 on My (-56.6 against -56.6).

Two things are visible here that the aggregate hides.  The two tip
directions straddle the model rather than sitting on it, by about +-240
mN.m on My and +-60 on Mx -- the contact lever's direction asymmetry,
which the average removes (analysis/lever_fit.py).  And a single run is
not smooth: the per-run residual has an RMS of 137 mN.m on Mx, most of
which the estimator's own span absorbs, leaving a propagated 0.02 mm on
the identified offset (analysis/residual_to_mm.py).

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/ge_single_run_figure.py hd.npz [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs')

POS, NEG, AVG, MOD = '#b4451f', '#2a78d6', '#0b0b0b', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
SHOW = {'Mx': 'case_05', 'My': 'case_02'}

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
fig.subplots_adjust(left=0.085, right=0.98, bottom=0.14, top=0.90,
                    wspace=0.08)


def pick(case, axn, tp):
    """the run of this class whose onset residual is closest to its median"""
    m = ((d['case'] == case) & (d['axis'] == axn) & (d['tip'] == tp))
    idx = np.flatnonzero(m)
    val = []
    for i in idx:
        s = (rid == i) & (phi >= 0) & (phi < 0.4)
        val.append(np.median(d['resid'][s]) if s.sum() else np.nan)
    val = np.array(val)
    return int(idx[np.nanargmin(np.abs(val - np.nanmedian(val)))])


for k, axn in enumerate(('Mx', 'My')):
    a_ = axes[k]
    case = SHOW[axn]
    ip, inn = pick(case, axn, 'pos'), pick(case, axn, 'neg')
    sp, sn = rid == ip, rid == inn
    for s, col, lab in ((sp, POS, 'pos tip'), (sn, NEG, 'neg tip')):
        a_.plot(phi[s], inv[s], color=col, lw=1.9, alpha=0.9, zorder=4,
                label=lab)
    # the average, on the tilt grid both runs cover
    hi = min(phi[sp].max(), phi[sn].max())
    g = np.linspace(0, hi, 120)
    av = 0.5 * (np.interp(g, phi[sp], inv[sp])
                + np.interp(g, phi[sn], inv[sn]))
    a_.plot(g, av, color=AVG, lw=2.8, zorder=6, label='their average')
    a_.plot(g, np.interp(g, phi[sp], mod[sp]), color=MOD, lw=2.8, zorder=5,
            label='image-superposition model')
    a_.set_xlabel(r'$\varphi$  [deg]', color=INK2)
    a_.set_xlim(0, max(phi[sp].max(), phi[sn].max()) * 1.02)
    a_.grid(alpha=0.22, lw=0.6, color=MUTED)
    a_.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        a_.spines[sp_].set_visible(False)
    a_.set_title(f'$M_{axn[1]}$   {case.replace("_", " ")}', fontsize=12.5,
                 color=INK, loc='left', pad=6)
    if k == 0:
        a_.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
        a_.legend(fontsize=10, frameon=False, loc='lower left',
                  labelcolor=INK2)
    md = np.interp(g, phi[sp], mod[sp])
    print(f"\n{axn}  {case}   {str(d['bag'][ip])} / {str(d['bag'][inn])}")
    print(f"  reach            {phi[sp].max():.1f} / {phi[sn].max():.1f} deg")
    print(f"  onset            pos {inv[sp][0]:7.0f}   neg {inv[sn][0]:7.0f}"
          f"   model {mod[sp][0]:6.0f} mN.m")
    print(f"  average - model  median {np.median(av - md):+.0f}   RMS "
          f"{np.sqrt(np.mean((av - md) ** 2)):.0f} mN.m")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_single_run.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_single_run.png', bbox_inches='tight', dpi=600)
print(f"\n-> {OUT / 'fig_ge_single_run.pdf'}")
