#!/usr/bin/env python3
"""fig_ge_single_run: one representative run pair per axis, unaggregated.

Everything else in this set is a median over 70 runs.  This is what a
single measurement looks like: the ground-effect moment from the
dynamic inversion against the parameter-free model, for one positive
and one negative tip run of one case, on each axis, plus their average
-- the combination M_ff = sign * 0.5 * (M_pos + M_neg) the
identification forms.

The cases are chosen on BOTH the level and the gradient of the
direction-averaged residual, each scored against its axis median in
units of that axis's spread, so neither panel is a best case.  Choosing
on the level alone picks case_02 on My, whose gradient is +28.9
mN.m/deg against an axis median of -17.2 -- the steepest of the five,
and misleading about the gradient it was not chosen for.

The tilt range is the aggregate figure's, 5.0 deg on Mx and 3.4 on My.
Past that a single run runs on to its own window end, where it is going
over fast and the balance stops holding -- Mx/pos reaches -1050 mN.m by
6.5 deg -- and the aggregate, being a median over 70 runs, neither
shows those tails nor extends into them.

What is visible here that the aggregate hides is the straddle: the two
tip directions sit either side of the model rather than on it, by
roughly +-100 mN.m on Mx and +-120 on My, and only their average
approaches it.  That is the contact lever's direction asymmetry, which
the pivot-free average removes by construction
(analysis/lever_fit.py).

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
# Chosen on BOTH the onset level and the gradient of the
# direction-averaged residual, each scored against its axis median in
# units of that axis's interquartile spread.  Picking on the level
# alone put case_02 on My, whose gradient is +28.9 mN.m/deg against an
# axis median of -17.2 -- the steepest of the five, and misleading.
SHOW = None

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


def representative(axn):
    """the case closest to its axis median in level AND in gradient"""
    lim = 5.0 if axn == 'Mx' else 3.4
    lv, gr, cs = [], [], []
    for c in sorted(set(d['case'])):
        a = []
        for tp in ('pos', 'neg'):
            m = ((d['case'][rid] == c) & (d['axis'][rid] == axn)
                 & (d['tip'][rid] == tp) & (phi <= lim))
            if m.sum() < 30:
                break
            a.append((np.median(d['resid'][m & (phi < 0.4)]),
                      np.polyfit(phi[m], d['resid'][m], 1)[0]))
        if len(a) == 2:
            cs.append(c)
            lv.append(0.5 * (a[0][0] + a[1][0]))
            gr.append(0.5 * (a[0][1] + a[1][1]))
    lv, gr = np.array(lv), np.array(gr)
    sl = np.subtract(*np.percentile(lv, [75, 25])) or 1.0
    sg = np.subtract(*np.percentile(gr, [75, 25])) or 1.0
    score = (np.abs(lv - np.median(lv)) / abs(sl)
             + np.abs(gr - np.median(gr)) / abs(sg))
    k = int(np.argmin(score))
    print(f"  {axn}: {cs[k]}   level {lv[k]:+.1f} (median "
          f"{np.median(lv):+.1f}),  gradient {gr[k]:+.1f} (median "
          f"{np.median(gr):+.1f})")
    return cs[k]


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


print("representative case per axis, scored on level and gradient\n")
SHOW = {axn: representative(axn) for axn in ('Mx', 'My')}

for k, axn in enumerate(('Mx', 'My')):
    a_ = axes[k]
    case = SHOW[axn]
    ip, inn = pick(case, axn, 'pos'), pick(case, axn, 'neg')
    sp, sn = rid == ip, rid == inn
    lim = 5.0 if axn == 'Mx' else 3.4
    for s, col, lab in ((sp, POS, 'pos tip'), (sn, NEG, 'neg tip')):
        c_ = s & (phi <= lim)
        a_.plot(phi[c_], inv[c_], color=col, lw=1.9, alpha=0.9, zorder=4,
                label=lab)
    # Same tilt range the aggregate figure uses.  A single run carries
    # on to its own window end, where the vehicle is going over fast and
    # the balance stops holding -- Mx/pos reaches -1050 mN.m by 6.5 deg
    # -- and the aggregate is a median over 70 runs, which is insensitive
    # to those tails and is cut at 5.0 deg on Mx and 3.4 on My anyway.
    # Showing them here would display a regime the result excludes.
    hi = min(phi[sp].max(), phi[sn].max(), 5.0 if axn == 'Mx' else 3.4)
    g = np.linspace(0, hi, 120)
    av = 0.5 * (np.interp(g, phi[sp], inv[sp])
                + np.interp(g, phi[sn], inv[sn]))
    a_.plot(g, av, color=AVG, lw=2.8, zorder=6, label='their average')
    a_.plot(g, np.interp(g, phi[sp], mod[sp]), color=MOD, lw=2.8, zorder=5,
            label='image-superposition model')
    a_.set_xlabel(r'$\varphi$  [deg]', color=INK2)
    a_.set_xlim(0, hi * 1.02)
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
