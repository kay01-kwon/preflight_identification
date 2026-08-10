#!/usr/bin/env python3
"""fig_ge_symmetry: why the per-direction spread does not reach the result.

Three panels, in the order the argument runs.

(a) Every case/axis group as a dumbbell: the dynamic residual read in
    the pos tip direction, the same read in neg, and the midpoint.  The
    endpoints are hundreds of mN.m apart and sit on opposite sides of
    zero; the midpoints sit on it.  That is the whole point -- the
    disagreement is antisymmetric in the tip direction.

(b) The antisymmetric half as a length, case by case.  It is constant
    within each axis (+7.7 +- 2.2 mm on My, -1.4 +- 1.9 mm on Mx), so it
    belongs to the rig's geometry about each contact line, not to the
    vehicle.  Were it the assumed CoM offset the points would follow the
    dashed identity; they do not (slope 0.14).

(c) The symmetric half -- the only part the pivot-free average passes
    through to the identified offset -- against the static check on the
    same groups.

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/mcrit_prediction.py OUTDIR
  python analysis/ge_dynamic_symmetry_figure.py hd.npz OUTDIR/mcrit_prediction.csv [outdir]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])
CSV = Path(sys.argv[2])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('docs')

POS, NEG, SYM, MOD = '#b4451f', '#2a78d6', '#0b0b0b', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
OFF_MM = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}
W, BAND = 31.59, 0.4

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
res, mod = d['resid'], d['model']
grp = np.array([f"{c}/{a}/{t}"
                for c, a, t in zip(d['case'], d['axis'], d['tip'])])
CASES = sorted(set(d['case']))


def onset(g, y):
    m = (grp[rid] == g) & (phi >= 0) & (phi < BAND)
    return np.median(y[m])


keys, rp, rn, anti, off, sym, mods = [], [], [], [], [], [], []
for c in CASES:
    for axn in ('Mx', 'My'):
        k = f'{c}/{axn}'
        p, n = onset(k + '/pos', res), onset(k + '/neg', res)
        keys.append(k)
        rp.append(p)
        rn.append(n)
        anti.append(0.5 * (p - n) * 1e-3 / W * 1e3)
        off.append(OFF_MM[(c, axn)] * OFF_SIGN[axn])
        sym.append(0.5 * (p + n))
        mods.append(0.5 * (onset(k + '/pos', mod) + onset(k + '/neg', mod)))
rp, rn, anti, off, sym, mods = map(np.asarray,
                                   (rp, rn, anti, off, sym, mods))

S = {}
for r in csv.DictReader(open(CSV)):
    S.setdefault(f"{r['case']}/{r['axis']}", []).append(
        abs(float(r['M_pred_interf'])) - abs(float(r['M_ident'])))
stat = np.array([np.mean(S[k]) * 1e3 for k in keys])

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8),
                         gridspec_kw=dict(width_ratios=[1.5, 1, 0.9]))
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.80,
                    wspace=0.28)


def dress(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, color=INK2)
    ax.set_ylabel(ylab, color=INK2)
    ax.grid(alpha=0.22, lw=0.6, color=MUTED)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=11.5, color=INK, loc='left', pad=8)


# ---- (a) the dumbbells ----------------------------------------------
ax = axes[0]
x = np.arange(len(keys))
ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
ax.vlines(x, rn, rp, color=MUTED, lw=1.4, zorder=2)
ax.plot(x, rp, 'o', ms=7, color=POS, mew=0, zorder=4, label='pos tip')
ax.plot(x, rn, 'o', ms=7, color=NEG, mew=0, zorder=4, label='neg tip')
ax.plot(x, sym, '_', ms=17, mew=3.0, color=SYM, zorder=5,
        label='midpoint = what the average keeps')
ax.set_xticks(x)
ax.set_xticklabels([k.replace('case_0', 'c') for k in keys], rotation=45,
                   ha='right', fontsize=9.5)
dress(ax, '', 'residual  inversion $-$ model  [mNm]',
      '(a)  the disagreement flips sign with the tip direction')
ax.legend(fontsize=9.5, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.22), ncol=3, labelcolor=INK2,
          columnspacing=1.4, handletextpad=0.3)

# ---- (b) the antisymmetric half, as a length ------------------------
ax = axes[1]
lim = 1.15 * max(np.abs(anti).max(), np.abs(off).max())
ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.2, ls=(0, (4, 3)),
        zorder=1)
ax.annotate('if it were the assumed\nCoM offset', (lim, lim),
            textcoords='offset points', xytext=(-8, -30), ha='right',
            color=INK2, fontsize=9.5, linespacing=1.35)
ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
for axn, col, dy in (('My', POS, 7), ('Mx', NEG, -17)):
    m = np.array([k.endswith(axn) for k in keys])
    ax.plot(off[m], anti[m], 'o', ms=8, color=col, mew=0, zorder=4,
            label=axn)
    v = anti[m]
    ax.axhline(np.mean(v), color=col, lw=1.8, ls=(0, (5, 3)), zorder=3)
    ax.annotate(f'{np.mean(v):+.1f} $\\pm$ {np.std(v):.1f} mm',
                (-lim, np.mean(v)), textcoords='offset points',
                xytext=(6, dy), color=col, fontsize=10, fontweight='bold')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
dress(ax, 'assumed CoM offset  [mm]',
      'antisymmetric half  [mm]',
      '(b)  it is a rig constant, not the offset')
ax.legend(fontsize=10, frameon=False, loc='lower right', labelcolor=INK2)

# ---- (c) what survives the average ----------------------------------
ax = axes[2]
ax.axhline(0, color=MOD, lw=2.2, zorder=4)
ax.annotate('model', (1.5, 0), textcoords='offset points', xytext=(-2, 7),
            ha='right', color=MOD, fontsize=10.5, fontweight='bold')
for k, (v, col, lab) in enumerate(((sym, SYM, 'dynamic\n(symmetric half)'),
                                   (stat, INK2, 'static\n(direction-avg)'))):
    ax.plot(np.full(len(v), k) + np.linspace(-.17, .17, len(v)), v, 'o',
            ms=6, color=col, alpha=0.75, mew=0, zorder=5)
    ax.plot([k - .3, k + .3], [np.median(v)] * 2, color=col, lw=2.8,
            zorder=6)
    # headroom is added below, so the summary sits clear of the points
    ax.annotate(f'median {np.median(v):+.0f}\nRMS {np.sqrt(np.mean(v**2)):.0f}',
                (k, 0.90), xycoords=('data', 'axes fraction'),
                ha='center', va='top', color=col, fontsize=10,
                fontweight='bold', linespacing=1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(['dynamic\n(symmetric half)', 'static\n(direction-avg)'],
                   fontsize=9.5)
ax.set_xlim(-0.5, 1.5)
lo, hi = min(sym.min(), stat.min()), max(sym.max(), stat.max())
ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.55 * (hi - lo))
dress(ax, '', 'residual  [mNm]', '(c)  what reaches the result')

fig.suptitle('the per-direction spread is antisymmetric, and the '
             'pivot-free average removes it — 140 runs, 10 case/axis '
             'groups', fontsize=12, color=INK, x=0.055, ha='left', y=0.965)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_symmetry.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_symmetry.png', bbox_inches='tight', dpi=600)
print(f"-> {OUT / 'fig_ge_symmetry.pdf'}")
print(f"   antisymmetric  My {np.mean(anti[1::2]):+.2f} +- "
      f"{np.std(anti[1::2]):.2f} mm,  Mx {np.mean(anti[0::2]):+.2f} +- "
      f"{np.std(anti[0::2]):.2f} mm")
print(f"   symmetric half median {np.median(sym):+.1f}, "
      f"RMS {np.sqrt(np.mean(sym ** 2)):.1f}")
print(f"   static          median {np.median(stat):+.1f}, "
      f"RMS {np.sqrt(np.mean(stat ** 2)):.1f}")
