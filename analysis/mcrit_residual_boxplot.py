#!/usr/bin/env python3
"""fig_mcrit_residual: the static critical-moment residual, run by run.

analysis/mcrit_prediction.py reports this comparison as twenty group
means, one per case/axis/direction, which is the right unit for the
headline number but says nothing about the scatter behind it.  This is
the same quantity one run at a time:

    residual = M_onset - M_crit(GE)

with M_onset the moment the onset detector returns and M_crit the static
threshold of Eqs. (7)/(14) built from independently measured
ingredients -- weight and CoM offset from ground truth, the collective
thrust at that run's own onset, and that run's own contact arm from the
mocap circle fit.  Nothing here is fitted to the onset moments.

Three ground-effect treatments are shown side by side because the
comparison is what selects the model, not the level of any one of them:

    none        no ground effect at all
    single      per-rotor Cheeseman superposition, rotor-rotor
                interference neglected
    interf      image superposition over rotor-centre distances,
                mutual interference included -- the reported model,
                with no empirical constant of any kind

The two tip directions are drawn separately, and they have to be.  With
no ground-effect term the residual is almost purely antisymmetric --
the positive direction sits low and the negative equally high -- and
the ground-effect correction is antisymmetric too, since it enters the
threshold as -sgn * c_a * f * l.  So what the boxes show being removed
is the antisymmetric part, and pooling the directions would hide the
whole effect:

    axis  treatment   median pos  median neg  |resid| med   antisym   M_ff
    Mx    none            -185.7      +158.2        166.0    -172.0   -13.8
    Mx    single          -139.4      +110.1        118.5    -124.8   -14.7
    Mx    interf           -11.1       -32.7         61.4     +10.8   -21.9
    My    none            -243.0      +191.5        209.1    -217.2   -25.8
    My    single          -203.4      +158.0        170.4    -180.7   -22.7
    My    interf           -82.5       +57.1         65.7     -69.8   -12.7

antisym = 0.5 (median_pos - median_neg), M_ff = 0.5 (median_pos +
median_neg).  The interference model takes the antisymmetric part from
-172 to +11 on Mx and from -217 to -70 on My; the single-rotor
reference, which neglects rotor-rotor interference, removes about a
quarter of it and leaves -125 and -181.

The last column is the combination the identification actually forms,
M_ff = sign * 0.5 * (M_pos + M_neg).  It cancels any
direction-antisymmetric error identically, which is why it stays within
13 to 26 mN.m whatever the ground-effect treatment -- an order below
the per-direction residuals.  The deliverable is therefore insensitive
to this choice; the per-direction thresholds of Eqs. (7)/(14) are not,
and that is what selects the model.

Usage:
  python analysis/mcrit_prediction.py <outdir>       # writes the CSV
  python analysis/mcrit_residual_boxplot.py <outdir>/mcrit_per_run.csv [docs]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs')

POS, NEG = '#b4451f', '#2a78d6'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
TREAT = (('resid_none_mNm', 'no ground effect'),
         ('resid_single_mNm', 'single-rotor'),
         ('resid_interf_mNm', 'interference'))

rows = list(csv.DictReader(open(SRC)))
if not rows:
    sys.exit(f"no rows in {SRC}")


def take(axn, dirn, col):
    return np.array([float(r[col]) for r in rows
                     if r['axis'] == axn and r['dir'] == dirn])


plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
fig.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.93,
                    wspace=0.08)

print("static critical-moment residual, run by run  [mN.m]\n")
for k, axn in enumerate(('Mx', 'My')):
    a_ = axes[k]
    a_.axhline(0, color=MUTED, lw=0.9, zorder=1)
    print(f"{axn}")
    print(f"  {'treatment':18}{'dir':>6}{'n':>5}{'q1':>9}{'median':>9}"
          f"{'q3':>9}{'|resid| med':>13}")
    for j, (col, lab) in enumerate(TREAT):
        for dirn, col_, dx in (('pos', POS, -0.19), ('neg', NEG, +0.19)):
            v = take(axn, dirn, col)
            bp = a_.boxplot([v], positions=[j + dx], widths=0.31,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color=INK, lw=1.8),
                            whiskerprops=dict(color=col_, lw=1.1),
                            capprops=dict(color=col_, lw=1.1),
                            flierprops=dict(marker='.', markersize=4,
                                            markerfacecolor=col_,
                                            markeredgecolor='none',
                                            alpha=0.55))
            for b in bp['boxes']:
                b.set(facecolor=col_, alpha=0.30, edgecolor=col_, lw=1.2)
        vp, vn = take(axn, 'pos', col), take(axn, 'neg', col)
        both = np.concatenate([vp, vn])
        for dirn, v in (('pos', vp), ('neg', vn)):
            print(f"  {lab if dirn == 'pos' else '':18}{dirn:>6}{len(v):5d}"
                  f"{np.percentile(v, 25):9.1f}{np.median(v):9.1f}"
                  f"{np.percentile(v, 75):9.1f}"
                  f"{np.median(np.abs(v)):13.1f}")
        # the antisymmetric half is what the ground-effect term acts on,
        # since it enters the threshold as -sgn c_a f l; M_ff is the
        # combination the identification forms, and cancels it.
        print(f"  {'':18}{'both':>6}{len(both):5d}"
              f"{'antisym':>9}{0.5 * (np.median(vp) - np.median(vn)):9.1f}"
              f"{'M_ff':>6}{0.5 * (np.median(vp) + np.median(vn)):7.1f}"
              f"{np.median(np.abs(both)):13.1f}")
    a_.set_xticks(range(len(TREAT)))
    a_.set_xticklabels([lab for _, lab in TREAT])
    a_.set_xlim(-0.55, len(TREAT) - 0.45)
    a_.grid(axis='y', alpha=0.22, lw=0.6, color=MUTED)
    a_.set_axisbelow(True)
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
    a_.set_title(f'$M_{axn[1]}$', fontsize=13, color=INK, loc='left', pad=6)
    print()

axes[0].set_ylabel(r'$M_{\mathrm{onset}} - M_{\mathrm{crit}}$  [mNm]',
                   color=INK2)
handles = [plt.Line2D([], [], color=c, lw=7, alpha=0.45, label=t)
           for c, t in ((POS, 'positive tip'), (NEG, 'negative tip'))]
axes[0].legend(handles=handles, fontsize=10.5, frameon=False,
               loc='lower right', labelcolor=INK2)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_mcrit_residual.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_mcrit_residual.png', bbox_inches='tight', dpi=600)
print(f"-> {OUT / 'fig_mcrit_residual.pdf'}")
