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

Two figures come out, because two different things are being asked.

fig_mcrit_residual pools the five cases and puts the three
ground-effect treatments side by side, since the comparison is what
selects the model, not the level of any one of them:

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

But the width of those pooled boxes must not be read as run-to-run
scatter, and that is what the second figure is for.  Split by case, the
runs of one class agree closely and the classes do not agree with each
other:

    axis / dir     spread of the five case medians    median within-case IQR
    Mx / pos                                229                          21
    Mx / neg                                270                          66
    My / pos                                130                          29
    My / neg                                160                          33

so between 3 and 10 times more of the pooled width is case-to-case
offset than is run-to-run repeatability.  fig_mcrit_residual_case draws
the interference model per case, where each box is one CoM
configuration and the tightness of the individual boxes is the
measurement's actual repeatability.

That case-to-case term is symmetric in the tip direction, so it is not
a ground-effect error: it survives all three treatments nearly
unchanged (Mx/case_03 M_ff -83.8 with no correction, -95.5 with
interference).  It is the part the pivot-free average
M_ff = sign * 0.5 * (M_pos + M_neg) carries, and it ranges from -96 to
+72 mN.m on Mx and -83 to +38 on My.  The pooled medians of -22 and -13
in the table above are five cases of both signs averaging each other
out and understate it; per case it is 3 mm of arm at the balance's own
sensitivity, and it mixes the ground-truth CoM offset with the
direction asymmetry of the contact arm (analysis/lever_fit.py).  What
does hold for every case is that the ground-effect treatment barely
moves it -- the model choice acts on the antisymmetric part only.

Usage:
  python analysis/mcrit_prediction.py <outdir>       # writes the CSV
  python analysis/mcrit_residual_boxplot.py <outdir>/mcrit_per_run.csv
                                            [--outdir DIR] [--dpi N]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('csv', type=Path, help='mcrit_per_run.csv written by '
                                       'analysis/mcrit_prediction.py')
ap.add_argument('--outdir', type=Path, default=Path('docs'),
                help='directory the figures are written to')
ap.add_argument('--dpi', type=int, default=600,
                help='raster resolution of the PNG outputs')
args = ap.parse_args()
SRC, OUT = args.csv, args.outdir

POS, NEG = '#b4451f', '#2a78d6'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
TREAT = (('resid_none_mNm', 'No ground effect'),
         ('resid_single_mNm', 'Single rotor'),
         ('resid_interf_mNm', 'Rotor interference'))

rows = list(csv.DictReader(open(SRC)))
if not rows:
    sys.exit(f"no rows in {SRC}")
CASES = sorted({r['case'] for r in rows})

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})


def take(axn, dirn, col, case=None):
    return np.array([float(r[col]) for r in rows
                     if r['axis'] == axn and r['dir'] == dirn
                     and (case is None or r['case'] == case)])


def iqr(v):
    return float(np.subtract(*np.percentile(v, [75, 25])))


def box(a_, v, at, col):
    bp = a_.boxplot([v], positions=[at], widths=0.31, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=INK, lw=1.8),
                    whiskerprops=dict(color=col, lw=1.1),
                    capprops=dict(color=col, lw=1.1))
    for b in bp['boxes']:
        b.set(facecolor=col, alpha=0.30, edgecolor=col, lw=1.2)


def dress(a_, ticks, labels, title, first, legend='lower right'):
    a_.axhline(0, color=MUTED, lw=0.9, zorder=1)
    a_.set_xticks(ticks)
    a_.set_xticklabels(labels)
    a_.set_xlim(-0.55, len(ticks) - 0.45)
    a_.grid(axis='y', alpha=0.22, lw=0.6, color=MUTED)
    a_.set_axisbelow(True)
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
    a_.set_title(title, fontsize=13, color=INK, loc='left', pad=6)
    a_.set_ylabel(r'$M_{\mathrm{crit,est}} - M_{\mathrm{crit,th}}$'
                  r' [mN$\cdot$m]', color=INK2)
    if first:
        h = [plt.Line2D([], [], color=c, lw=7, alpha=0.45, label=t)
             for c, t in ((POS, 'positive tip'), (NEG, 'negative tip'))]
        a_.legend(handles=h, fontsize=10.5, frameon=False,
                  loc=legend, labelcolor=INK2)


# ---- pooled over the cases, the three treatments, stacked 2x1 --------
fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.6), sharey=True)
fig.subplots_adjust(left=0.145, right=0.97, bottom=0.075, top=0.955,
                    hspace=0.2)
print("static critical-moment residual, run by run  [mN.m]\n")
print("pooled over the five cases\n")
for k, axn in enumerate(('Mx', 'My')):
    a_ = axes[k]
    print(f"{axn}")
    print(f"  {'treatment':18}{'dir':>6}{'n':>5}{'q1':>9}{'median':>9}"
          f"{'q3':>9}{'|resid| med':>13}")
    for j, (col, lab) in enumerate(TREAT):
        for dirn, colr, dx in (('pos', POS, -0.19), ('neg', NEG, +0.19)):
            box(a_, take(axn, dirn, col), j + dx, colr)
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
    dress(a_, range(len(TREAT)), [t for _, t in TREAT], f'$M_{axn[1]}$',
          k == 0)
    print()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_mcrit_residual.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_mcrit_residual.png', bbox_inches='tight',
            dpi=args.dpi)
print(f"-> {OUT / 'fig_mcrit_residual.pdf'}\n")

# ---- the reported model, case by case --------------------------------
# The pooled boxes above are wide mostly because the five cases sit at
# different levels, not because the runs of one case disagree.  Drawn
# per case the runs are tight and the case offsets are visible as what
# they are: a direction-SYMMETRIC term, present with or without the
# ground-effect correction, which the model choice does not touch.
COL = 'resid_interf_mNm'
fig2, axes2 = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
fig2.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.93,
                     wspace=0.08)
print("the interference model, case by case\n")
for k, axn in enumerate(('Mx', 'My')):
    a_ = axes2[k]
    print(f"{axn}   {'case':10}{'n':>4}{'pos med':>9}{'neg med':>9}"
          f"{'antisym':>9}{'M_ff':>8}{'pos IQR':>9}{'neg IQR':>9}")
    for j, case in enumerate(CASES):
        for dirn, colr, dx in (('pos', POS, -0.19), ('neg', NEG, +0.19)):
            box(a_, take(axn, dirn, COL, case), j + dx, colr)
        vp, vn = take(axn, 'pos', COL, case), take(axn, 'neg', COL, case)
        print(f"     {case:10}{len(vp) + len(vn):4d}{np.median(vp):9.1f}"
              f"{np.median(vn):9.1f}"
              f"{0.5 * (np.median(vp) - np.median(vn)):9.1f}"
              f"{0.5 * (np.median(vp) + np.median(vn)):8.1f}"
              f"{iqr(vp):9.1f}{iqr(vn):9.1f}")
    for dirn in ('pos', 'neg'):
        med = np.array([np.median(take(axn, dirn, COL, c)) for c in CASES])
        wi = np.median([iqr(take(axn, dirn, COL, c)) for c in CASES])
        print(f"     {axn}/{dirn}: spread of the case medians "
              f"{med.max() - med.min():.0f}, median within-case IQR "
              f"{wi:.0f} mN.m")
    dress(a_, range(len(CASES)), [c.replace('case_', '') for c in CASES],
          f'$M_{axn[1]}$', k == 0, legend='upper right')
    a_.set_xlabel('case', color=INK2)
    print()
fig2.savefig(OUT / 'fig_mcrit_residual_case.pdf', bbox_inches='tight')
fig2.savefig(OUT / 'fig_mcrit_residual_case.png', bbox_inches='tight',
             dpi=args.dpi)
print(f"-> {OUT / 'fig_mcrit_residual_case.pdf'}")
