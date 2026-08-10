#!/usr/bin/env python3
"""fig_ge_dynamics_sym: the moment against tilt, before and after the average.

The same comparison as fig_ge_dynamics -- the ground-effect moment from
the parameter-free image-superposition model against the dynamic
inversion, plotted against the tilt excursion -- but split by what the
identification actually forms.

(a) Per tip direction.  The two directions straddle the model by
    100-150 mN.m from the onset onwards and neither follows it.  Read
    this way the check fails, and that is what earlier versions of the
    figure showed: pooling the two directions hid the split behind a
    single median.

(b) After the pivot-free average.  The two direction medians are
    averaged at each tilt, which is the combination
    M_ff = sign * 0.5 * (M_pos + M_neg) the deliverable is built from,
    so the antisymmetric term -- +7.7 mm on My and -1.4 mm on Mx
    expressed as a length -- is removed by construction rather than by
    choice.  The level then lands on the model at the onset and drifts
    below it as the tilt grows.

Coverage.  Medians are pooled over runs, not formed per group first:
requiring twelve samples per case/axis group per direction stops every
curve at 3.4 deg, since one group holds only seven runs per direction.
Pooling carries both directions to 5.2 deg with 25+ samples and 36+
runs in every bin, which is where the figure ends -- past it one
direction drops below half its runs and the average stops being over
matched conditions.  The band is the interquartile range across runs in
panel (a) and across the ten case/axis groups in panel (b), in both
cases dispersion rather than an uncertainty on the median.

Usage:
  HD_DERIV=polyk:6 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/ge_dynamics_symmetric_figure.py hd.npz [outdir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs')

POS, NEG, DYN, MOD = '#b4451f', '#2a78d6', '#2a78d6', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
# Stop at 5.2 deg.  Beyond it fewer than half the runs are still
# tipping in one direction or the other, so the two direction medians
# come from different subsets -- only the runs that tipped far -- and
# their average is no longer over matched conditions.  The curve does
# climb to +122 above the model out there, but that is a change in
# which runs are being averaged, not a measurement.  MAX_PHI is checked
# against the run counts below, so widening it re-arms the shading.
MAX_PHI = 5.2
EDGES = np.arange(0.0, MAX_PHI + 0.01, 0.4)
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
MIN_S, MIN_R, MIN_G = 25, 10, 4      # samples, runs, groups per bin
# a single case/axis group holds only seven runs per direction, so the
# pooled thresholds above can never be met within one group; the band
# needs its own, looser pair
GRP_S, GRP_R = 8, 4

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
inv, mod = d['resid'] + d['model'], d['model']
tip = d['tip'][rid]
gs_run = np.array([f"{c}/{a}" for c, a in zip(d['case'], d['axis'])])
gs = gs_run[rid]
GROUPS = sorted(set(gs_run))
BINS = [(phi >= a) & (phi < b) for a, b in zip(EDGES[:-1], EDGES[1:])]


def pooled(sel, y, min_s=MIN_S, min_r=MIN_R):
    """median of y per bin, pooled over runs; NaN where too thin"""
    out = np.full(len(CTR), np.nan)
    for i, b in enumerate(BINS):
        m = sel & b
        if m.sum() >= min_s and len(np.unique(rid[m])) >= min_r:
            out[i] = np.median(y[m])
    return out


def runs_in(sel):
    """how many distinct runs back each bin"""
    return np.array([len(np.unique(rid[sel & b])) for b in BINS])


def run_iqr(sel, y):
    """interquartile range across per-run medians, per bin"""
    lo, hi = np.full(len(CTR), np.nan), np.full(len(CTR), np.nan)
    for i, b in enumerate(BINS):
        m = sel & b
        r = np.unique(rid[m])
        if len(r) < MIN_R:
            continue
        v = [np.median(y[m & (rid == k)]) for k in r]
        lo[i], hi[i] = np.percentile(v, [25, 75])
    return lo, hi


# per case/axis group, the symmetric combination, for panel (b)'s band
sym_g = np.full((len(GROUPS), len(CTR)), np.nan)
for j, g in enumerate(GROUPS):
    p = pooled((gs == g) & (tip == 'pos'), inv, GRP_S, GRP_R)
    n = pooled((gs == g) & (tip == 'neg'), inv, GRP_S, GRP_R)
    sym_g[j] = 0.5 * (p + n)

# Past the tilt where a direction retains fewer than half its runs, the
# two medians come from different subsets -- only the runs that tipped
# far -- so their average is no longer over matched conditions.  Shade
# it rather than trimming it away.
n_pos, n_neg = runs_in(tip == 'pos'), runs_in(tip == 'neg')
half = min(n_pos[0], n_neg[0]) / 2
thin = np.flatnonzero((n_pos < half) | (n_neg < half))
THIN = CTR[thin[0]] - 0.2 if len(thin) else None

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.80,
                    wspace=0.24)


def dress(ax, title):
    ax.axhline(0, color=MUTED, lw=0.9, zorder=0)
    ax.set_xlabel(r'$\varphi$  [deg]', color=INK2)
    ax.set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mNm]', color=INK2)
    ax.set_xlim(-0.1, EDGES[-1])
    ax.grid(alpha=0.22, lw=0.6, color=MUTED)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=11.5, color=INK, loc='left', pad=8)


# ---- (a) per direction ----------------------------------------------
ax = axes[0]
med_dir = {}
for t, col, lab in (('pos', POS, 'pos tip'), ('neg', NEG, 'neg tip')):
    sel = tip == t
    m_ = pooled(sel, inv)
    med_dir[t] = m_
    lo, hi = run_iqr(sel, inv)
    ok = ~np.isnan(m_)
    ax.fill_between(CTR[ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0,
                    zorder=2)
    ax.plot(CTR[ok], m_[ok], color=col, lw=2.6, zorder=5, label=lab,
            solid_capstyle='round')
    print(f"  {t} tip: drawn to {CTR[ok][-1]:.1f} deg")
mm_all = pooled(np.ones(len(phi), bool), mod)
ok = ~np.isnan(mm_all)
ax.plot(CTR[ok], mm_all[ok], color=MOD, lw=2.6, zorder=6, label='model',
        solid_capstyle='round')
if THIN is not None:
    ax.axvspan(THIN, EDGES[-1], color=MUTED, alpha=0.16, lw=0, zorder=0)
dress(ax, '(a)  per tip direction')
ax.legend(fontsize=10, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.17), ncol=3, labelcolor=INK2)
YL = ax.get_ylim()

# ---- (b) after the pivot-free average -------------------------------
ax = axes[1]
# The line and the band must be the same estimator, or the line can sit
# outside its own band -- as it did when the line was the pooled median
# of all runs while the band was the quartiles across groups.  The
# symmetric combination only exists per group (a run tips one way
# only), so both are taken across groups.
ok = (~np.isnan(sym_g)).sum(axis=0) >= MIN_G
sym = np.where(ok, np.nanmedian(np.where(ok, sym_g, np.nan), axis=0),
               np.nan)
ax.fill_between(CTR[ok], np.nanpercentile(sym_g, 25, axis=0)[ok],
                np.nanpercentile(sym_g, 75, axis=0)[ok], color=DYN,
                alpha=0.16, lw=0, zorder=2)
# The set of groups that can form an average shrinks with tilt, and it
# does not shrink evenly: by 4.2 deg every My group has dropped out and
# the curve is an Mx-only statistic, while at low tilt it is balanced
# five and five.  Draw the balanced part solid and the rest faint, and
# report the residual over the balanced part.
ngrp = (~np.isnan(sym_g)).sum(axis=0)
full = ok & (ngrp == len(GROUPS))
ax.plot(CTR[ok], sym[ok], color=DYN, lw=2.0, alpha=0.35, zorder=4,
        solid_capstyle='round')
ax.plot(CTR[full], sym[full], color=DYN, lw=2.8, zorder=5,
        label='dynamic inversion, direction-averaged',
        solid_capstyle='round')
ax.plot(CTR[ok], mm_all[ok], color=MOD, lw=2.8, zorder=6,
        label='image-superposition model', solid_capstyle='round')
for i in np.flatnonzero(ok):
    ax.annotate(f'{ngrp[i]}', (CTR[i], 0.03), xycoords=('data', 'axes fraction'),
                ha='center', color=INK2 if ngrp[i] == len(GROUPS) else MUTED,
                fontsize=8)
ax.text(0.015, 0.095, 'groups averaged', transform=ax.transAxes,
        color=INK2, fontsize=8)
if THIN is not None:
    ax.axvspan(THIN, EDGES[-1], color=MUTED, alpha=0.16, lw=0, zorder=0)
    ax.annotate('fewer than half the\nruns still tipping', (THIN, 0.97),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                xytext=(5, -4), va='top', color=INK2, fontsize=9,
                linespacing=1.35)
dress(ax, '(b)  after the pivot-free average')
ax.set_ylim(YL)
ax.legend(fontsize=10, frameon=False, loc='upper left', labelcolor=INK2)
res = sym[ok] - mm_all[ok]
res_s = (sym - mm_all)[full]
ax.text(0.03, 0.20,
        f'over the balanced range {CTR[full][0]:.1f}–{CTR[full][-1]:.1f}°:  '
        f'median {np.median(res_s):+.0f}, RMS '
        f'{np.sqrt(np.mean(res_s**2)):.0f} mNm\n'
        f'faint: fewer groups reach the bin, My dropping out first\n'
        f'band: interquartile range across case/axis groups',
        transform=ax.transAxes, fontsize=9.5, color=INK, linespacing=1.4,
        bbox=dict(fc=SURF, ec=MUTED, lw=0.5, pad=4, alpha=0.93))

fig.suptitle('the same comparison, before and after the combination the '
             'identification forms — 140 runs',
             fontsize=12, color=INK, x=0.075, ha='left', y=0.965)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'fig_ge_dynamics_sym.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ge_dynamics_sym.png', bbox_inches='tight', dpi=190)
print(f"-> {OUT / 'fig_ge_dynamics_sym.pdf'}")
print(f"   averaged curve spans {CTR[ok][0]:.1f}-{CTR[ok][-1]:.1f} deg")
print(f"   balanced {CTR[full][0]:.1f}-{CTR[full][-1]:.1f} deg (all "
      f"{len(GROUPS)} groups):  median {np.median(res_s):+.1f}"
      f"  RMS {np.sqrt(np.mean(res_s ** 2)):.1f}")
print(f"   residual  median {np.median(res):+.1f}  RMS "
      f"{np.sqrt(np.mean(res ** 2)):.1f}  max |{np.max(np.abs(res)):.0f}| mNm")
for c, v, s_, m_ in zip(CTR[ok], res, sym[ok], mm_all[ok]):
    print(f"     phi {c:4.1f}   inversion {s_:6.1f}   model {m_:6.1f}"
          f"   diff {v:+7.1f}")
