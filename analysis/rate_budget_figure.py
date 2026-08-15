#!/usr/bin/env python3
"""The rate bound against what the runs actually leave, rate by rate.

The comparison only means something per ramp rate, because the two
things being compared have opposite rate dependence.  The VI-E bound
falls fivefold across the sweep -- the moment swept before the tilt
limit collapses as Mdot^(2/3) -- while the measured residual is flat,
and so is the floor set by the onset grid.  Where the bound crosses
below that floor it must be exceeded, and it does.

Top two rows: one configuration through all seven rates, the measured
rate against the predicted curve with the onset marked, and the residual
beneath with the pre-onset noise band shaded.  Constants are the frozen
two-stage pair, so this is the reported fit.

Bottom row, pooled over all 140 runs:

  left    endpoint residual against the bound at the 10-degree design
          limit, the bound re-solved at the window each rate reached,
          and the quantisation floor.  The onset is an integer index at
          100 Hz, so |dt| <= Ts/2 and the endpoint keeps dt*omega_dot.
  middle  the same in absolute terms, with the pre-onset noise floor.
  right   converted to threshold: Ts/2 * Mdot against the (109) budget.

Usage: python analysis/rate_budget_figure.py [case_01] [Mx] [out.png]
"""
import contextlib
import collections
import csv
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

TS = 0.010                      # 100 Hz: the onset index is integer
BUDGET = 0.01204                # (109), roll, worst admissible [N.m]
# (110) evaluated at the 10-degree design limit, and re-solved at the
# window each rate actually reached.  From analysis/rate_residual.py.
CAP = {0.10: 35.9, 0.20: 20.1, 0.30: 14.3, 0.45: 10.3,
       0.65: 7.6, 0.90: 5.9, 1.20: 4.7}
REALISED = {0.10: 8.7, 0.20: 4.7, 0.30: 3.9, 0.45: 3.6,
            0.65: 2.6, 0.90: 2.0, 1.20: 1.7}


def fits_by_rate(case, axis_dir):
    axis = 'x' if axis_dir == 'Mx' else 'y'
    c2, k = PNLS_CONSTANTS[(case, axis_dir)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
    out = {}
    for bag in bags:
        rate = cvp.commanded_ramp_rate(bag.name)
        if rate is None or not bag.name.lower().startswith('pos'):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            sig = cvp.prepare_signals(bag, axis)
        i0, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
        w = slice(i0, i1 + 1)
        t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
        md = float(np.polyfit(t, M, 1)[0])
        pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                c2_fixed=c2, moment_floor=0.0,
                                ramp_gain=k, ramp_rate=md)
        out[rate] = (t - t[0], om, pw['omega_pred'], pw['onset_idx'],
                     float(pw['c']))
    return out, c2, k


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_01'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'Mx'
    out = sys.argv[3] if len(sys.argv) > 3 else f'rate_budget_{case}.png'

    runs, c2, k = fits_by_rate(case, axis_dir)
    rates = sorted(runs)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 7, hspace=0.5, wspace=0.34,
                          height_ratios=[2.3, 1.0, 2.6])

    for i, rate in enumerate(rates):
        t, om, pred, j, c = runs[rate]
        ax, axr = fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])
        base = float(np.median(om[:j]))
        span = float(np.max(np.abs(om[j:] - base)))
        res = om - pred
        floor = float(np.std(om[:j] - base))
        end = 100 * abs(float(np.mean(res[j:][-3:]))) / span
        ax.plot(t, om, '.', ms=2.2, color='0.55', zorder=1)
        ax.plot(t, pred, '-', lw=1.5, color='#c0392b', zorder=3)
        ax.axvline(t[j], color='#c0392b', lw=0.9, ls='--', alpha=0.6)
        ax.plot(t[j], om[j], 'o', ms=6, mfc='none', mec='#c0392b', mew=1.6,
                zorder=4)
        ax.set_title(rf'$\dot M$ = {rate:.2f}' + '\n'
                     + rf'$x = C_2\tau_{{\rm end}}$ = '
                     + f'{c2 * (t[-1] - t[j]):.2f}', fontsize=8.5)
        ax.tick_params(labelsize=6.5, labelbottom=False)
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=8)
        axr.axhspan(-floor, floor, color='#2874a6', alpha=0.16, zorder=0)
        axr.plot(t, res, '-', lw=0.7, color='0.30', zorder=2)
        axr.axvline(t[j], color='#c0392b', lw=0.9, ls='--', alpha=0.6)
        axr.axhline(0, color='k', lw=0.5, alpha=0.5)
        axr.set_xlabel('$t$ [s]', fontsize=7.5)
        axr.tick_params(labelsize=6.5)
        axr.grid(alpha=0.25, lw=0.4)
        axr.text(0.04, 0.07, f'{end:.1f}%', transform=axr.transAxes,
                 fontsize=7, color='#c0392b')
        if i == 0:
            axr.set_ylabel(r'$\omega-\hat\omega$', fontsize=8)

    # ── pooled budget panels ─────────────────────────────────────────
    rows = list(csv.DictReader(open(ROOT / 'rate_residual_runs.csv')))
    g = collections.defaultdict(list)
    for r in rows:
        g[float(r['rate'])].append(r)
    rr = sorted(g)

    def med(rate, col, f=float):
        return np.median([f(r[col]) for r in g[rate]])

    end_med = np.array([med(r, 'end_pct') for r in rr])
    end_lo = np.array([np.percentile([float(x['end_pct']) for x in g[r]], 25)
                       for r in rr])
    end_hi = np.array([np.percentile([float(x['end_pct']) for x in g[r]], 75)
                       for r in rr])
    fl = np.array([med(r, 'floor_pct') for r in rr])
    quant = []
    for r in rr:
        x, pk = med(r, 'x_fit'), med(r, 'span')
        c1 = pk / (np.cosh(x) - 1)
        quant.append(100 * (TS / 2) * c1 * med(r, 'c2_fit') * np.sinh(x) / pk)
    quant = np.array(quant)

    ax = fig.add_subplot(gs[2, :3])
    ax.fill_between(rr, end_lo, end_hi, color='0.55', alpha=0.25,
                    label='measured, IQR')
    ax.plot(rr, end_med, 'o-', color='0.20', lw=1.8, ms=5,
            label='measured endpoint residual')
    ax.plot(rr, [CAP[r] for r in rr], 's--', color='#c0392b', lw=1.6, ms=5,
            label=r'(110)--(112) at the $10^\circ$ limit')
    ax.plot(rr, [REALISED[r] for r in rr], '^--', color='#e67e22', lw=1.6,
            ms=5, label='same, at the window reached')
    ax.plot(rr, quant, 'd-', color='#2874a6', lw=1.6, ms=5,
            label=r'onset grid, $T_s/2\cdot\dot\omega_{\rm end}$')
    ax.plot(rr, fl, 'v:', color='#1e8449', lw=1.4, ms=5,
            label='pre-onset noise floor')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(rr)
    ax.set_xticklabels([f'{r:g}' for r in rr])
    ax.set_xlabel(r'$\dot M$ [N$\cdot$m/s]', fontsize=9)
    ax.set_ylabel('endpoint deviation [% of peak]', fontsize=9)
    ax.set_title('the bound falls, the measurement does not', fontsize=10)
    ax.legend(fontsize=7, loc='lower left', framealpha=0.92)
    ax.grid(alpha=0.3, which='both', lw=0.4)
    ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[2, 3:5])
    pk = np.array([med(r, 'span') for r in rr])
    ax2.plot(rr, end_med / 100 * pk, 'o-', color='0.20', lw=1.8, ms=5,
             label='measured')
    ax2.plot(rr, quant / 100 * pk, 'd-', color='#2874a6', lw=1.6, ms=5,
             label='onset grid')
    ax2.plot(rr, fl / 100 * pk, 'v:', color='#1e8449', lw=1.4, ms=5,
             label='noise floor')
    ax2.set_xscale('log')
    ax2.set_xticks(rr)
    ax2.set_xticklabels([f'{r:g}' for r in rr])
    ax2.set_xlabel(r'$\dot M$ [N$\cdot$m/s]', fontsize=9)
    ax2.set_ylabel(r'$|e_\omega|$ at the window end [rad/s]', fontsize=9)
    ax2.set_title('in absolute terms', fontsize=10)
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.grid(alpha=0.3, lw=0.4)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(gs[2, 5:])
    ax3.axhline(1e3 * BUDGET, color='#c0392b', lw=1.8,
                label='(109) budget, 12.04')
    ax3.plot(rr, [1e3 * TS / 2 * r for r in rr], 'd-', color='#2874a6',
             lw=1.8, ms=5, label=r'$T_s/2\cdot\dot M$')
    ax3.set_xlabel(r'$\dot M$ [N$\cdot$m/s]', fontsize=9)
    ax3.set_ylabel(r'$|\Delta M_{\rm crit}|$ [mN$\cdot$m]', fontsize=9)
    ax3.set_title('what the onset grid costs the threshold', fontsize=10)
    ax3.legend(fontsize=7.5, loc='upper left')
    ax3.grid(alpha=0.3, lw=0.4)
    ax3.tick_params(labelsize=8)
    for r in (0.45, 1.20):
        ax3.annotate(f'{100 * TS / 2 * r / BUDGET:.0f}% of budget',
                     (r, 1e3 * TS / 2 * r), textcoords='offset points',
                     xytext=(-4, 8), fontsize=7, color='#2874a6', ha='right')

    fig.suptitle(f'{case}/{axis_dir}, reported constants '
                 rf'$C_2$={c2:.2f}, $K$={k:.3f}. Top: the fit at each ramp '
                 f'rate. Bottom: pooled over all 140 runs.',
                 fontsize=12, y=0.965)
    fig.savefig(out, dpi=135, bbox_inches='tight')
    print(f"  wrote {out}")
    for i, r in enumerate(rr):
        print(f"  Mdot {r:5.2f}   measured {end_med[i]:6.2f}%"
              f"   cap {CAP[r]:6.1f}%   realised {REALISED[r]:5.1f}%"
              f"   grid {quant[i]:5.2f}%   floor {fl[i]:5.2f}%"
              f"   {'OK' if end_med[i] < CAP[r] else 'EXCEEDS'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
