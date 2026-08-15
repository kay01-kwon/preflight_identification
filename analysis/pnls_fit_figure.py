#!/usr/bin/env python3
"""What the reported fit actually looks like, on every configuration.

The residual table of Sec. VI-E is a set of percentages; this draws the
curves behind them, at the frozen two-stage constants that every
reported number was produced with (analysis/pnls_constants.py), so the
quality of the fit can be judged rather than inferred.

For each of the ten configurations, at one ramp rate: the measured rate
against the predicted curve with the onset marked, and beneath it the
residual on its own axis with the pre-onset noise band drawn, so a
deformation of the curve can be told apart from a noisier run.

Nothing here is fitted per run.  C1 = K Mdot with Mdot measured, C2 is
the configuration's constant, the baseline is the pre-onset median, and
only the onset index is searched -- so the post-onset curve is a
prediction and the residual means something.

Usage: python analysis/pnls_fit_figure.py [rate] [out.png]
       rate defaults to 1.20 N.m/s, where the curvature is sharpest and
       the fit has the least room to hide.
"""
import contextlib
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

DIRECTION = 'pos'


def one(case, axis_dir, rate):
    axis = 'x' if axis_dir == 'Mx' else 'y'
    c2, k = PNLS_CONSTANTS[(case, axis_dir)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
    for bag in bags:
        if cvp.commanded_ramp_rate(bag.name) != rate:
            continue
        if not bag.name.lower().startswith(DIRECTION):
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
        return dict(t=t - t[0], om=om, pred=pw['omega_pred'],
                    j=pw['onset_idx'], c2=c2, k=k, c=float(pw['c']))
    return None


def main():
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 1.20
    out = sys.argv[2] if len(sys.argv) > 2 else f'pnls_fit_{rate:.2f}.png'
    keys = sorted(PNLS_CONSTANTS)

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(4, 5, hspace=0.55, wspace=0.30,
                          height_ratios=[2.4, 1, 2.4, 1])
    print(f"  {'configuration':16}{'C2':>8}{'K':>8}{'endpoint':>10}"
          f"{'RMS':>8}{'floor':>8}   [% of peak]")
    for i, key in enumerate(keys):
        r = one(*key, rate)
        if r is None:
            continue
        block, col = divmod(i, 5)
        ax = fig.add_subplot(gs[2 * block, col])
        axr = fig.add_subplot(gs[2 * block + 1, col], sharex=ax)
        t, om, pred, j = r['t'], r['om'], r['pred'], r['j']
        base = float(np.median(om[:j]))
        span = float(np.max(np.abs(om[j:] - base)))
        res = om - pred
        floor = float(np.std(om[:j] - base))
        end = 100 * abs(float(np.mean(res[j:][-3:]))) / span
        rms = 100 * float(np.sqrt(np.mean(res[j:] ** 2))) / span
        print(f"  {key[0] + '/' + key[1]:16}{r['c2']:8.3f}{r['k']:8.4f}"
              f"{end:9.2f}%{rms:7.2f}%{100 * floor / span:7.2f}%")

        ax.plot(t, om, '.', ms=2.5, color='0.55', zorder=1)
        ax.plot(t, pred, '-', lw=1.6, color='#c0392b', zorder=3)
        ax.axvline(t[j], color='#c0392b', lw=0.9, ls='--', alpha=0.6)
        ax.plot(t[j], om[j], 'o', ms=6, mfc='none', mec='#c0392b', mew=1.6,
                zorder=4)
        ax.set_title(f"{key[0].replace('case_', 'case ')}/{key[1]}\n"
                     rf"$C_2$={r['c2']:.2f}, $K$={r['k']:.3f}", fontsize=8.5)
        ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=7.5)
        ax.tick_params(labelsize=6.5, labelbottom=False)
        ax.grid(alpha=0.25, lw=0.4)

        axr.axhspan(-floor, floor, color='#2874a6', alpha=0.16, zorder=0)
        axr.plot(t, res, '-', lw=0.7, color='0.30', zorder=2)
        axr.axvline(t[j], color='#c0392b', lw=0.9, ls='--', alpha=0.6)
        axr.axhline(0, color='k', lw=0.5, alpha=0.5)
        axr.set_ylabel(r'$\omega-\hat\omega$', fontsize=7.5)
        axr.set_xlabel('$t$ [s]', fontsize=7.5)
        axr.tick_params(labelsize=6.5)
        axr.grid(alpha=0.25, lw=0.4)
        axr.text(0.03, 0.06, f'end {end:.1f}%   rms {rms:.1f}%',
                 transform=axr.transAxes, fontsize=6.5, color='#c0392b')

    fig.suptitle(rf'Reported fit at $\dot M$ = {rate:.2f} N$\cdot$m/s, '
                 f'{DIRECTION} direction. Grey: measured. Red: predicted, no '
                 f'per-run shape parameter.\nLower panel: residual, with the '
                 f'pre-onset noise band shaded.', fontsize=11, y=0.985)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
