#!/usr/bin/env python3
"""The method in one picture: the ramp, the onset, the critical moment.

One run at the calibrated effective constants (C2*, K*) of its dataset
(analysis/pnls_constants.py): the ramped moment on top, the measured
body rate with the fitted baseline-plus-cosh response below, and the
detected onset drawn through both so the critical moment is read off
the ramp at the instant the rate departs.

Nothing is fitted per run.  C1 = K* Mdot with Mdot measured from the
ramp, C2 = C2*, the baseline is the pre-onset median, and only the
onset index is searched -- so the red curve is a prediction of the
calibrated family, not a fit to this record.

Usage: python analysis/cosh_overview_figure.py [case] [axis] [run] [out.png]
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
from constrained_calibration import ROOT
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset

FIT = '#c0392b'          # the fitted response and the onset
SIG = '#1f3b66'          # measured signals
RMP = '#2874a6'          # the moment ramp
PAD = 0.4                # seconds of context drawn before the window


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_065'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_cosh_overview.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'
    c2, k = PNLS_CONSTANTS[(case, axis_dir)]

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        bag = next(b for b in bags if b.name == run)
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    w = slice(i0, i1 + 1)
    t_all, om_all, M_all = sig['t'], sig['omega'], sig['moment']
    t, om, M = t_all[w], om_all[w], M_all[w]
    md = float(np.polyfit(t, M, 1)[0])
    pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                            c2_fixed=c2, moment_floor=0.0,
                            ramp_gain=k, ramp_rate=md)
    j, C = pw['onset_idx'], float(pw['c'])
    t_c, M_crit = t[j], M[j]
    C1 = k * md

    print(f'{case}/{axis_dir} {bag.name}')
    print(f'  calibrated  C2* = {c2:.4f} rad/s   K* = {k:.4f} 1/(N.m)')
    print(f'  ramp        Mdot = {md:+.4f} N.m/s  ->  C1 = K* Mdot = {C1:+.4f}')
    print(f'  onset       t_c = {t_c:.3f} s   M_crit = {M_crit:+.4f} N.m   '
          f'C = {C:+.4f} rad/s')

    keep = (t_all >= t[0] - PAD) & (t_all <= t[-1])
    fig, (A, B) = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.0, 1.25],
                                                hspace=0.08))

    # ── applied moment: the ramp, with the critical moment read off ──
    A.plot(t_all[keep], M_all[keep], '-', lw=1.5, color=RMP, zorder=3)
    A.axhline(M_crit, ls='--', lw=1.0, color=FIT, alpha=0.8, zorder=2)
    A.axvline(t_c, ls='--', lw=1.1, color=FIT, alpha=0.8, zorder=2)
    A.plot(t_c, M_crit, 'o', ms=7, mfc='white', mec=FIT, mew=1.8, zorder=5)
    A.annotate(rf'$M_{{\mathrm{{crit}}}} = {M_crit:+.3f}$ N$\cdot$m',
               (t_c, M_crit), textcoords='offset points', xytext=(-8, 14),
               fontsize=9.5, color=FIT, ha='right')
    A.text(0.03, 0.93, rf'$\dot M = {md:+.2f}$ N$\cdot$m/s', color=RMP,
           fontsize=9, transform=A.transAxes, va='top')
    A.set_ylabel(rf'$M_{axis}$ [N$\cdot$m]', fontsize=10)

    # ── body rate: the calibrated cosh family, onset only searched ──
    B.plot(t_all[keep], om_all[keep], '.', ms=3.2, color='0.55', zorder=1,
           label=r'measured $\omega$')
    B.plot(t, pw['omega_pred'], '-', lw=1.8, color=FIT, zorder=3,
           label=r'$C_1(\cosh C_2^{*}\tau - 1) + C$')
    B.axvline(t_c, ls='--', lw=1.1, color=FIT, alpha=0.8, zorder=2)
    B.plot(t_c, om[j], 'o', ms=7, mfc='white', mec=FIT, mew=1.8, zorder=5)
    B.text(t_c, B.get_ylim()[1], r' $t_c$ (onset)', color=FIT, fontsize=9.5,
           ha='left', va='top')
    B.text(0.03, 0.93,
           '\n'.join((rf'$C_2^{{*}} = {c2:.2f}$ rad/s,  '
                      rf'$K^{{*}} = {k:.3f}$ (N$\cdot$m)$^{{-1}}$',
                      rf'$C_1 = K^{{*}}\dot M = {C1:+.3f}$ rad/s')),
           transform=B.transAxes, fontsize=8.8, va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3.0))
    B.set_ylabel(rf'$\omega_{axis}$ [rad/s]', fontsize=10)
    B.set_xlabel('$t$ [s]', fontsize=10)
    B.legend(fontsize=8.5, loc='center left', framealpha=0.9)

    for ax in (A, B):
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'written {out}')


if __name__ == '__main__':
    main()
