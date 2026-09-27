#!/usr/bin/env python3
"""The piecewise nonlinear least-squares fit and the cost it minimises.

One run, drawn at the calibration-time form of the fit: the onset is
seeded by the time-quadratic piecewise fit (the small-angle limit of
the same solution) and swept over a local bracket, and at every
candidate onset the three shape parameters (C1, C2, C) are re-fitted by
bounded nonlinear least squares.

  (a) the measured body rate with the fitted baseline-plus-cosh curve,
      the seed and the selected onset j*
  (b) the profiled cost of (3.8c) over the candidate onsets, with the
      bracket it is allowed to explore

The bracket is what keeps the fit honest.  The cost is nearly
degenerate along the C1-onset ridge -- an onset earlier by D is paid
for by an amplitude smaller by exp(C2 D), leaving the curve almost
unchanged -- so an unrestricted sweep slides down that ridge into the
pre-excitation transient.  Panel (b) shows the shallow tail this
leaves to the left of the minimum.

Usage: python analysis/pnls_cost_figure.py [case] [axis] [run] [out.png]
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
from utils.extractor import load_excitation_dataset

FIT = '#c0392b'          # the fit and the selected onset
SEED = '#1f77b4'         # the time-quadratic seed
SIG = '#1f3b66'          # measured signal
BRK = '#f39c12'          # the sweep bracket


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_045'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_pnls_cost.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        bag = next(b for b in bags if b.name == run)
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    w = slice(i0, i1 + 1)
    t, om, M = sig['t'][w], sig['omega'][w], sig['moment'][w]
    t = t - t[0]
    m_dot = float(np.polyfit(t, M, 1)[0])

    seed = cvp.piecewise_onset_fit(t, om)['onset_idx']
    pw = cvp.cosh_onset_fit(t, om, M, onset_guess=seed, moment_floor=0.0)
    j, C1, C2, C = pw['onset_idx'], *pw['params']
    K = abs(C1) / abs(m_dot)

    cost = pw['cost_of']
    js = np.array(sorted(cost))
    cs = np.array([cost[k] for k in js])

    # The same profiled cost outside the bracket, so the ridge the
    # bracket exists to block can be seen.  Each point is one call of
    # the locator with its sweep collapsed to a single candidate, so it
    # is the library's own cost, not a re-implementation of it.
    wide = range(8, len(t) - 8, max(1, (len(t) - 16) // 160))
    wj, wc = [], []
    for g in wide:
        r = cvp.cosh_onset_fit(t, om, M, onset_guess=g, sweep_back_s=0.0,
                               sweep_ahead_s=0.0, moment_floor=0.0)
        wj.append(g)
        wc.append(r['cost_of'][g])
    wj, wc = np.array(wj), np.array(wc)
    inside = (wj >= js[0]) & (wj <= js[-1])
    chk = max((abs(wc[i] - cost[int(wj[i])]) / cost[int(wj[i])]
               for i in np.flatnonzero(inside) if int(wj[i]) in cost),
              default=0.0)
    print(f'  wide profile agrees with the bracket sweep to {chk:.2e} '
          f'(relative)')
    j_wide = int(wj[int(np.argmin(wc))])

    print(f'{case}/{axis_dir} {bag.name}')
    print(f'  seed j    = {seed}  (t = {t[seed]:.3f} s, time-quadratic)')
    print(f'  bracket   = [{pw["sweep_lo"]}, {pw["sweep_hi"]}]  '
          f'({len(js)} candidates)')
    print(f'  onset j*  = {j}  (t = {t[j]:.3f} s), displaced {j - seed:+d} '
          f'samples from the seed')
    print(f'  C1 = {C1:+.4f} rad/s   C2 = {C2:.3f} rad/s   '
          f'C = {C:+.4f} rad/s')
    print(f'  Mdot = {m_dot:+.4f} N.m/s  ->  K = |C1|/|Mdot| = {K:.4f}')

    fig, ax = plt.subplots(2, 1, figsize=(5.4, 5.6),
                           gridspec_kw=dict(height_ratios=[1.15, 1],
                                            hspace=0.42))

    # ── (a) the fit ──────────────────────────────────────────────
    A = ax[0]
    A.plot(t, om, '.', ms=2.6, color='0.55', zorder=1, label=r'measured $\omega$')
    A.plot(t, pw['omega_pred'], '-', lw=1.6, color=FIT, zorder=3,
           label='PNLS fit')
    A.axhline(C, ls=':', lw=1.0, color='0.35', zorder=2)
    A.axvline(t[seed], ls='--', lw=1.1, color=SEED, alpha=0.85, zorder=2,
              label='seed (time-quadratic)')
    A.axvline(t[j], ls='--', lw=1.2, color=FIT, alpha=0.9, zorder=2,
              label=r'$j^{*}$')
    A.plot(t[j], om[j], 'o', ms=7, mfc='none', mec=FIT, mew=1.6, zorder=4)
    A.text(0.03, 0.95,
           '\n'.join((rf'$C_1 = {C1:+.3f}$,  $C_2 = {C2:.2f}$ rad/s',
                      rf'$C = {C:+.4f}$ rad/s',
                      rf'$K = |C_1|/|\dot M| = {K:.3f}$')),
           transform=A.transAxes, fontsize=7.8, va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3.0))
    A.set_ylabel(rf'$\omega_{axis}$ [rad/s]', fontsize=9)
    A.set_title('(a) piecewise fit at the selected onset', fontsize=9.5)
    A.legend(fontsize=7.2, loc='center left', framealpha=0.9)

    # ── (b) the profiled cost over the candidates ────────────────
    B = ax[1]
    B.axvspan(t[js[0]], t[js[-1]], color=BRK, alpha=0.13, lw=0, zorder=0)
    B.plot(t[wj], wc, '-', lw=0.9, color='0.62', zorder=1,
           label='outside the bracket')
    B.plot(t[js], cs, '-', lw=1.4, color=SIG, zorder=3,
           label='swept')
    B.axvline(t[seed], ls='--', lw=1.1, color=SEED, alpha=0.85, zorder=2)
    B.axvline(t[j], ls='--', lw=1.2, color=FIT, alpha=0.9, zorder=2)
    B.plot(t[j], cost[j], 'o', ms=7, mfc='none', mec=FIT, mew=1.6, zorder=4)
    B.set_yscale('log')
    B.set_ylabel(r'$\mathrm{cost}(j)$  [(rad/s)$^2$]', fontsize=9)
    B.set_xlabel('candidate onset $t_j$ [s]', fontsize=9)
    B.set_title('(b) profiled cost over the candidate onsets', fontsize=9.5)
    B.text(t[seed], B.get_ylim()[1], ' seed', color=SEED, fontsize=7.8,
           ha='left', va='top')
    B.text(t[j], B.get_ylim()[1], r'$j^{*}$ ', color=FIT, fontsize=8.5,
           ha='right', va='top')
    B.text(t[js[0]] + 0.04 * (t[js[-1]] - t[js[0]]), 0.46,
           r'bracket  $[-0.10,\,+0.30]$ s',
           transform=B.get_xaxis_transform(), color='#a5690a',
           fontsize=7.8, ha='center', va='center', rotation=90)
    B.text(0.05, 0.20, 'near-degenerate ridge: an earlier onset is\n'
           r'paid for by an amplitude smaller by $e^{C_2\Delta}$',
           transform=B.transAxes, fontsize=7.2, color='0.35',
           ha='left', va='bottom')
    B.legend(fontsize=7.2, loc='upper left', framealpha=0.9)

    for A_ in ax:
        A_.tick_params(labelsize=8)
        A_.grid(alpha=0.25, lw=0.4)
    ax[0].set_xlabel('$t$ [s]', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'written {out}')


if __name__ == '__main__':
    main()
