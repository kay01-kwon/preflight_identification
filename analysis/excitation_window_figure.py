#!/usr/bin/env python3
"""The excitation window and the onset-free run gates, on one run.

Everything the detector needs before it looks at the onset: the applied
moment reconstructed from the rotor speeds, the body rate it produces,
the window that the ramp defines, and the three gates that decide
whether the run is admitted -- all read off M(t) alone, so nothing here
depends on where the onset turns out to be.

  (a) M(t) with the +-10 mN.m detection threshold, the window
      [i_start, i_end], the least-squares ramp line whose slope is the
      measured Mdot, and the allocator feasibility limit M_max that
      truncates the window when the allocation saturates
  (b) omega(t) over the same span, with the window marked
  (c) the ramp residual M - Mhat with the linearity gate band, and the
      realized values of the three gates

Usage: python analysis/excitation_window_figure.py [case] [axis] [run]
                                                   [out.png]
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

WIN = '#f39c12'          # excitation window
CMD = '#c0392b'          # fitted ramp / thresholds
SIG = '#1f3b66'          # measured signal
PAD = 1.2                # seconds of context drawn either side


def load(case, axis_dir, run):
    axis = 'x' if axis_dir == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_bags(ROOT / case / axis_dir)
    bag = next(b for b in bags if b.name == run)
    with contextlib.redirect_stdout(io.StringIO()):
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    return axis, bag, sig, i0, i1


def load_bags(d):
    from utils.extractor import load_excitation_dataset
    return load_excitation_dataset(d)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_045'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_excitation_window.png'

    axis, bag, sig, i0, i1 = load(case, axis_dir, run)
    t, om, M = sig['t'], sig['omega'], sig['moment']
    w = slice(i0, i1 + 1)
    tw, Mw = t[w], M[w]
    n_full = i1 - i0 + 1

    # measured ramp rate: least squares over the whole window
    a, b = np.polyfit(tw, Mw, 1)
    res = Mw - (a * tw + b)
    lin = float(np.std(res))

    # the gate slope is fitted on the sub-segment where an onset can
    # occur, so spin-up imperfections at near-zero moment cannot reject
    # a run whose ramp is within spec where it matters
    floor = cvp.SLOPE_GATE_FLOOR.get(axis, 0.0)
    mask = np.abs(Mw) >= floor
    a_gate = (np.polyfit(tw[mask], Mw[mask], 1)[0]
              if int(mask.sum()) >= 10 else a)
    cmd = cvp.commanded_ramp_rate(bag.name)
    eps = (abs(float(a_gate)) - cmd) / cmd * 100.0

    cap = cvp.MOMENT_CAP[axis]
    sgn = 1.0 if a >= 0 else -1.0
    lo, hi = tw[0] - PAD, tw[-1] + PAD
    keep = (t >= lo) & (t <= hi)

    print(f'{case}/{axis_dir} {bag.name}')
    print(f'  window      [{i0}, {i1}]  N_full = {n_full}  '
          f'{tw[-1] - tw[0]:.2f} s')
    print(f'  ramp        Mdot = {a:+.4f} N.m/s  (commanded {cmd:.2f}, '
          f'gate slope {a_gate:+.4f}, eps = {eps:+.2f}%)')
    print(f'  linearity   M_lin,RMSE = {lin * 1e3:.1f} mN.m')
    print(f'  peak |M|    {np.max(np.abs(Mw)):.3f} N.m   cap {cap:.2f} N.m')

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.5))

    # ── (a) applied moment, window and ramp fit ──────────────────
    A = ax[0]
    A.axvspan(tw[0], tw[-1], color=WIN, alpha=0.13, lw=0, zorder=0)
    A.plot(t[keep], M[keep], '-', lw=1.1, color=SIG, zorder=3)
    A.plot(tw, a * tw + b, '--', lw=1.3, color=CMD, zorder=4)
    A.axhline(sgn * cap, ls=':', lw=1.1, color='0.35')
    A.axhline(0.01 * sgn, ls='-', lw=0.8, color='0.45')
    A.plot([tw[0], tw[-1]], [M[i0], M[i1]], 'o', ms=6, mfc='none',
           mec=CMD, mew=1.5, zorder=5)
    A.annotate(r'$i_{\mathrm{start}}$', (tw[0], M[i0]),
               textcoords='offset points', xytext=(4, 12), fontsize=9,
               color=CMD)
    A.annotate(r'$i_{\mathrm{end}}$', (tw[-1], M[i1]),
               textcoords='offset points', xytext=(-26, -4), fontsize=9,
               color=CMD)
    A.text(0.28, 0.04, r'threshold $|M| = 10$ mN$\cdot$m',
           transform=A.transAxes, fontsize=7.5, color='0.35')
    A.text(0.97, sgn * cap, r'$M_{\max}$ (allocator limit)',
           transform=A.get_yaxis_transform(), fontsize=7.5, color='0.35',
           ha='right', va='top' if sgn > 0 else 'bottom',
           bbox=dict(fc='white', ec='none', pad=1.5))
    A.text(0.55, 0.30, rf'$\dot M = {a:+.3f}$ N$\cdot$m/s',
           transform=A.transAxes, fontsize=8.5, color=CMD)
    A.set_ylabel(rf'$M_{axis}$ [N$\cdot$m]', fontsize=9)
    A.set_title('(a) applied moment and window', fontsize=9.5)
    A.set_ylim(min(-0.12, sgn * cap * 1.12 if sgn < 0 else -0.12),
               max(0.12, sgn * cap * 1.12 if sgn > 0 else 0.12))

    # ── (b) body rate over the same span ─────────────────────────
    B = ax[1]
    B.axvspan(tw[0], tw[-1], color=WIN, alpha=0.13, lw=0, zorder=0)
    B.plot(t[keep], om[keep], '-', lw=0.9, color=SIG, zorder=3)
    for x in (tw[0], tw[-1]):
        B.axvline(x, ls='--', lw=1.0, color=CMD, alpha=0.75)
    B.axhline(0, lw=0.5, color='0.5')
    B.set_ylabel(rf'$\omega_{axis}$ [rad/s]', fontsize=9)
    B.set_title('(b) measured body rate', fontsize=9.5)
    B.text(0.5 * (tw[0] + tw[-1]), B.get_ylim()[1], 'excitation window',
           ha='center', va='top', fontsize=8, color='#a5690a')

    # ── (c) ramp residual and the realized gates ─────────────────
    C = ax[2]
    C.axhspan(-lin * 1e3, lin * 1e3, color='#2874a6', alpha=0.15, lw=0,
              zorder=0)
    for y in (-30, 30):
        C.axhline(y, ls='--', lw=1.0, color='#1a5276')
    C.plot(tw, res * 1e3, '-', lw=0.9, color=SIG, zorder=3)
    C.axhline(0, lw=0.5, color='0.5')
    C.set_ylabel(r'$M-\widehat{M}$ [mN$\cdot$m]', fontsize=9)
    C.set_title('(c) ramp residual and run gates', fontsize=9.5)
    C.set_ylim(-42, 42)
    C.text(0.04, 0.95,
           '\n'.join((rf'$\varepsilon = {eps:+.2f}\%$  '
                      r'$(\leq 3\%)$',
                      rf'$M_{{\mathrm{{lin,RMSE}}}} = {lin * 1e3:.1f}$'
                      r' mN$\cdot$m  $(\leq 30)$',
                      rf'$N_{{\mathrm{{full}}}} = {n_full}$  '
                      r'$(\geq 38)$')),
           transform=C.transAxes, fontsize=8, va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3.5))
    C.text(0.97, 30, r'linearity gate $\pm 30$ mN$\cdot$m',
           transform=C.get_yaxis_transform(), fontsize=7.5,
           color='#1a5276', ha='right', va='bottom')
    C.text(0.97, 0.04, rf'shaded: realized $\pm{lin * 1e3:.1f}$ mN$\cdot$m',
           transform=C.transAxes, fontsize=7.5, color='#2874a6', ha='right')

    for A in ax:
        A.set_xlabel('$t$ [s]', fontsize=9)
        A.tick_params(labelsize=8)
        A.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'written {out}')


if __name__ == '__main__':
    main()
