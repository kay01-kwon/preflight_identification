#!/usr/bin/env python3
"""The configuration whose fitted C2 exceeds the geometric ceiling.

Sec. VI-E bounds C2 from geometry alone, the mass cancelling:

    C2^2 = W z_CoM / J_P <= g z_CoM / (z_CoM^2 + (l_p + p_off)^2),

which gives 5.05 /s on roll.  The pipeline's own closed-form calibration
returns 8.000 for case_01/Mx -- 59% above a quantity that is supposed to
be an upper bound.  Since (110) is used as a bound in the manuscript,
that needs looking at rather than a footnote.

Two things are suspicious before any plot is drawn.  8.000 is exactly
the top of the search interval c2_bounds = (3.0, 8.0), so the objective
may simply be running into the wall.  And the frozen two-stage constants
of analysis/pnls_constants.py give 5.2988 for the same configuration, so
the two calibration routes disagree by 51%.

This script draws what the data actually says:

  * per rate, the measured rate against the fitted curve, with the onset
    marked -- and the same fit redone at the two competing C2 values, so
    the eye can judge whether the data discriminates between them;
  * the calibration objective itself as a function of C2, which settles
    whether 8.000 is a minimum or a boundary.

Usage: python analysis/c2_ceiling_figure.py [case_01] [Mx] [out.png]
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

G, W, Z = 9.81, 31.59, 0.30
ARMS = {'Mx': 0.140 + 0.020, 'My': 0.110 + 0.020}


def ceiling(axis):
    a = ARMS[axis]
    return np.sqrt(G * Z / (Z ** 2 + a ** 2))


def refit(t, om, c2, k, m_dot):
    """The same constrained fit, at a different C2."""
    return cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                              c2_fixed=float(c2), moment_floor=0.0,
                              ramp_gain=float(k), ramp_rate=float(m_dot))


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_01'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'Mx'
    out = sys.argv[3] if len(sys.argv) > 3 else f'{case}_{axis_dir}_c2.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'
    d = ROOT / case / axis_dir

    with contextlib.redirect_stdout(io.StringIO()) as buf:
        bags = load_excitation_dataset(d)
        crits, fits = cvp.extract_piecewise_batch(bags, axis)
    log = buf.getvalue()
    c2_pipe = float(fits[0]['alpha'])
    k_gain = None
    for line in log.splitlines():
        if 'Rig constants' in line:
            k_gain = float(line.split('K=')[1].split()[0])
    c2_pnls = PNLS_CONSTANTS[(case, axis_dir)][0]
    c2_geo = ceiling(axis_dir)
    print(f"  {case}/{axis_dir}")
    print(f"    pipeline calibration   C2 = {c2_pipe:.4f}"
          f"   ({100 * (c2_pipe / c2_geo - 1):+.1f}% vs ceiling)")
    print(f"    frozen two-stage       C2 = {c2_pnls:.4f}"
          f"   ({100 * (c2_pnls / c2_geo - 1):+.1f}%)")
    print(f"    geometric ceiling (110) C2 = {c2_geo:.4f}")
    print(f"    search interval        {cvp.cosh_onset_fit.__defaults__[3]}"
          f"   <- note where the pipeline value sits")
    print(f"    ramp gain K = {k_gain}")

    # ── per-run panels ────────────────────────────────────────────────
    seen = {b.name: b for b in bags}
    runs = []
    for crit, pw in zip(crits, fits):
        rate = cvp.commanded_ramp_rate(crit.bag_name)
        if rate is None or not crit.bag_name.lower().startswith('pos'):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            sig = cvp.prepare_signals(seen[crit.bag_name], axis)
        i0, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
        win = slice(i0, i1 + 1)
        t, om, M = sig['t'][win], sig['omega'][win], sig['moment'][win]
        m_dot = float(np.polyfit(t, M, 1)[0])
        runs.append((rate, t - t[0], om, pw, m_dot))
    runs.sort(key=lambda r: r[0])

    n = len(runs)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    for i, (rate, t, om, pw, m_dot) in enumerate(runs[:7]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.plot(t, om, '.', ms=2.5, color='0.55', label='measured', zorder=1)
        for c2, style, col, lab in (
                (c2_pipe, '-', '#c0392b', f'fit, $C_2$={c2_pipe:.2f}'),
                (c2_pnls, '--', '#2874a6', f'$C_2$={c2_pnls:.2f} (two-stage)'),
                (c2_geo, ':', '#1e8449', f'$C_2$={c2_geo:.2f} (ceiling)')):
            p = refit(t, om, c2, k_gain, m_dot)
            ax.plot(t, p['omega_pred'], style, lw=1.5, color=col, label=lab,
                    zorder=3)
            ax.axvline(t[p['onset_idx']], color=col, lw=0.8, ls=style,
                       alpha=0.45, zorder=2)
        j = pw['onset_idx']
        ax.plot(t[j], om[j], 'o', ms=7, mfc='none', mec='#c0392b', mew=1.8,
                zorder=4)
        ax.set_title(rf'$\dot M$ = {rate:.2f} N$\cdot$m/s', fontsize=10)
        ax.set_xlabel('$t$ from window start [s]', fontsize=8)
        ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)

    # ── the objective as a function of C2 ─────────────────────────────
    prepared = []
    for bag in bags:
        with contextlib.redirect_stdout(io.StringIO()):
            sig = cvp.prepare_signals(bag, axis)
        i0, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
        win = slice(i0, i1 + 1)
        t, om, M = sig['t'][win], sig['omega'][win], sig['moment'][win]
        side = 'neg' if bag.name.lower().startswith('neg') else 'pos'
        prepared.append((side, t - t[0], om, M,
                         float(np.polyfit(t, M, 1)[0])))

    grid = np.arange(3.0, 9.51, 0.125)
    score, cost = [], []
    for c2 in grid:
        groups, tot = {}, 0.0
        for side, t, om, M, md in prepared:
            p = refit(t, om, c2, k_gain, md)
            groups.setdefault(side, []).append(float(M[p['onset_idx']]))
            tot += float(p['total_residual'])
        s = 0.0
        for v in groups.values():
            mu = abs(float(np.mean(v)))
            s += float(np.std(v)) / mu if mu > 1e-9 else np.inf
        score.append(s)
        cost.append(tot)
    score, cost = np.array(score), np.array(cost)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(grid, 100 * score, '-', lw=1.6, color='#c0392b')
    ax.axvspan(3.0, 8.0, color='0.9', zorder=0)
    for c2, col, lab in ((c2_pipe, '#c0392b', 'pipeline'),
                         (c2_pnls, '#2874a6', 'two-stage'),
                         (c2_geo, '#1e8449', 'ceiling (110)')):
        ax.axvline(c2, color=col, lw=1.2, ls='--', alpha=0.8)
        ax.text(c2, ax.get_ylim()[1], f' {lab}', rotation=90, va='top',
                ha='left', fontsize=6.5, color=col)
    ax.set_title('calibration objective', fontsize=10)
    ax.set_xlabel('$C_2$ [1/s]', fontsize=8)
    ax.set_ylabel('CV of $|M_{crit}|$, summed over\ntip directions [%]',
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.text(0.02, 0.03, 'shaded: search interval (3.0, 8.0)',
            transform=ax.transAxes, fontsize=6.5, color='0.35')

    fig.suptitle(f'{case}/{axis_dir}: the fitted $C_2$ against the geometric '
                 f'ceiling of (110)', fontsize=12, y=0.975)
    fig.savefig(out, dpi=145, bbox_inches='tight')
    print(f"\n  objective minimum at C2 = {grid[np.argmin(score)]:.3f}"
          f" (CV {100 * score.min():.2f}%),"
          f" at the interval edge 8.0 it is {100 * score[grid == 8.0][0]:.2f}%")
    print(f"  best total residual at C2 = {grid[np.argmin(cost)]:.3f}")
    print(f"  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
