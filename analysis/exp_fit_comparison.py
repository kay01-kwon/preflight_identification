#!/usr/bin/env python3
"""Constrained COSH against the free PNLS fit on one measured run.

The E2/My positive tip at the fastest ramp (1.2 N.m/s), the run where
the two readouts disagree most visibly: the free fit halves the
residual by absorbing the sub-threshold compliance shoulder and reads
the threshold 1.1 N.m early, while the constrained fit ignores the
shoulder and reads it where the exponential branch begins -- within a
few mN.m of the independent static threshold (load-cell truth,
measured onset collective, mocap contact arm, rotor-interference
ground effect), drawn as the third vertical marker.

Usage
-----
  PYTHONPATH=<stubs> python analysis/exp_fit_comparison.py <scratch>
                     [--outdir DIR] [--dpi N]

<scratch> must hold mcrit_prediction.csv (analysis/mcrit_prediction.py)
for the static threshold.
"""
import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import critical_value_getter_piecewise as cvp          # noqa: E402
from utils.extractor import load_excitation_dataset    # noqa: E402
from analysis.pnls_constants import PNLS_CONSTANTS     # noqa: E402

CASE, SIMAX, BAG = 'case_02', 'My', 'pos_My_120'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scratch', type=Path,
                    help='directory holding mcrit_prediction.csv')
    ap.add_argument('--outdir', type=Path, default=Path('docs'))
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pred = {(r['case'], r['axis'], r['dir']): r
            for r in csv.DictReader(
                open(args.scratch / 'mcrit_prediction.csv'))}
    m_th = float(pred[(CASE, SIMAX, 'pos')]['M_pred_interf'])

    axis = 'x' if SIMAX == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(
            Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
            / CASE / SIMAX)
    bag = {b.name: b for b in bags}[BAG]
    c2_pn, k_pn = PNLS_CONSTANTS[(CASE, SIMAX)]
    with contextlib.redirect_stdout(io.StringIO()):
        crit_c, pw_c = cvp.extract_piecewise(bag, axis, model='cosh',
                                             cosh_c2=c2_pn,
                                             ramp_gain=k_pn)
        crit_f, pw_f = cvp.extract_piecewise(bag, axis, model='cosh',
                                             cosh_c2=None,
                                             ramp_gain=None)
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    t, om, M = sig['t'], np.degrees(sig['omega']), sig['moment']
    # time at which the applied moment crosses the static threshold
    j_th = i0 + int(np.argmax(M[i0:i1 + 1] >= m_th))
    t_th = float(t[j_th])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sl = slice(i0, i1 + 1)
    ax.plot(t[sl], om[sl], color='0.55', lw=0.9,
            label=r'measured $\omega$')
    # applied moment on a twin right axis
    ax2 = ax.twinx()
    ax2.plot(t[sl], M[sl], color='#7B4FA6', lw=1.4, alpha=0.85,
             label=r'applied moment $M$')
    ax2.set_ylabel(r'$M$ [N$\cdot$m]', fontsize=9.5, color='#7B4FA6')
    ax2.tick_params(axis='y', colors='#7B4FA6')
    for pw, crit, col, lab in (
            (pw_c, crit_c, '#0072B2', 'COSH'),
            (pw_f, crit_f, '#D55E00', 'PNLS')):
        prd = np.degrees(pw['omega_pred'])
        n = min(len(prd), i1 + 1 - i0)
        ax.plot(t[i0:i0 + n], prd[:n], color=col, lw=1.9,
                label=f'{lab}: $M_{{\\mathrm{{crit,est}}}}'
                      f' = {crit.onset_moment:+.3f}$ N·m')
        ax.axvline(crit.onset_time, color=col, lw=1.2, ls='--')
    ax.axvline(t_th, color='#009E73', lw=1.6, ls=(0, (4.5, 1.8)),
               label=f'static threshold (GE-corrected): '
                     f'$M_{{\\mathrm{{crit,th}}}} = {m_th:+.3f}$ N·m')
    ax.set_title(r'E2/$M_{y,+}$, $\dot M = 1.2$ N·m/s', loc='left',
                 fontsize=10)
    ax.set_xlabel('t [s]', fontsize=9.5)
    ax.set_ylabel(r'$\omega$ [deg/s]', fontsize=9.5)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.2, loc='upper left',
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.outdir / 'exp_fit_comparison.png',
                bbox_inches='tight', dpi=args.dpi)
    print(f'onset cosh {crit_c.onset_moment:+.3f} / pnls '
          f'{crit_f.onset_moment:+.3f} / static {m_th:+.3f} N·m; '
          f'threshold crossing t = {t_th:.3f} s, cosh onset '
          f't = {crit_c.onset_time:.3f} s')
    print(f'written to {args.outdir}/exp_fit_comparison.png')


if __name__ == '__main__':
    main()
