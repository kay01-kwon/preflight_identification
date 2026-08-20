#!/usr/bin/env python3
"""The residual cap with the method-blind SG campaign constant.

Same check as Fig. 2 (eq. (20) against the campaign), but the noise
term is the single Savitzky-Golay campaign maximum N* = 1.81 deg/s
times sqrt(1 + kappa_sup^2) -- one number for every run, no model
curve anywhere in the noise side.  Coverage stays 140/140 at used
0.25-0.28: the price of full method independence is a factor ~2 in
slack against the deployed per-run anchor.

Usage: python analysis/sg_bound_figure.py [out.png]
"""
import os
import pickle
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import t as student

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failing_runs import split, FC
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from fit_quality_bound import rho_bar
from kernel_free_bound import model_term

HERE = os.path.dirname(os.path.abspath(__file__))
C_DE, C_N, C_MEAS = '#7b3294', '0.62', '#148f77'


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'sg_bound.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    ksup = max(d['kimp'] for d in rows)
    mult = np.sqrt(1 + ksup ** 2)
    hi = lambda v: float(np.sqrt(np.mean(v ** 2)))
    sgs = []
    for d in rows:
        om = np.asarray(d['om'], float)
        dt, N = d['dt'], len(d['om'])
        win = max(int(round(2.0 / (FC * dt))) | 1, 7)
        win = min(win, N - 1 if (N - 1) % 2 else N - 2)
        sgs.append(hi(split(om - savgol_filter(om, win, 3), dt)[1]))
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        de, dpre = model_term(d, rb)
        d['m'] = de + dpre
    Nstar = float(np.rad2deg(max(sgs)))
    noise = Nstar * mult
    for d in rows:
        d['cap'] = d['m'] + noise
    rates = sorted({d['rate'] for d in rows})
    grp = [[d for d in rows if d['rate'] == r] for r in rates]
    i = np.arange(len(rates))
    mmod = [float(np.mean([d['m'] for d in v])) for v in grp]
    meas = [float(np.mean([d['rms_min'] for d in v])) for v in grp]
    ci = [student.ppf(0.975, len(v) - 1)
          * np.std([d['rms_min'] for d in v], ddof=1) / np.sqrt(len(v))
          for v in grp]
    ins = sum(1 for d in rows if d['rms_min'] <= d['cap'])
    used = [d['rms_min'] / d['cap'] for d in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(i, mmod, 0.62, color=C_DE,
           label='model term (17)+(18), per run')
    ax.bar(i, [noise] * len(i), 0.62, bottom=mmod, color=C_N,
           label=(r'$N^\ast\sqrt{1+\kappa_{\sup}^2} = %.2f^\circ$/s, '
                  'one campaign constant' % noise))
    ax.errorbar(i, meas, yerr=ci, fmt='o', color=C_MEAS, ms=7,
                capsize=4, lw=1.8,
                label=r'measured $\mathrm{RMS}(r)$, mean $\pm$ 95% CI')
    for j, v in enumerate(grp):
        u = np.mean([d['rms_min'] / d['cap'] for d in v])
        ax.text(j, mmod[j] + noise + 0.06, f'used {u:.2f}',
                ha='center', fontsize=8.5)
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=10)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=11)
    ax.set_ylabel(r'[$^\circ$/s]', fontsize=11)
    ax.set_ylim(0, (max(mmod) + noise) * 1.16)
    ax.set_title('Equation (20) with the method-blind noise constant: '
                 f'$N^\\ast = {Nstar:.2f}^\\circ$/s\n'
                 '(campaign max of the Savitzky--Golay anchor) --- '
                 f'inside {ins}/140, worst used {max(used):.2f}',
                 fontsize=11.5)
    ax.legend(fontsize=9.5, loc='center right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}; N* = {Nstar:.2f}, noise term {noise:.2f}, "
          f"inside {ins}/140, worst used {max(used):.2f}, "
          f"mean used {np.mean(used):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
