#!/usr/bin/env python3
"""The (20) check in the side-by-side bar style: stacked bound bar
(model term + method-blind SG disturbance constant) against the
measured minimiser RMS, per rate, with 95% intervals and per-rate
used fractions.

The disturbance term is one campaign constant, N_n = 3 N_med =
2.31 deg/s: three times the campaign median of the Savitzky-Golay
high-band anchor.  The factor 3 is declared (1.5 level spread x 1.96
shape ceiling, rounded up); the noise term never touches the fit.

Usage: python analysis/sg_bound_bars.py [out.png]
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
FACTOR = 3.0
C_M, C_N, C_MEAS = '#1a5276', '0.72', '#148f77'


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'sg_bound_bars.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
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
    n_med = float(np.rad2deg(np.median(sgs)))
    noise = FACTOR * n_med
    for d in rows:
        d['cap'] = d['m'] + noise
    rates = sorted({d['rate'] for d in rows})
    grp = [[d for d in rows if d['rate'] == r] for r in rates]
    i = np.arange(len(rates))

    def stat(key, v):
        a = np.array([d[key] for d in v])
        return a.mean(), student.ppf(0.975, len(a) - 1) \
            * a.std(ddof=1) / np.sqrt(len(a))

    mmod = [stat('m', v)[0] for v in grp]
    capm = [stat('cap', v) for v in grp]
    meas = [stat('rms_min', v) for v in grp]
    ins = sum(1 for d in rows if d['rms_min'] <= d['cap'])
    used_all = [d['rms_min'] / d['cap'] for d in rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    w = 0.38
    ax.bar(i - 0.21, mmod, w, color=C_M,
           label='model term (17)+(18)')
    ax.bar(i - 0.21, [noise] * len(i), w, bottom=mmod, color=C_N,
           label=(r'$N_n = 3N_{\rm med} = %.2f^\circ$/s '
                  '(SG filter, declared factor 3)' % noise))
    ax.errorbar(i - 0.21, [c[0] for c in capm],
                yerr=[c[1] for c in capm], fmt='none', ecolor='k',
                capsize=4, lw=1.4)
    ax.bar(i + 0.21, [m[0] for m in meas], w, color=C_MEAS,
           label=r'measured $\mathrm{RMS}(r)$')
    ax.errorbar(i + 0.21, [m[0] for m in meas],
                yerr=[m[1] for m in meas], fmt='none', ecolor='k',
                capsize=4, lw=1.4,
                label='mean of 20 runs, 95% CI')
    for j, v in enumerate(grp):
        u = np.mean([d['rms_min'] / d['cap'] for d in v])
        ax.text(j + 0.21, meas[j][0] + meas[j][1] + 0.06, f'{u:.2f}',
                ha='center', fontsize=10, color=C_MEAS,
                fontweight='bold')
    ax.text(len(rates) - 0.55, max(m[0] for m in meas) + 0.62,
            'fraction of the bound used', color=C_MEAS, fontsize=10,
            ha='right')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=10)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=11)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=11)
    ax.set_ylim(0, (max(mmod) + noise) * 1.42)
    ax.set_title(
        r'$\mathrm{RMS}(r) \leq (M_2\dot\rho_2 + M_1\dot\rho_1)/Wz_{CoM}'
        r' + \Delta_{\rm pre} + N_n$ on all %d runs' % ins + '\n'
        'the noise term never touches the fit; '
        f'used {min(np.mean([d["rms_min"]/d["cap"] for d in v]) for v in grp):.2f} '
        f'to {max(np.mean([d["rms_min"]/d["cap"] for d in v]) for v in grp):.2f}, '
        f'worst run {max(used_all):.2f}', fontsize=12)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}; noise term {noise:.2f}, inside {ins}/140, "
          f"worst used {max(used_all):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
