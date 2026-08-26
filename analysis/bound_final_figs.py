#!/usr/bin/env python3
"""The final bound-validation figure pair, simulation and hardware.

One construction on both campaigns: the free-fit PNLS residual RMS of
every run against the small-angle envelope-witness cap

    RMS(e_omega) <= rho_bar * K * C2 * sqrt(B(x)/x),
    B(x) = sinh(2x)/4 - x/2,

evaluated at the design tilt cap: 5 deg in simulation, 8 deg on
hardware, each chosen to dominate the realized in-window tilt of its
campaign (the gate stops the simulator near 4 deg; the slightly
uneven ground of the test site lets the vehicle reach 7.7 deg).  The
simulation panel carries the envelope alone -- no noise term of any
kind.  Hardware adds the channel the vehicle contributes twice over:
the ground-effect term inside rho_bar and the measured vibration term
n_hi * sqrt(1 + kappa_b^2) outside it.

Inputs are the per-run sweeps already on disk:
  docs/sim_env_witness_runs.csv   (cap column includes the declared
                                   gyro sigma; subtracted here)
  docs/hw_env_noise_runs.csv      (free-fit residual + hi-band noise)

Usage
-----
  python analysis/bound_final_figs.py
"""
import csv

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

G = 9.81
PHI = np.deg2rad(8.0)                # hardware design tilt cap
SIM_SIGMA = 2.45e-4                  # declared gyro noise std [rad/s]
Z, BETA_M = 0.30, 0.0345             # hardware design box
R_PHI, R_GE = 1 / 7, 1 / 5
J1, KB = 0.0537, 1.31
MASS = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
        'case_04': 3.220, 'case_05': 3.220}
ARM = {'Mx': 0.160, 'My': 0.130}
LP = {'Mx': 0.140, 'My': 0.110}


def draw(rate, res, cap, title, cap_label, fname, exceed_label='exceeds'):
    bad = res > cap
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    rng = np.random.default_rng(0)
    ur = np.array(sorted(set(rate)))
    ix = {v: i for i, v in enumerate(ur)}
    xj = np.array([ix[v] for v in rate]) + rng.uniform(-0.16, 0.16,
                                                       len(rate))
    ax.plot(xj, cap, 'o', ms=3.4, color='#D55E00', alpha=0.45,
            label=cap_label, zorder=2)
    ax.plot(xj[~bad], res[~bad], 'o', ms=3.4, color='#0072B2', alpha=0.5,
            label='PNLS residual RMS', zorder=3)
    if bad.any():
        ax.plot(xj[bad], res[bad], 'o', ms=5.0, mfc='none', mew=1.5,
                color='#CC0000', label=exceed_label, zorder=5)
    for arr, c in ((cap, '#D55E00'), (res, '#0072B2')):
        ax.plot(range(len(ur)), [np.mean(arr[rate == v]) for v in ur],
                '-', lw=1.9, color=c, zorder=4)
    ins = int(np.sum(~bad))
    ax.set_xticks(range(len(ur)))
    ax.set_xticklabels([f'{v:g}' for v in ur])
    ax.set_xlabel(r'ramp rate $\dot M$ [N·m/s]', fontsize=9.5)
    ax.set_ylabel('residual RMS [deg/s]', fontsize=9.5)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    ax.set_title(f'{title}: {ins}/{len(res)} runs inside',
                 fontsize=10, loc='left')
    ax.legend(fontsize=8.4, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(f'docs/{fname}.{ext}', bbox_inches='tight', **kw)
    print(fname, f'{ins}/{len(res)} inside, worst usage '
          f'{np.max(res / cap):.2f}')


def main():
    # -- simulation: envelope only, no noise anywhere -----------------
    rows = list(csv.DictReader(open('docs/sim_env_witness_runs.csv')))
    rate = np.array([float(r['rate']) for r in rows])
    res = np.array([float(r['res']) for r in rows])
    cap = (np.array([float(r['cap']) for r in rows])
           - np.degrees(SIM_SIGMA))                     # pure envelope
    draw(rate, res, cap, 'simulation, design tilt cap 5°',
         r'small-angle envelope $\bar\rho K C_2\sqrt{B(x)/x}$ (theory)',
         'fig_bound_final_sim')

    # -- hardware: envelope (8 deg, GE incl.) + noise term ------------
    rows = list(csv.DictReader(open('docs/hw_env_noise_runs.csv')))
    out = []
    for r in rows:
        m = MASS[r['case']]
        W = m * G
        ax_ = r['ax']
        md = float(r['rate'])
        jp = J1 + m * (Z ** 2 + LP[ax_] ** 2)
        c2 = np.sqrt(W * Z / jp)
        k = 1.0 / (W * Z)
        x = brentq(lambda v: np.sinh(v) - v - PHI * W * Z * c2 / md,
                   1e-3, 40)
        dmw = md * x / c2
        rb = (R_PHI * 0.5 * W * ARM[ax_] * PHI ** 2
              + R_GE * BETA_M * dmw * PHI)
        B = 0.25 * np.sinh(2 * x) - 0.5 * x
        env = np.degrees(rb * k * c2 * np.sqrt(B / x))
        out.append((md, float(r['res']),
                    env + float(r['nhi']) * np.sqrt(1 + KB ** 2)))
    rate = np.array([o[0] for o in out])
    res = np.array([o[1] for o in out])
    cap = np.array([o[2] for o in out])
    draw(rate, res, cap, 'hardware, design tilt cap 8°',
         r'envelope + $n_{hi}\sqrt{1+\kappa_b^2}$ (theory + measured noise)',
         'fig_bound_final_hw')


if __name__ == '__main__':
    main()
