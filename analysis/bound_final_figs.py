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


def draw(rate, res, cap, title, cap_label, fname, exceed_label='exceeds',
         out=None):
    """cap is ONE value per ramp rate -- the worst case over the whole
    design box -- so the theory enters the figure as a single curve,
    not a per-run scatter. out: optional boolean mask marking runs
    OUTSIDE the design box (deliberate probes); they are drawn hollow
    grey and excluded from the residual mean line, since the guarantee
    is issued for the in-box population only."""
    bad = res > cap
    if out is None:
        out = np.zeros(len(res), dtype=bool)
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    rng = np.random.default_rng(0)
    ur = np.array(sorted(set(rate)))
    ix = {v: i for i, v in enumerate(ur)}
    xj = np.array([ix[v] for v in rate]) + rng.uniform(-0.16, 0.16,
                                                       len(rate))
    ax.plot(range(len(ur)), [cap[rate == v][0] for v in ur],
            '-o', lw=2.0, ms=4.5, color='#D55E00',
            label=cap_label, zorder=4)
    m_in = ~bad & ~out
    ax.plot(xj[m_in], res[m_in], 'o', ms=3.4, color='#0072B2', alpha=0.5,
            label='PNLS residual RMS (in-box)' if out.any()
            else 'PNLS residual RMS', zorder=3)
    if out.any():
        ax.plot(xj[out & ~bad], res[out & ~bad], 'o', ms=4.2, mfc='none',
                mew=1.2, color='0.45',
                label='out-of-box probes (S9, S11)', zorder=3)
    if bad.any():
        ax.plot(xj[bad], res[bad], 'o', ms=5.0, mfc='none', mew=1.5,
                color='#CC0000', label=exceed_label, zorder=5)
    ax.plot(range(len(ur)),
            [np.mean(res[(rate == v) & ~out]) for v in ur],
            '-', lw=1.9, color='#0072B2', zorder=4)
    ins = int(np.sum(~bad))
    ax.set_xticks(range(len(ur)))
    ax.set_xticklabels([f'{v:g}' for v in ur])
    ax.set_xlabel(r'ramp rate $\dot M$ [N·m/s]', fontsize=9.5)
    ax.set_ylabel('residual RMS [deg/s]', fontsize=9.5)
    ax.grid(alpha=0.4, lw=0.8, color='0.6')
    ax.set_axisbelow(True)
    if out.any():
        n_in, k_in = int((~out).sum()), int((~bad & ~out).sum())
        head = (f'{title}: {k_in}/{n_in} in-box inside '
                f'({ins}/{len(res)} overall)')
    else:
        head = f'{title}: {ins}/{len(res)} runs inside'
    ax.set_title(head, fontsize=10, loc='left')
    ax.legend(fontsize=8.4, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(f'docs/{fname}.{ext}', bbox_inches='tight', **kw)
    print(fname, f'{ins}/{len(res)} inside, worst usage '
          f'{np.max(res / cap):.2f}')


# simulation design box: leg arms per excitation axis, CoM height, and
# the +/-25 mm offset rectangle the identification is issued for
J_CAD = {'x': 0.051085, 'y': 0.050564}
SIM_L = {'Mx': (0.110, 'x'), 'My': (0.140, 'y')}
SIM_Z, SIM_BOX = 0.272, 0.025
SIM_PHI = np.deg2rad(5.0)
MASSES = (3.066, 3.220)


def _env(mdot, phi, W, z, jp, arm, ge):
    """Window RMS of the envelope, rho_bar*K*C2*sqrt(B(x)/x) [deg/s]."""
    c2 = np.sqrt(W * z / jp)
    k = 1.0 / (W * z)
    x = brentq(lambda v: np.sinh(v) - v - phi * W * z * c2 / mdot,
               1e-6, 40)
    B = 0.25 * np.sinh(2 * x) - 0.5 * x
    rb = R_PHI * 0.5 * W * arm * phi ** 2
    if ge:
        rb += R_GE * BETA_M * (mdot * x / c2) * phi
    return np.degrees(rb * k * c2 * np.sqrt(B / x))


def sim_capline(mdot):
    """Worst case over the whole design box: both leg arms, the offset
    rectangle's far corner, both campaign masses. One number per rate."""
    return max(_env(mdot, SIM_PHI, m * G, SIM_Z,
                    J_CAD[axis] + m * (SIM_Z ** 2 + lp ** 2),
                    lp + SIM_BOX, ge=False)
               for (lp, axis) in SIM_L.values() for m in MASSES)


def hw_capline(mdot):
    """Box-worst hardware envelope (10/8-deg design tilt, GE channel
    included), maximised over both axes and both masses."""
    return max(_env(mdot, PHI, m * G, Z, J1 + m * (Z ** 2 + LP[ax] ** 2),
                    ARM[ax], ge=True)
               for ax in ARM for m in MASSES)


def main():
    # -- simulation: box-worst envelope alone, one curve per rate -----
    rows = list(csv.DictReader(open('docs/sim_env_witness_runs.csv')))
    rate = np.array([float(r['rate']) for r in rows])
    res = np.array([float(r['res']) for r in rows])
    caps = {v: sim_capline(v) for v in sorted(set(rate))}
    cap = np.array([caps[v] for v in rate])
    # S9 (32,32) and S11 (38,14) mm sit beyond the design rectangle the
    # certificate is issued for -- deliberate out-of-box probes
    out = np.array([r['case'] in ('S9', 'S11') for r in rows])
    draw(rate, res, cap, 'simulation, design tilt cap 5°',
         r'box-worst envelope $\bar\rho K C_2\sqrt{B(x)/x}$ (theory)',
         'fig_bound_final_sim', out=out)

    # -- hardware: box-worst envelope + campaign vibration constant ---
    rows = list(csv.DictReader(open('docs/hw_env_noise_runs.csv')))
    rate = np.array([float(r['rate']) for r in rows])
    res = np.array([float(r['res']) for r in rows])
    # the one measured input: the campaign's largest out-of-band
    # amplitude, extended to the full band -- a single constant that
    # dominates every run's vibration term by construction
    n_camp = max(float(r['nhi']) for r in rows) * np.sqrt(1 + KB ** 2)
    caps = {v: hw_capline(v) + n_camp for v in sorted(set(rate))}
    cap = np.array([caps[v] for v in rate])
    draw(rate, res, cap, 'hardware, design tilt cap 8°',
         r'box-worst envelope + $\max n_{hi}\sqrt{1+\kappa_b^2}$'
         ' (theory + measured noise)',
         'fig_bound_final_hw')


if __name__ == '__main__':
    main()
