#!/usr/bin/env python3
"""Design-box sweep: the pre-experiment RMS model bound and the
critical-moment shift, as ranges over the vehicle parameter box.

Box (from the design spec):
    mass        3.000 - 3.220  kg
    l_p         0.110 - 0.140  m
    p_off      -0.020 - 0.020  m   (CoM offset; arm = l_p - p_off)
    z_CoM       0.20  - 0.30   m
    J_CoM       0.050          kg m^2  (fixed)
    Mdot        the seven protocol rates
Held fixed: phi_max = 10 deg (design tilt box), beta_M = -0.03446 (GE
model), shape safety 1.05.  The window of each combination is the
tilt-limited one: x solves sinh(x) - x = phi_max Wz C2 / Mdot.

For every combination the script evaluates, all closed form:
  * the model term of the residual cap, (17) + (18):
        (M2 rho2_dot + M1 rho1_dot)/Wz + Delta_pre        [deg/s]
  * the critical-moment shift of (21):
        |dM_crit| = Mdot |dt_c|,  dt_c from (9) with e_w <= E,
    together with its ceiling rho_bar                      [mN m]

and draws both as min-median-max bands against Mdot.

Usage: python analysis/design_range_chart.py [out.png]
"""
import sys

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = 9.81
J_COM = 0.050
BETA_M = 0.03446
PHI = np.deg2rad(10.0)
SAFETY = 1.05
RATES = np.array([0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20])
MASS = np.linspace(3.000, 3.220, 5)
LP = np.linspace(0.110, 0.140, 4)
POFF = np.linspace(-0.020, 0.020, 5)
ZCOM = np.linspace(0.20, 0.30, 6)
C_A, C_B, C_CEIL = '#7b3294', '#2874a6', '#e08214'


def M2(x, n=1501):
    u = np.linspace(-x, 0.0, n)
    v = (np.sinh(u + x) + np.exp(-2 * x) * np.sinh(-u)) / np.sinh(x) \
        - np.exp(2 * u)
    return float(np.abs(v).max() / 3.0)


def M1(x, n=1501):
    u = np.linspace(-x, 0.0, n)
    v = (x / 2.0) * np.sinh(u + x) / np.sinh(x) - (u + x) * np.exp(u) / 2.0
    return float(np.abs(v).max())


def combo(m, lp, poff, z, md):
    W = m * G
    arm = lp - poff
    wz = W * z
    jp = J_COM + m * (z * z + arm * arm)
    c2 = np.sqrt(wz / jp)
    # tilt-limited window: phi_nom(tau_end) = PHI
    rhs = PHI * wz * c2 / md
    x = brentq(lambda q: np.sinh(q) - q - rhs, 0.05, 25.0)
    T = x / c2
    dmw = md * T
    om = md * np.sinh(x) / wz
    # Wz term dropped by sign: d(rho_grav)/dphi = W a sin(phi)
    # + Wz (cos(phi) - 1) with the second term negative, and
    # a >= z tan(phi/2) everywhere in the box (min 0.090 vs max 0.026).
    rd2 = SAFETY * (W * arm * PHI) * om
    rd1 = BETA_M * (md * PHI + dmw * om)
    model = (M2(x) * rd2 + M1(x) * rd1) / wz
    rb = (1.0 / 7.0) * 0.5 * W * arm * PHI * PHI \
        + (1.0 / 5.0) * BETA_M * dmw * PHI
    c1 = md / wz
    beta = rb / (jp * c2)
    dt = np.arctanh(min(beta / c1, 0.99)) / c2
    a = c2 * dt
    I = 1.5 * dt + np.sinh(2 * a) / (4 * c2) - 2 * np.sinh(a) / c2
    dpre = c1 * np.sqrt(max(I, 0.0) / T)
    return (np.rad2deg(model + dpre),      # RMS model bound  [deg/s]
            1e3 * md * dt,                 # dM_crit shift    [mN m]
            1e3 * rb,                      # ceiling rho_bar  [mN m]
            x, 1e3 * dt)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'design_range.png'
    rms = {r: [] for r in RATES}
    dmc = {r: [] for r in RATES}
    ceil = {r: [] for r in RATES}
    xs = {r: [] for r in RATES}
    dts = {r: [] for r in RATES}
    for m in MASS:
        for lp in LP:
            for poff in POFF:
                for z in ZCOM:
                    for r in RATES:
                        b, d, c, x, dt = combo(m, lp, poff, z, r)
                        rms[r].append(b)
                        dmc[r].append(d)
                        ceil[r].append(c)
                        xs[r].append(x)
                        dts[r].append(dt)

    lo = lambda q: np.array([np.min(q[r]) for r in RATES])
    hi = lambda q: np.array([np.max(q[r]) for r in RATES])
    md_ = lambda q: np.array([np.median(q[r]) for r in RATES])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.845, bottom=0.13,
                        wspace=0.22)

    a1.fill_between(RATES, lo(rms), hi(rms), color=C_A, alpha=0.18, lw=0)
    a1.plot(RATES, md_(rms), 'o-', color=C_A, lw=2.2, ms=6)
    a1.plot(RATES, lo(rms), '-', color=C_A, lw=0.9, alpha=0.55)
    a1.plot(RATES, hi(rms), '-', color=C_A, lw=0.9, alpha=0.55)
    a1.text(RATES[-1], hi(rms)[-1] + 0.02,
            'box max: arm 0.160 m, $z$ 0.20 m', ha='right',
            fontsize=8.5, color=C_A)
    a1.text(RATES[-1], lo(rms)[-1] - 0.06,
            'box min: arm 0.090 m, $z$ 0.30 m', ha='right',
            fontsize=8.5, color=C_A)
    a1.set_xscale('log')
    a1.set_xticks(RATES)
    a1.set_xticklabels([f'{r:.2f}' for r in RATES], fontsize=9)
    a1.minorticks_off()
    a1.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    a1.set_ylabel(r'bound on $\mathrm{RMS}(\delta e_\omega)$ [$^\circ$/s]',
                  fontsize=10)
    a1.set_ylim(0, hi(rms).max() * 1.18)
    a1.set_title(r'(a) bound on $\mathrm{RMS}(\delta e_\omega)$, the model'
                 ' term of (20)\n(17)$+$(18) over the design box;'
                 ' the cap adds $\hat n$ on top', fontsize=11)
    a1.grid(alpha=0.22, lw=0.4)

    a2.fill_between(RATES, lo(dmc), hi(dmc), color=C_B, alpha=0.18, lw=0)
    a2.plot(RATES, md_(dmc), 'o-', color=C_B, lw=2.2, ms=6,
            label=r'shift $\dot M\,|\delta t_c|$, eq. (21)')
    a2.plot(RATES, lo(dmc), '-', color=C_B, lw=0.9, alpha=0.55)
    a2.plot(RATES, hi(dmc), '-', color=C_B, lw=0.9, alpha=0.55)
    a2.plot(RATES, lo(ceil), '--', color=C_CEIL, lw=1.6)
    a2.plot(RATES, hi(ceil), '--', color=C_CEIL, lw=1.6,
            label=r'ceiling $\bar\rho$ (box min/max)')
    a2.text(RATES[-1], hi(dmc)[-1] - 0.55,
            'box max: $m$ 3.22 kg, arm 0.160 m, $z$ 0.30 m', ha='right',
            va='top', fontsize=8.5, color=C_B)
    a2.text(RATES[0], lo(dmc)[0] - 0.45,
            'box min: $m$ 3.00 kg, arm 0.090 m, $z$ 0.20 m', ha='left',
            va='top', fontsize=8.5, color=C_B)
    a2.set_xscale('log')
    a2.set_xticks(RATES)
    a2.set_xticklabels([f'{r:.2f}' for r in RATES], fontsize=9)
    a2.minorticks_off()
    a2.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    a2.set_ylabel(r'$|\Delta M_{\rm crit}|$ [mN m]', fontsize=10)
    a2.set_ylim(0, hi(ceil).max() * 1.18)
    a2.set_title('(b) critical-moment shift from the absorbed onset '
                 'shift\nvs its ceiling $\\bar\\rho$; exact artanh form '
                 'exceeds it $\\leq$12% at the slow corner', fontsize=11)
    a2.legend(fontsize=9, loc='center right')
    a2.grid(alpha=0.22, lw=0.4)

    n = len(MASS) * len(LP) * len(POFF) * len(ZCOM)
    fig.suptitle(f'Design box sweep ({n} parameter combinations '
                 r'$\times$ 7 rates): mass 3.00-3.22 kg, '
                 r'$l_p$ 0.110-0.140 m, $p_{\rm off}$ $\pm$0.020 m, '
                 r'$z_{CoM}$ 0.20-0.30 m', fontsize=11.5, y=0.975)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}   ({n} combos per rate)\n")
    print(f"  {'Mdot':>6}{'x range':>13}{'RMS bound [deg/s]':>22}"
          f"{'dM_crit [mN m]':>20}{'rho_bar':>14}{'dt_c [ms]':>13}")
    print(f"  {'':6}{'':13}{'min   med   max':>22}"
          f"{'min   med   max':>20}{'min - max':>14}{'min - max':>13}")
    for r in RATES:
        print(f"  {r:6.2f}{np.min(xs[r]):6.2f}-{np.max(xs[r]):5.2f}"
              f"{np.min(rms[r]):8.2f}{np.median(rms[r]):6.2f}"
              f"{np.max(rms[r]):6.2f}"
              f"{np.min(dmc[r]):8.2f}{np.median(dmc[r]):6.2f}"
              f"{np.max(dmc[r]):6.2f}"
              f"{np.min(ceil[r]):7.1f}-{np.max(ceil[r]):5.1f}"
              f"{np.min(dts[r]):7.1f}-{np.max(dts[r]):5.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
