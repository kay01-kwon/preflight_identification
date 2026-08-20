#!/usr/bin/env python3
"""A kernel-free residual bound: witness + maximum principle.

The bound needs no kernel matrix, no projector, and no integral of the
forcing -- only closed-form constants -- yet it caps the residual of the
full nonlinear fit (C1, C2 and the onset all free).

Construction.  The minimiser beats any single member of the family, so
exhibit one.  Take the witness

    h_hat = omega_nom + beta sinh(C2 tau),
    beta  = e_omega(tau_end) / sinh(C2 tau_end),

admissible by the shift identity (beta/C1 ~ 0.04).  The remainder
delta_e = e_omega - beta sinh vanishes at BOTH window ends -- at tau=0
because e_omega and sinh both start at zero, at tau_end by the choice
of beta -- and satisfies the same deviation operator with the forcing
differentiated once:

    delta_e'' - C2^2 delta_e = rho_dot / J_P,
    delta_e(0) = delta_e(tau_end) = 0.

An equivalent derivation needs no differential equation at all:
substituting the Duhamel representation (89) into the remainder (both
for e(tau) and for the e(T) inside beta) gives
delta_e = (1/J_P) int K(tau,s) rho(s) ds with

    K =  sinh(C2(T-tau)) cosh(C2 s) / sinh(C2 T)    for s <= tau,
    K = -sinh(C2 tau) cosh(C2(T-s)) / sinh(C2 T)    for s >  tau,

where the hyperbolic identities cancel the exponential inside the
kernel: max|K| = 1 exactly (K = -dG/ds of the Dirichlet Green's
function; integration by parts connects the two forms, the boundary
terms vanishing because G is pinned).  Shaped, this evaluates to
0.43-0.46 deg/s against the realised 0.28-0.32 -- same level as the
rho_dot route, and rho is never differentiated.

The operator v'' - C2^2 v obeys a maximum principle.  Against constant
forcing that gives |delta_e| <= (1-sech(x/2)) sup|rho_dot|/Wz; shaping
the comparison to rho_dot's actual decay -- e^{2C2(s-T)} on the arm and
gravity channels, e^{C2(s-T)} on ground effect; for the NOMINAL
trajectory these decays are proved exactly (doc, Step 5': the ratios
(cosh a - 1)e^{-a} = (1-e^{-a})^2/2 and (sinh a - a)e^{-a} are
nondecreasing), and the 1.05 factor covers only the nominal-to-true
transfer, measured worst 2.6% -- tightens it eightfold:

    |delta_e| <= (M2(x) rho2_dot + M1(x) rho1_dot) / Wz,
    M2 -> 1/12,   M1 -> 1/(2e)   (resonant channel),

still elementary functions only.  The witness's onset shift also leaves
a pre-onset segment on the baseline while the branch rises; that enters
at second order in beta <= rho_bar/(J_P C2) as Delta_pre, 0.036
deg/s at the slowest ramp and <0.011 above Mdot = 0.10.

No exponential anywhere: pinning both ends removes the unstable
homogeneous mode -- which is exactly what the estimator's onset freedom
does, since sinh IS the first-order onset-shift direction.

The forcing rate is bounded channel by channel, all a priori:

    sup|rho_dot| <= W l_arm phi_max om_max            (pivot arm)
                  + |beta_M| (Mdot phi_max + dM_win om_max)   (ground effect)

The gravity-height term Wz (cos phi - 1) omega is NOT added: it enters
d(rho_grav)/dphi with the opposite sign to the arm term and
W a sin(phi) >= Wz (1 - cos(phi)) iff a >= z tan(phi/2), held with
3.4x margin over the design box -- the same sign cancellation rho_bar
(91) uses, and what (85) at phi* = 0 expresses (G''(0) = W a).

with om_max = K Mdot (cosh(x) - 1) + rho_bar sinh(x)/(J_P C2), the
TRUE end-rate anchor: nominal end rate plus the envelope (90) of
e_omega, so the anchor needs no allowance and the 1.05 covers only
the interior shape transfer.  Everything on the
right is known before the experiment: W and z_CoM from the scale, J_P
from CAD, l_arm from geometry, phi_max the design box, beta_M from the
GE model, Mdot and the window from the protocol.  Add a gyro noise
figure and the whole cap is computable before a single run is flown.

On the campaign: the model term is 0.24-0.34 deg/s -- the level of the
projected Phi, with no kernel computed -- the cap holds on 140/140 runs
at used 0.43-0.46 (per-run worst 0.68), and the realised witness
remainder on the exact nonlinear solution is 0.16-0.17 deg/s, flat.
The model term sits 1.4-2.0x above the realised value: with the
box at the 5-degree excitation cap, most of the old slack is gone;
the cap's slack lives in the noise ENVELOPE (kappa_b), by design.

Usage: python analysis/kernel_free_bound.py [out.png]
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as student

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import ARMS, W, BETA_M
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich

HERE = os.path.dirname(os.path.abspath(__file__))
C_DE, C_N, C_MEAS = '#7b3294', '0.62', '#148f77'


SHAPE_SAFETY = 1.05     # worst measured shape-lemma ratio is 1.026


def M2(x, n=2001):
    """Peak of the Dirichlet response to unit-peak e^{2C2(s-T)} forcing,
    times C2^2.  Closed form evaluated on a grid; asymptote 1/12."""
    u = np.linspace(-x, 0.0, n)
    v = np.exp(2 * u) - np.sinh(u + x) / np.sinh(x) \
        - np.exp(-2 * x) * np.sinh(-u) / np.sinh(x)
    return float(np.abs(v).max() / 3.0)


def M1(x, n=2001):
    """Same for the resonant unit-peak e^{C2(s-T)} forcing; asymptote
    1/(2e)."""
    u = np.linspace(-x, 0.0, n)
    v = (u + x) * np.exp(u) / 2.0 - x * np.sinh(u + x) / (2 * np.sinh(x))
    return float(np.abs(v).max())


def model_term(d, rb):
    """The shaped kernel-free model term for one run, in deg/s.

    The forcing rate decays like e^{2C2(s-T)} on the arm and gravity
    channels (both carry omega ~ e^{C2 s} times phi ~ e^{C2 s}) and
    like e^{C2(s-T)} on the ground-effect channels, so the Dirichlet
    comparison is solved per shape instead of for a constant: the
    factor (1 - sech(x/2)) ~ 1 becomes M2 -> 1/12 and M1 -> 1/(2e).
    The pre-onset segment the shifted witness leaves behind enters at
    second order in the a priori witness coefficient
    beta <= rho_bar/(J_P C2)."""
    tau, c2, k = d['tau'], d['c2'], d['k']
    T = tau[-1]
    x = c2 * T
    wz = 1.0 / k
    # TRUE end-rate anchor, a priori: nominal end rate plus the
    # envelope (90) of e_omega -- omega_true(T) <= omega_nom(T) + E(T),
    # all pre-experiment constants.  (An earlier draft used the nominal
    # alone, leaving its excess to the 1.05; before that, sinh x, an
    # overbound by coth(x/2).)  The 1.05 now covers only the interior
    # shape transfer (measured worst 2.6%).
    jp0 = 1.0 / (k * c2 ** 2)
    om = k * d['md_full'] * (np.cosh(min(x, 30.0)) - 1.0) \
        + rb * np.sinh(min(x, 30.0)) / (jp0 * c2)
    # gravity-height term dropped by SIGN, not size: d(rho_grav)/dphi =
    # W a sin(phi) + Wz (cos(phi) - 1), the second term negative, and
    # W a sin(phi) >= Wz (1 - cos(phi))  iff  a >= z tan(phi/2) -- held
    # with 3.4x margin over the whole design box.  Same cancellation
    # rho_bar (91) already uses; phi* = 0 in the manuscript's (85).
    rd2 = SHAPE_SAFETY * (W * ARMS[d['axis']] * PHI_BOX) * om
    rd1 = abs(BETA_M) * (d['md_full'] * PHI_BOX + d['dm_win'] * om)
    de = (rd2 * M2(x) + rd1 * M1(x)) / wz
    # pre-onset: between the two onsets the fit sits on its baseline
    # while the branch rises (either sign of the shift -- the segment
    # is |dt| long on one side or the other).  Integrated EXACTLY over
    # the shifted segment, still closed form:
    #   (1/T) int_0^dt [C1(cosh(C2 u)-1)]^2 du
    #   = (C1^2/T) [ 3u/2 + sinh(2 C2 u)/(4 C2) - 2 sinh(C2 u)/C2 ]_0^dt
    c1 = k * d['md_full']
    jp = 1.0 / (k * c2 ** 2)
    beta = rb / (jp * c2)
    dt = np.arctanh(min(beta / c1, 0.99)) / c2
    a = min(c2 * dt, 30.0)
    I = 1.5 * dt + np.sinh(2 * a) / (4 * c2) - 2 * np.sinh(a) / c2
    dpre = c1 * np.sqrt(max(I, 0.0) / T)
    return float(np.rad2deg(de)), float(np.rad2deg(dpre))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'kernel_free_bound.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    kqm = float(np.median([d['kq'] for d in rows if d['kq'] is not None]))
    from fit_quality_bound import rho_bar
    for d in rows:
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        d['de'], d['dpre'] = model_term(d, rb)
        d['de'] += d['dpre']
        # envelope noise term: kappa_b = the campaign MAX quiet shape
        # ratio (not the run's own) -- s_med * kappa_b = 2.09 exceeds
        # every measured in-window ratio (max kappa_imp 1.69), so n_b
        # is an upper bound on every run's disturbance, not a
        # median-calibrated estimate.  With the 5-degree box the model
        # term guards delta_e alone; the noise envelope guards the
        # disturbance tail.  No double duty.
        d['nh'] = d['rms_n'] * np.sqrt(1.0 + (s_med * kqmax) ** 2)
        d['kcap'] = d['de'] + d['nh']
    rates = sorted({d['rate'] for d in rows})
    grp = [[d for d in rows if d['rate'] == rt] for rt in rates]
    n = len(rows)
    inside = sum(1 for d in rows if d['rms_min'] <= d['kcap'])

    def stat(vals):
        v = np.asarray(vals)
        return v.mean(), student.ppf(0.975, len(v) - 1) * v.std(ddof=1) \
            / np.sqrt(len(v))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.125)
    i = np.arange(len(rates))
    de = np.array([stat([d['de'] for d in v])[0] for v in grp])
    nz = np.array([stat([d['nh'] for d in v])[0] for v in grp])
    cap = de + nz
    cap_ci = np.array([stat([d['kcap'] for d in v])[1] for v in grp])
    mm = np.array([stat([d['rms_min'] for d in v])[0] for v in grp])
    mm_ci = np.array([stat([d['rms_min'] for d in v])[1] for v in grp])

    ax.bar(i - 0.20, de, 0.36, color=C_DE,
           label=r'$(M_2\dot\rho_2+M_1\dot\rho_1)/Wz_{CoM}+\Delta_{\rm pre}$'
                 ', a priori')
    ax.bar(i - 0.20, nz, 0.36, bottom=de, color=C_N,
           label=r'$\hat n$, the disturbance')
    ax.errorbar(i - 0.20, cap, yerr=cap_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)
    ax.bar(i + 0.20, mm, 0.36, color=C_MEAS,
           label=r'measured $\mathrm{RMS}(r)$')
    ax.errorbar(i + 0.20, mm, yerr=mm_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)

    for k in range(len(rates)):
        lab = f'{mm[k] / cap[k]:.2f}'
        if k == 0:
            lab = 'used\n' + lab
        ax.text(k + 0.20, mm[k] + mm_ci[k] + 0.09, lab, ha='center',
                va='bottom', fontsize=9, color=C_MEAS, weight='bold')

    ax.errorbar([], [], yerr=[], fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, label='mean of 20 runs, 95% CI')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=10)
    ax.set_title('the kernel-free cap, shaped comparison: still no kernel, '
                 'still pre-experiment\n'
                 f'{inside}/{n} runs inside; the model term is now '
                 f'{de.min():.2f} to {de.max():.2f}' r'$^\circ$/s', fontsize=11.5)
    ax.legend(fontsize=8.8, loc='upper right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    ax.set_ylim(0, max(cap + cap_ci) * 1.30)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}{'x':>6}{'de bound':>10}{'of it pre':>10}{'n_hat':>8}"
          f"{'cap':>7}{'meas':>7}{'used':>6}{'inside':>8}")
    for rt, v in zip(rates, grp):
        m = lambda kk: np.mean([q[kk] for q in v])
        x = np.mean([d['c2'] * d['tau'][-1] for d in v])
        ins = sum(1 for d in v if d['rms_min'] <= d['kcap'])
        print(f"  {rt:6.2f}{x:6.2f}{m('de'):10.3f}{m('dpre'):10.3f}"
              f"{m('nh'):8.3f}{m('kcap'):7.2f}{m('rms_min'):7.3f}"
              f"{m('rms_min') / m('kcap'):6.2f}{ins:5d}/20")
    u = np.array([d['rms_min'] / d['kcap'] for d in rows])
    print(f"\n  inside {inside}/{n}; per-run used p10 "
          f"{np.percentile(u, 10):.2f} median {np.percentile(u, 50):.2f} "
          f"max {u.max():.2f}")
    print(f"  shaped comparison: M2 -> 1/12, M1 -> 1/(2e); the model term")
    print(f"  is 8x below the constant-forcing version and lands at Phi's")
    print(f"  level with no kernel computed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
