#!/usr/bin/env python3
"""What band-limited denoising does to the dynamic ground-effect check.

The Savitzky-Golay window of the omega_dot differentiator is swept from
the deployed 9 samples toward the band-limit rule of the noise model
(docs/noise_model_notes.tex, w ~ 2/(f_c T_s), i.e. 41 samples for
f_c = 5 Hz at T_s = 9.9 ms), and the three results are drawn on one
pair of axes.

Widening the window at the DEPLOYED order steepens the anomaly
(-42 -> -53 -> -90 mN.m/deg at 9/21/41 samples) -- but that is a
filter artefact, not physics, and the fix is the polynomial ORDER
rather than the window length.  omega_dot ~ sinh(C2 tau) with an
e-folding time 1/C2 = 163 ms, so a 41-sample (405 ms) window spans
2.5 e-foldings, far more curvature than the deployed local parabola
can follow (rate_derivative.omega_dot defaults to poly=2, deriv=1:
the analytic slope of a parabola fitted to the raw omega, so the raw
difference is never formed).  The parabola clips the growth of
J_P omega_dot, which is exactly the term that has to cancel
-W z_CoM sin(phi), and the residual slope opens up.

Raising the order over the SAME window removes the distortion
completely.  On a noise-free cosh with this run's constants, the
endpoint error of the derivative is -6.4% at (41, 3) and -0.2% at
(41, 5); the deployed (9, 2) is faithful only because its window is
short, at the cost of the worst noise gain of the set (1.89 rad/s^2
RMS at the endpoint for a 1 deg/s disturbance, against 0.69 at
(41, 5) -- 85% of the peak signal versus 31%).

With the order matched, the anomaly becomes INVARIANT to the
differentiator: -37 to -43 mN.m/deg across 90-510 ms windows, and
-23 to -30 for the windowless anchored polynomial of
rate_derivative.omega_dot_poly, against the model's -3.1.  That is a
stronger negative result than the original, which could still be
read as an artefact of the differentiator.

The noise model's window rule therefore DOES transfer, with one
amendment: there the smooth curve is DISCARDED and only the residue
above f_c is kept, so distorting it errs safe and any order will do;
here the smooth curve IS the measurement, so the order must be high
enough to represent the signal inside the window.

Usage: python analysis/ge_savgol_overlay.py [out.png]
       [--dir DataSet/exp/case_02/Mx] [--bag pos_Mx_01] [--z-com 0.261]
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import critical_value_getter_piecewise as cvp             # noqa: E402
from utils.extractor import load_excitation_dataset       # noqa: E402
from utils import math_tools                              # noqa: E402
from error_budget import ge_moment, LP                    # noqa: E402
from analysis.rate_derivative import omega_dot            # noqa: E402
from analysis.ge_dynamics_check import (                  # noqa: E402
    MASS_KG, G, OFF_SIGN, OFF_MM, j_parallel)

#            window, poly order, colour, dash
VARIANTS = [(9, 2, '#1f77b4', '-'),        # deployed
            (21, 2, '#7b3294', '-'),
            (41, 2, '#c0392b', '-'),
            (41, 5, '#148f77', '--')]      # order matched to the curvature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='ge_savgol_overlay.png')
    p.add_argument('--dir', default='DataSet/exp/case_02/Mx')
    p.add_argument('--bag', default=None)
    p.add_argument('--z-com', type=float, default=0.261)
    p.add_argument('--jp-mode', choices=['parallel', 'identity'],
                   default='parallel',
                   help="parallel: J_COM + m(z^2 + l_p^2), the CAD "
                        "parallel-axis value the batch uses; identity: "
                        "W z / C2^2, which sits below the rigid-body floor")
    a_ = p.parse_args()

    d = Path(a_.dir)
    axis = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    W = MASS_KG[case] * G
    off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2, k_gain = cvp.estimate_rig_constants(bags, axis)
        crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2,
                                               ramp_gain=k_gain)
    name = a_.bag or sorted(c.bag_name for c in crits
                            if c.bag_name.startswith('pos'))[0]
    crit = next(c for c in crits if c.bag_name == name)
    bag = next(b for b in bags if b.name == name)
    pos = name.startswith('pos')
    s = 1.0 if pos else -1.0

    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
    lp = (piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs'])
          else LP[axis])
    arm = lp + s * off_truth
    j_p = (j_parallel(axis, a_.z_com, MASS_KG[case])
           if a_.jp_mode == 'parallel' else W * a_.z_com / c2 ** 2)

    sig = cvp.prepare_signals(bag, axis)
    roll, pitch = math_tools.quaternion_to_euler_vectorized(
        bag.odom.quaternion)
    phi_all = roll if axis == 'x' else pitch
    n = min(len(phi_all), len(sig['t']))
    _, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    j = crit.onset_idx
    i1 = min(i1, n - 1)
    sl = slice(j, i1 + 1)

    tau = sig['t'][sl] - sig['t'][j]
    phi = s * (phi_all[sl] - phi_all[j])
    m = s * sig['moment'][sl]
    f = sig['f_col'][sl]
    dt = float(np.median(np.diff(tau)))
    om_full = s * sig['omega'][:n]
    # ge_moment returns the whole trace; `window` only scopes its R/4 test
    ge_mod = s * ge_moment(bag, sig, axis, n, pos, window=sl)[sl]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.775, bottom=0.115,
                        wspace=0.24)

    print(f"\n  {case}/{axname}/{name}:  C2 = {c2:.3f} rad/s, "
          f"1/C2 = {1e3/c2:.0f} ms,  T_s = {1e3*dt:.1f} ms,  "
          f"window {len(tau)} samples / {tau[-1]:.2f} s")
    print(f"  J_P = {j_p:.4f} kg m^2 ({a_.jp_mode});  for reference: "
          f"parallel axis {j_parallel(axis, a_.z_com, MASS_KG[case]):.4f}, "
          f"identity W z/C2^2 {W*a_.z_com/c2**2:.4f}\n")
    print(f"  {'w':>4}{'ord':>5}{'[ms]':>7}{'e-fold':>9}{'f_c~[Hz]':>10}"
          f"{'|om_dot|max':>13}{'slope':>10}{'at phi=0':>10}")
    for w, p_, c, ls in VARIANTS:
        od = omega_dot(om_full, dt, w, p_)[sl]
        ge_dyn = (j_p * od - m - f * lp
                  + W * arm * np.cos(phi) - W * a_.z_com * np.sin(phi))
        sd, id_ = np.polyfit(phi, ge_dyn, 1)
        lab = (f'SG {w} samples ({1e3*w*dt:.0f} ms), order {p_}')
        a1.plot(tau, 1e3 * ge_dyn, ls, color=c, lw=1.5, label=lab)
        a2.plot(tau, od, ls, color=c, lw=1.5, label=lab)
        print(f"  {w:4d}{p_:5d}{1e3*w*dt:7.0f}{w*dt*c2:9.2f}{2.0/(w*dt):10.1f}"
              f"{np.abs(od).max():13.2f}"
              f"{1e3*sd*np.pi/180:10.1f}{1e3*id_:10.1f}")

    a1.plot(tau, 1e3 * ge_mod, '-', color='#e08214', lw=2.2,
            label='image-superposition model')
    a1.axhline(0, color='0.6', lw=0.7)
    a1.set_xlabel(r'$\tau$ [s]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) the dynamic inversion: the window-driven steepening\n'
                 'is a filter artefact; the anomaly itself is not',
                 fontsize=11)
    a1.legend(fontsize=8.5, loc='lower left')
    a1.grid(alpha=0.22, lw=0.4)

    a2.set_xlabel(r'$\tau$ [s]', fontsize=10)
    a2.set_ylabel(r'$\dot\omega$ [rad/s$^2$]', fontsize=10)
    a2.set_title(r'(b) why: a parabola cannot follow '
                 r'$\dot\omega\sim\sinh(C_2\tau)$ over' '\n'
                 f'{41*dt*c2:.1f} e-foldings (163 ms each) -- '
                 'order 5 over the same window can', fontsize=11)
    a2.legend(fontsize=8.5, loc='upper left')
    a2.grid(alpha=0.22, lw=0.4)

    jp_note = ('CAD parallel axis' if a_.jp_mode == 'parallel'
               else r'from $Wz/C_2^2$')
    fig.suptitle('Denoising the GE dynamic inversion: raise the order, '
                 'not the window\n'
                 f'({case}/{axname}/{name}, '
                 rf'$z_{{CoM}}$ = {a_.z_com:.3f} m, '
                 rf'$J_P$ = {j_p:.3f} kg m$^2$, {jp_note})',
                 fontsize=11.5, y=0.995)
    fig.savefig(a_.out, dpi=150)
    print(f"\n  wrote {a_.out}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
