#!/usr/bin/env python3
"""Is the ground-effect model right DURING rotation?  A dynamic inversion.

The static check (analysis/mcrit_prediction.py) evaluates the balance at
the onset only, where omega_dot = 0, and there ground effect is
degenerate with a contact-lever offset.  Once the vehicle is rotating
the balance carries an independent, measured term -- J_P omega_dot --
and the rotor heights sweep through a range, so the ATTITUDE DEPENDENCE
of the ground-effect moment becomes observable.

Rearranging the exact contact-phase dynamics (tipping-positive sense),

    J_P omega_dot = m + f l_p + dM_GE - W a cos(phi) + W z_CoM sin(phi)

for the one unmeasured term gives a per-sample estimate

    dM_GE^dyn = J_P omega_dot - m - f l_p + W a cos(phi) - W z_CoM sin(phi)

with  m = s * M_cmd,  a = l_p + s * sgn_axis * lambda_truth,  s = +-1 the
tip direction.  Every ingredient is measured or a ground truth: the
commanded moment, the collective thrust from the rotor speeds, the
attitude and rate from odometry, the pivot arm from the odometry circle
fit, and the CoM offset from the load-cell table.

This is compared against the parameter-free image-superposition model
(the 'interf' branch of analysis/ge_trajectory.py).

CONDITIONING WARNING.  dM_GE is a few percent of terms that are
themselves of order W a ~ 5 N.m, so the inversion amplifies errors in
a, z_CoM and J_P.  Rules of thumb at phi = 5 deg:
    1 mm of l_p        -> 32 mN.m      (~20% of the GE signal)
    10 mm of z_CoM     -> 28 mN.m
    10% of J_P         -> ~75 mN.m at the largest omega_dot
So the LEVEL of dM_GE^dyn is not trustworthy on its own.  What is
comparatively well conditioned is its ATTITUDE SLOPE, because the
gravity terms contribute known functional forms (cos, sin) whose
coefficients are fixed by a and z_CoM, while J_P omega_dot is measured.
Both are reported, with a sensitivity sweep over J_P and z_CoM.

Usage
-----
PYTHONPATH=<stubs> python analysis/ge_dynamics_check.py \
    [--dir DataSet/exp/case_02/Mx] [--bag pos_Mx_01] [--z-com 0.250]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment, LP

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}                  # Table 7
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dir', default='DataSet/exp/case_02/Mx')
    p.add_argument('--bag', default=None,
                   help="bag name; default = first slow positive run")
    p.add_argument('--z-com', type=float, default=0.250)
    p.add_argument('--savgol', type=int, default=9,
                   help="Savitzky-Golay window for omega_dot [samples]")
    p.add_argument('--out', default='ge_dynamics_check.pdf')
    args = p.parse_args()

    d = Path(args.dir)
    axis = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    W = MASS_KG[case] * G
    off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3   # m, signed

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2, k_gain = cvp.estimate_rig_constants(bags, axis)
        crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2,
                                               ramp_gain=k_gain)
    name = args.bag or sorted(c.bag_name for c in crits
                              if c.bag_name.startswith('pos'))[0]
    crit = next(c for c in crits if c.bag_name == name)
    bag = next(b for b in bags if b.name == name)
    pos = name.startswith('pos')
    s = 1.0 if pos else -1.0

    # ---------------------------------------------------------- geometry
    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
    lp = (piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs'])
          else LP[axis])
    lam = s * off_truth                     # CoM offset in the tipping sense
    a = lp + lam
    j_p = W * args.z_com / c2 ** 2          # effective inertia, self-consistent

    print(f"\n{case}/{axname}/{name}   ({'pos' if pos else 'neg'})")
    print(f"  W = {W:.2f} N,  z_CoM = {args.z_com:.3f} m,  "
          f"C2 = {c2:.3f} rad/s,  K = {k_gain:.4f}")
    print(f"  l_p (odom fit) = {1e3*lp:.1f} mm,  lambda_truth = "
          f"{1e3*lam:+.2f} mm,  a = {1e3*a:.1f} mm")
    print(f"  J_P = W z/C2^2 = {j_p:.4f} kg.m^2")

    # ---------------------------------------------------------- signals
    sig = cvp.prepare_signals(bag, axis)
    t, om_s, mom_s, f_s = (sig['t'], sig['omega'], sig['moment'],
                           sig['f_col'])
    roll, pitch = math_tools.quaternion_to_euler_vectorized(
        bag.odom.quaternion)
    phi_all = roll if axis == 'x' else pitch
    n = min(len(phi_all), len(t))
    _, i1 = cvp.detect_excitation_window(mom_s,
                                         moment_cap=cvp.MOMENT_CAP.get(axis))
    j = crit.onset_idx
    i1 = min(i1, n - 1)
    sl = slice(j, i1 + 1)

    tau = t[sl] - t[j]
    phi = s * (phi_all[sl] - phi_all[j])      # excursion, tipping-positive
    om = s * om_s[sl]
    m = s * mom_s[sl]
    f = f_s[sl]

    w = min(args.savgol if args.savgol % 2 else args.savgol + 1,
            len(tau) - (1 - len(tau) % 2))
    dt = float(np.median(np.diff(tau)))
    om_dot = savgol_filter(om, w, 2, deriv=1, delta=dt)

    # ------------------------------------------- dynamic inversion vs model
    ge_dyn = (j_p * om_dot - m - f * lp
              + W * a * np.cos(phi) - W * args.z_com * np.sin(phi))
    ge_mod_raw = ge_moment(bag, sig, axis, n, pos)
    ge_mod = s * ge_mod_raw[sl] if ge_mod_raw is not None else None
    if ge_mod is not None and np.mean(ge_mod) < 0:
        ge_mod = -ge_mod                      # tipping-positive convention

    def lin(y):
        return np.polyfit(phi, y, 1)          # [slope per rad, intercept]

    print(f"\n  window: {len(tau)} samples, tau {tau[-1]:.3f} s, "
          f"excursion {np.rad2deg(phi[-1]):.2f} deg, "
          f"|omega_dot| max {np.abs(om_dot).max():.2f} rad/s^2")
    print(f"\n  {'quantity':22} {'mean':>9} {'at phi=0':>10} "
          f"{'slope [mN.m/deg]':>18}")
    for lbl, y in (('dM_GE dynamic', ge_dyn), ('dM_GE model', ge_mod)):
        if y is None:
            continue
        sl_, ic = lin(y)
        print(f"  {lbl:22} {1e3*np.mean(y):9.1f} {1e3*ic:10.1f} "
              f"{1e3*sl_*np.pi/180:18.2f}")
    if ge_mod is not None:
        r = ge_dyn - ge_mod
        print(f"  {'residual (dyn-model)':22} {1e3*np.mean(r):9.1f} "
              f"{'':10} RMS {1e3*np.sqrt(np.mean(r**2)):.1f} mN.m")

    # ------------------------------------------------------- sensitivity
    print(f"\n  sensitivity of the DYNAMIC estimate "
          f"(mean level / slope [mN.m, mN.m/deg]):")
    print(f"  {'':10}", end='')
    for zz in (0.20, 0.25, 0.30):
        print(f"{'z=' + format(zz, '.2f'):>22}", end='')
    print()
    for fac in (0.8, 1.0, 1.2):
        print(f"  J_P x{fac:.1f}  ", end='')
        for zz in (0.20, 0.25, 0.30):
            jj = fac * W * zz / c2 ** 2
            y = (jj * om_dot - m - f * lp
                 + W * a * np.cos(phi) - W * zz * np.sin(phi))
            sl_, _ = np.polyfit(phi, y, 1)
            print(f"{1e3*np.mean(y):11.1f} /{1e3*sl_*np.pi/180:9.2f}", end='')
        print()
    print(f"\n  for scale: model mean = "
          f"{1e3*np.mean(ge_mod):.1f} mN.m, slope = "
          f"{1e3*lin(ge_mod)[0]*np.pi/180:.2f} mN.m/deg" if ge_mod is not None
          else "")

    # ------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    dphi_deg = np.rad2deg(phi)
    ax[0].plot(tau, 1e3 * ge_dyn, lw=1.2, label='dynamic inversion')
    if ge_mod is not None:
        ax[0].plot(tau, 1e3 * ge_mod, lw=2.0, label='image-superposition model')
    ax[0].set_xlabel(r'$\tau$ [s]'); ax[0].set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]')
    ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=.25)
    ax[1].plot(dphi_deg, 1e3 * ge_dyn, lw=1.2)
    if ge_mod is not None:
        ax[1].plot(dphi_deg, 1e3 * ge_mod, lw=2.0)
    ax[1].set_xlabel(r'excursion $\delta\varphi$ [deg]'); ax[1].grid(alpha=.25)
    for a_ in ax:
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
    fig.suptitle(f'{case}/{axname}/{name}   '
                 rf'($z_{{CoM}}={args.z_com:.3f}$ m, $J_P={j_p:.3f}$ kg m$^2$)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f"\n  figure -> {args.out}")


if __name__ == '__main__':
    main()
