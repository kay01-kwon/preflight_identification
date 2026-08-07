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

# Physically consistent constants from analysis/constrained_calibration.py:
# one z_CoM for the vehicle, one J_P per axis, shared by all ten datasets.
# These replace the per-dataset (C2, K), whose 1/K implies z_CoM between
# 0.061 and 0.554 m and so cannot be used in a dynamic inversion.
Z_COM_SHARED = 0.174                       # m   (W z_CoM = 5.50 N.m)
J_AXIS = {'x': 0.240, 'y': 0.153}          # kg.m^2

# Sweeping z_CoM requires J_P to follow: the parallel-axis theorem forces
# J_P = J_CoM + m (z_CoM^2 + l_p^2).  Anchoring J_CoM on the constrained fit
# at z = 0.174 m and m = 3.220 kg gives the body-frame inertias below, from
# which J_P at any other z_CoM follows without a second free parameter.
J_COM = {'x': J_AXIS['x'] - 3.220 * (Z_COM_SHARED ** 2 + LP['x'] ** 2),
         'y': J_AXIS['y'] - 3.220 * (Z_COM_SHARED ** 2 + LP['y'] ** 2)}


def j_parallel(axis, z_com, mass):
    """Parallel-axis J_P at an arbitrary z_CoM (no extra free parameter)."""
    return J_COM[axis] + mass * (z_com ** 2 + LP[axis] ** 2)


def analyse(bag, crit, axis, sig, phi_all, n, z_com, j_p, savgol=9):
    """Dynamic inversion of dM_GE for one run; returns the fit summary."""
    case_w = analyse.W
    pos = crit.bag_name.startswith('pos')
    s = 1.0 if pos else -1.0
    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
    lp = (piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs'])
          else LP[axis])
    a = lp + s * analyse.off_truth

    _, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    j = crit.onset_idx
    i1 = min(i1, n - 1)
    if i1 - j < 15:
        return None
    sl = slice(j, i1 + 1)
    tau = sig['t'][sl] - sig['t'][j]
    phi = s * (phi_all[sl] - phi_all[j])
    om = s * sig['omega'][sl]
    m = s * sig['moment'][sl]
    f = sig['f_col'][sl]

    w = min(savgol if savgol % 2 else savgol + 1,
            len(tau) - (1 - len(tau) % 2))
    if w < 5:
        return None
    dt = float(np.median(np.diff(tau)))
    om_dot = savgol_filter(om, w, 2, deriv=1, delta=dt)

    ge_dyn = (j_p * om_dot - m - f * lp
              + case_w * a * np.cos(phi) - case_w * z_com * np.sin(phi))
    raw = ge_moment(bag, sig, axis, n, pos)
    if raw is None:
        return None
    ge_mod = s * raw[sl]
    if np.mean(ge_mod) < 0:
        ge_mod = -ge_mod

    sd, id_ = np.polyfit(phi, ge_dyn, 1)
    sm, im = np.polyfit(phi, ge_mod, 1)
    # An effective arm varying with tilt would show up as an extra W a' phi,
    # so a' = (slope_model - slope_dyn)/W.  CAUTION: this is fully degenerate
    # with an error in (J_P, z_CoM) -- see the batch summary -- and is
    # reported only to expose that degeneracy, not as a measurement.
    a_rate = -(sd - sm) / case_w              # m/rad
    return dict(bag=crit.bag_name, dir='pos' if pos else 'neg',
                lp_mm=1e3 * lp, dphi_deg=float(np.rad2deg(phi[-1])),
                int_dyn=1e3 * id_, int_mod=1e3 * im,
                slope_dyn=1e3 * sd * np.pi / 180,
                slope_mod=1e3 * sm * np.pi / 180,
                a_rate_mm_per_deg=1e3 * a_rate * np.pi / 180)


def batch(z_list, savgol, jp_mode='parallel'):
    """All ten datasets, both tip directions, for each z_CoM in z_list.

    Each dataset is loaded once and reused across the z values.  J_P either
    follows the parallel-axis theorem (default, physically forced) or is
    pinned at the constrained-fit value (jp_mode='fixed'), which lets the
    two effects be separated.
    """
    root = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
    out = {z: [] for z in z_list}
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        case, axname = d.parent.name, d.name
        mass = MASS_KG[case]
        analyse.W = mass * G
        analyse.off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        by = {b.name: b for b in bags}
        cache = {}
        for z_com in z_list:
            j_p = (j_parallel(axis, z_com, mass) if jp_mode == 'parallel'
                   else J_AXIS[axis])
            with contextlib.redirect_stdout(io.StringIO()):
                c2 = float(np.sqrt(analyse.W * z_com / j_p))
                crits, _ = cvp.extract_piecewise_batch(
                    bags, axis, cosh_c2=c2,
                    ramp_gain=1.0 / (analyse.W * z_com))
            for crit in crits:
                bag = by[crit.bag_name]
                if crit.bag_name not in cache:
                    sig = cvp.prepare_signals(bag, axis)
                    roll, pitch = math_tools.quaternion_to_euler_vectorized(
                        bag.odom.quaternion)
                    cache[crit.bag_name] = (
                        sig, roll if axis == 'x' else pitch)
                sig, phi_all = cache[crit.bag_name]
                nn = min(len(phi_all), len(sig['t']))
                r = analyse(bag, crit, axis, sig, phi_all, nn, z_com, j_p,
                            savgol)
                if r:
                    r.update(case=case, axis=axname, z_com=z_com, j_p=j_p)
                    out[z_com].append(r)
        print(f"  {case}/{axname}: "
              + ', '.join(f"z={z:.2f} J={j_parallel(axis, z, mass):.3f} "
                          f"n={sum(1 for r in out[z] if r['case'] == case and r['axis'] == axname)}"
                          for z in z_list))
    return out


def sweep_report(by_z):
    """One summary line per z_CoM, so the trend is visible at a glance."""
    print(f"\n{'z_CoM':>7} {'J_roll':>8} {'J_pitch':>8} | "
          f"{'intercept dyn-model [mN.m]':>28} | "
          f"{'slope dyn-model [mN.m/deg]':>28} | {'-W z':>9}")
    print(f"{'[m]':>7} {'':8} {'':8} | {'mean':>9} {'RMS':>9} {'<50':>8} | "
          f"{'mean':>9} {'RMS':>9} {'':8} | {'[mN.m/deg]':>9}")
    print('-' * 104)
    for z, rows in sorted(by_z.items()):
        keys = sorted({(r['case'], r['axis'], r['dir']) for r in rows})
        di, ds = [], []
        for k in keys:
            g = [r for r in rows if (r['case'], r['axis'], r['dir']) == k]
            di.append(np.mean([r['int_dyn'] - r['int_mod'] for r in g]))
            ds.append(np.mean([r['slope_dyn'] - r['slope_mod'] for r in g]))
        di, ds = np.array(di), np.array(ds)
        jr = j_parallel('x', z, 3.220)
        jq = j_parallel('y', z, 3.220)
        wz_slope = -31.59 * z * np.pi / 180 * 1e3
        print(f"{z:7.2f} {jr:8.3f} {jq:8.3f} | {di.mean():+9.1f} "
              f"{np.sqrt(np.mean(di**2)):9.1f} {int((np.abs(di) < 50).sum()):3d}/{len(di):<4d} | "
              f"{ds.mean():+9.2f} {np.sqrt(np.mean(ds**2)):9.2f} {'':8} | "
              f"{wz_slope:9.1f}")
    print('-' * 104)
    print("The last column is d/dphi[-W z_CoM sin phi], the term the growth")
    print("of J_P omega_dot has to cancel.  The model's own slope is")
    print("-0.2 .. -2.5 mN.m/deg, so it is testable only if that cancellation")
    print("closes to a couple of percent.")


def report(rows):
    import csv as _csv
    if not rows:
        raise SystemExit("no runs analysed -- check the dataset path")
    print(f"\n{'case':9} {'ax':3} {'dir':4} {'n':>3} {'dphi':>6} | "
          f"{'intercept [mN.m]':>22} | {'slope [mN.m/deg]':>22} | "
          f"{'a-rate':>9}")
    print(f"{'':9} {'':3} {'':4} {'':3} {'[deg]':>6} | "
          f"{'dyn':>10} {'model':>11} | {'dyn':>10} {'model':>11} | "
          f"{'[mm/deg]':>9}")
    print('-' * 92)
    keys = sorted({(r['case'], r['axis'], r['dir']) for r in rows})
    agg = {}
    for k in keys:
        g = [r for r in rows if (r['case'], r['axis'], r['dir']) == k]
        mean = {f: float(np.mean([r[f] for r in g])) for f in
                ('dphi_deg', 'int_dyn', 'int_mod', 'slope_dyn', 'slope_mod',
                 'a_rate_mm_per_deg')}
        agg[k] = mean
        print(f"{k[0]:9} {k[1]:3} {k[2]:4} {len(g):3d} {mean['dphi_deg']:6.2f} | "
              f"{mean['int_dyn']:10.1f} {mean['int_mod']:11.1f} | "
              f"{mean['slope_dyn']:10.2f} {mean['slope_mod']:11.2f} | "
              f"{mean['a_rate_mm_per_deg']:9.2f}")
    print('-' * 92)
    d_int = np.array([agg[k]['int_dyn'] - agg[k]['int_mod'] for k in keys])
    d_slp = np.array([agg[k]['slope_dyn'] - agg[k]['slope_mod'] for k in keys])
    ar = np.array([agg[k]['a_rate_mm_per_deg'] for k in keys])
    print(f"intercept  dyn-model: mean {d_int.mean():+7.1f} mN.m, "
          f"RMS {np.sqrt(np.mean(d_int**2)):.1f}, "
          f"{int((np.abs(d_int) < 50).sum())}/{len(d_int)} within 50 mN.m")
    print(f"slope      dyn-model: mean {d_slp.mean():+7.2f} mN.m/deg, "
          f"RMS {np.sqrt(np.mean(d_slp**2)):.2f}")
    print(f"implied arm rate    : mean {ar.mean():+6.2f} mm/deg, "
          f"std {ar.std(ddof=1):.2f}  "
          f"-- DEGENERATE with (J_P, z_CoM), not a measurement")
    print(f"\nWhy the SLOPE carries no information about the model:")
    print(f"  d/dphi[-W z_CoM sin phi] = -W z_CoM = -96 mN.m/deg, which the")
    print(f"  growth of J_P omega_dot must cancel.  The observed slopes span")
    print(f"  -11 .. -100, i.e. they measure how well that cancellation")
    print(f"  closes.  The model's OWN slope is -0.2 .. -2.5 mN.m/deg, i.e.")
    print(f"  0.2-2.6% of the term being cancelled: testing it would need")
    print(f"  z_CoM and J_P to ~2%, and z_CoM is not known to a factor of 2.")
    print(f"  The slope also correlates with the fitted excursion range")
    print(f"  (My neg, 2.3-4.2 deg, is steepest; Mx, 5.3-7.2 deg, flattest),")
    print(f"  the signature of a fitting artefact rather than a physical rate.")
    # antisymmetry test: same sign in both tip directions -> cancels in M_ff
    pairs = [(agg[(c, a, 'pos')]['a_rate_mm_per_deg'],
              agg[(c, a, 'neg')]['a_rate_mm_per_deg'])
             for (c, a, dd) in keys if dd == 'pos'
             and (c, a, 'neg') in agg]
    p_, n_ = np.array([p for p, _ in pairs]), np.array([n for _, n in pairs])
    print(f"\ncommon-mode test (tipping frame): pos {p_.mean():+.2f} vs "
          f"neg {n_.mean():+.2f} mm/deg;  half-difference "
          f"{0.5*np.abs(p_ - n_).mean():.2f} mm/deg")
    print("  a common-mode arm change cancels in M_ff = +-0.5 (M_pos + M_neg);")
    print("  only the half-difference survives the pivot-free average.")
    with open('ge_dynamics_runs.csv', 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nrows -> ge_dynamics_runs.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dir', default='DataSet/exp/case_02/Mx')
    p.add_argument('--bag', default=None,
                   help="bag name; default = first slow positive run")
    p.add_argument('--all', action='store_true',
                   help="batch over all datasets and both tip directions")
    p.add_argument('--z-sweep', default=None,
                   help="comma-separated z_CoM values, e.g. 0.10,0.20,0.30")
    p.add_argument('--jp-mode', choices=['parallel', 'fixed'],
                   default='parallel',
                   help="parallel: J_P follows the parallel-axis theorem")
    p.add_argument('--z-com', type=float, default=Z_COM_SHARED)
    p.add_argument('--savgol', type=int, default=9,
                   help="Savitzky-Golay window for omega_dot [samples]")
    p.add_argument('--out', default='ge_dynamics_check.pdf')
    args = p.parse_args()

    if args.z_sweep:
        zs = [float(v) for v in args.z_sweep.split(',')]
        by_z = batch(zs, args.savgol, args.jp_mode)
        sweep_report(by_z)
        for z in zs:
            print(f"\n===== z_CoM = {z:.3f} m =====")
            report(by_z[z])
        return
    if args.all:
        report(batch([args.z_com], args.savgol, args.jp_mode)[args.z_com])
        return

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
