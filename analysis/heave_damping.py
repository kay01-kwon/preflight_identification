#!/usr/bin/env python3
"""Is the dynamic-inversion residual the rotors' own heave damping?

The applied moment is reconstructed from a STATIC thrust map,
m = C_T sum_i b_i Omega_i^2.  That map is blind to the vehicle's motion.
While the airframe tips at rate omega about the contact line, rotor i --
sitting a horizontal distance r_i from that line -- climbs (or descends)
at v_i = omega * r_i.  A climbing rotor at fixed RPM makes LESS thrust,
so the reconstructed moment overstates the real one by

    dM_damp = sum_i (dT/dv_c)_i * (omega r_i) * r_i
            = omega * sum_i k_i r_i^2 ,        k_i = (dT/dv_c)_i < 0.

This is a genuine unmodelled term, it is NOT ground effect and NOT an
inertia error -- which is why the (J_P, z_CoM) and C_T sweeps could not
absorb it.  Two properties make it the natural suspect:

  * sign.  k_i < 0, so the term is negative for omega > 0.  The measured
    residual falls with attitude.
  * it vanishes at the onset.  omega = 0 there by definition, so however
    large this term is, it cannot move M_crit.

The a-priori magnitude has no free constant beyond the inflow model.
Momentum theory at fixed RPM gives the hover heave derivative

    k_i = -2 T_i / v_h,i ,    v_h,i = sqrt(T_i / (2 rho A)),

which is the ideal (upper-bound) value; blade-element-momentum with
finite solidity reduces it by a factor eta in [0.3, 1].  So the test is:
regress the residual on the PREDICTED damping regressor and read eta
off.  If eta lands inside the aerodynamically admissible band AND is
invariant across ramp rates, the residual is explained.

phi, omega and omega_dot are collinear over a ramp window, so a good fit
alone proves nothing -- the discriminator is that eta must not drift
with ramp rate, and must agree with the aerodynamics in magnitude.

Usage: PYTHONPATH=<stubs> python analysis/heave_damping.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment
from analysis.pnls_constants import PNLS_CONSTANTS
from analysis.rate_derivative import omega_dot, edge_margin

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G, Z = 9.81, 0.261
RHO_AIR, R_ROTOR, L_ARM, C_T = 1.225, 0.127, 0.265, 1.3175e-7
A_DISK = np.pi * R_ROTOR ** 2
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}
BINS = [(0.0, 0.2, 'slow'), (0.2, 0.6, 'mid'), (0.6, 9.9, 'fast')]


def pivot_arms(axis, pos, lp):
    """Horizontal rotor arms about the contact line, tipping-positive."""
    ang = np.deg2rad(30 + 60 * np.arange(6))
    lx, ly = L_ARM * np.cos(ang), L_ARM * np.sin(ang)
    if axis == 'x':
        return ly + (lp if pos else -lp)
    return -(lx + (-lp if pos else lp))


rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    ax = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    mass = MASS[case]
    W = mass * G
    j_p = J_CAD[ax] + mass * (Z ** 2 + LP[ax] ** 2)
    c2f, kf = PNLS_CONSTANTS[(case, axname)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2f,
                                               ramp_gain=kf)
    by = {b.name: b for b in bags}
    for crit in crits:
        bag = by[crit.bag_name]
        s = 1.0 if crit.bag_name.startswith('pos') else -1.0
        roll, pitch = math_tools.quaternion_to_euler_vectorized(
            bag.odom.quaternion)
        phi_all = roll if ax == 'x' else pitch
        sig = cvp.prepare_signals(bag, ax)
        n = min(len(phi_all), len(sig['t']))
        i0w, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
        j = crit.onset_idx
        i1 = min(i1, n - 1)
        if i1 - j < 15:
            continue
        sl = slice(j, i1 + 1)
        tau = sig['t'][sl] - sig['t'][j]
        w = min(9, len(tau) - (1 - len(tau) % 2))
        if w < 5:
            continue

        phi_abs = s * phi_all[sl]
        phi_rel = s * (phi_all[sl] - phi_all[j])
        m = s * sig['moment'][sl]
        f = sig['f_col'][sl]
        # differentiate the FULL trace, then slice: slicing first would
        # put the onset on the filter's extrapolated left edge, exactly
        # where omega_dot must be zero (analysis/rate_derivative.py)
        dt = float(np.median(np.diff(sig['t'][:n])))
        om_full = s * sig['omega'][:n]
        omd_full = omega_dot(om_full, dt, w)
        om, omd = om_full[sl], omd_full[sl]
        if not edge_margin(n, j, i1, w)['ok']:
            continue

        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) \
            else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest)
        if raw is None:
            continue

        ge = (j_p * omd - m - f * lp + W * a * np.cos(phi_abs)
              - W * Z * np.sin(phi_abs))
        resid = ge - s * raw[sl]                      # N.m

        # --- a-priori heave-damping regressor, no fitted constant ------
        t0 = bag.odom.t[0]
        t6 = C_T * bag.rpm.rpm.astype(np.float64) ** 2
        thrust = np.vstack([np.interp(sig['t'][sl], bag.rpm.t - t0, t6[:, q])
                            for q in range(6)]).T          # (N, 6)
        thrust = np.clip(thrust, 1e-3, None)
        arms = pivot_arms(ax, s > 0, lp)                   # (6,)
        v_h = np.sqrt(thrust / (2.0 * RHO_AIR * A_DISK))
        k_i = -2.0 * thrust / v_h                          # N/(m/s), ideal
        # dM_damp = omega * sum_i k_i r_i^2   (already tipping-positive:
        # omega and the arms are both taken in the tipping sense)
        d_ideal = (k_i * arms ** 2).sum(axis=1)            # N.m/(rad/s)
        reg = d_ideal * om                                 # N.m

        mdot = abs(float(np.polyfit(tau, m, 1)[0])) or np.nan
        rows.append(dict(case=case, ax=axname, mdot=mdot, tau=tau,
                         phi=phi_rel, om=om, omd=omd, resid=resid,
                         reg=reg, d_ideal=d_ideal, arms=arms,
                         model=s * raw[sl]))
    print(f"  loaded {case}/{axname}", flush=True)

print(f"\n{len(rows)} runs\n")

# ---------------------------------------------------------------------
# 1. a-priori magnitude
r2sum = np.mean([np.sum(r['arms'] ** 2) for r in rows])
dmed = np.median([np.median(np.abs(r['d_ideal'])) for r in rows])
print(f"geometry:   <sum r_i^2> = {r2sum:.4f} m^2")
print(f"ideal heave damping  |D| = {dmed:.3f} N.m/(rad/s)  (momentum "
      f"theory, eta = 1)\n")

# 2. what damping would the residual need?
print("per-run single-regressor fits (no intercept beyond a constant):")
print(f"{'regressor':<26}{'unit':>14}{'all':>10}"
      + ''.join(f"{lab:>9}" for _, _, lab in BINS) + f"{'max/min':>11}"
      + f"{'R^2':>7}")
for name, key, unit in [('phi   (stiffness)', 'phi', 'N.m/rad'),
                        ('omega (damping)', 'om', 'N.m/(rad/s)'),
                        ('omega_dot (inertia)', 'omd', 'kg.m^2'),
                        ('eta * ideal damping', 'reg', '-')]:
    coefs, r2s = [], []
    per = {lab: [] for _, _, lab in BINS}
    for r in rows:
        x, y = r[key], r['resid']
        A = np.column_stack([x, np.ones_like(x)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = 1 - np.sum((y - A @ c) ** 2) / max(
            np.sum((y - y.mean()) ** 2), 1e-18)
        coefs.append(c[0])
        r2s.append(ss)
        for lo, hi, lab in BINS:
            if lo <= r['mdot'] < hi:
                per[lab].append(c[0])
    mu = [np.mean(per[lab]) if per[lab] else np.nan for _, _, lab in BINS]
    am = [abs(v) for v in mu]
    ratio = max(am) / min(am) if min(am) else np.inf
    print(f"{name:<26}{unit:>14}{np.mean(coefs):10.3f}"
          + ''.join(f"{v:9.3f}" for v in mu)
          + f"{ratio:11.2f}{np.mean(r2s):7.2f}")

# 3. how much of the phi-slope does the a-priori term remove?
print("\nattitude slope of the residual, before and after subtracting the "
      "a-priori\nheave-damping term (eta = 1, nothing fitted):")
print(f"{'':10}{'bin':>8}{'n':>5}{'slope before':>15}{'slope after':>14}"
      f"{'removed':>10}")
for lo, hi, lab in BINS:
    sel = [r for r in rows if lo <= r['mdot'] < hi]
    if not sel:
        continue
    b, a_ = [], []
    for r in sel:
        deg = np.rad2deg(r['phi'])
        if np.ptp(deg) < 0.2:
            continue
        b.append(1e3 * np.polyfit(deg, r['resid'], 1)[0])
        a_.append(1e3 * np.polyfit(deg, r['resid'] - r['reg'], 1)[0])
    rem = 100 * (1 - abs(np.median(a_)) / abs(np.median(b)))
    print(f"{'':10}{lab:>8}{len(b):5d}{np.median(b):12.1f} mN.m/deg"
          f"{np.median(a_):11.1f}{rem:9.0f}%")
allb = [1e3 * np.polyfit(np.rad2deg(r['phi']), r['resid'], 1)[0]
        for r in rows if np.ptp(np.rad2deg(r['phi'])) >= 0.2]
alla = [1e3 * np.polyfit(np.rad2deg(r['phi']), r['resid'] - r['reg'], 1)[0]
        for r in rows if np.ptp(np.rad2deg(r['phi'])) >= 0.2]
print(f"{'':10}{'ALL':>8}{len(allb):5d}{np.median(allb):12.1f} mN.m/deg"
      f"{np.median(alla):11.1f}"
      f"{100 * (1 - abs(np.median(alla)) / abs(np.median(allb))):9.0f}%")

# ---------------------------------------------------------------------
# 4. LEVEL, not just slope.  The per-run regressions above carry a free
#    intercept, so they test only the attitude gradient.  Does the
#    heave-corrected inversion also agree with the GE model in
#    magnitude?  Compare
#        inv_corrected = inversion - dM_damp   against   model
#    over the window, and separately at the onset itself (tau = 0, where
#    omega = 0 so the damping term is identically zero and the balance
#    reduces to the static threshold check).
print("\nLEVEL comparison [mN.m], median over runs:")
print(f"{'':4}{'GE model':>12}{'inversion':>12}{'inv - model':>13}"
      f"{'inv-damp-model':>16}{'at onset':>11}")
mod = np.array([1e3 * np.median(r['model']) for r in rows])
inv = np.array([1e3 * np.median(r['resid'] + r['model']) for r in rows])
d0 = np.array([1e3 * np.median(r['resid']) for r in rows])
d1 = np.array([1e3 * np.median(r['resid'] - r['reg']) for r in rows])
on = np.array([1e3 * float(r['resid'][0]) for r in rows])
print(f"{'':4}{np.median(mod):12.1f}{np.median(inv):12.1f}"
      f"{np.median(d0):13.1f}{np.median(d1):16.1f}{np.median(on):11.1f}")
print(f"{'RMS':>4}{'':12}{'':12}{np.sqrt(np.mean(d0**2)):13.1f}"
      f"{np.sqrt(np.mean(d1**2)):16.1f}{np.sqrt(np.mean(on**2)):11.1f}")
print(f"\nratio (heave-corrected inversion) / (GE model): "
      f"median {np.median((inv - 1e3*np.array([np.median(r['reg']) for r in rows])) / mod):.2f}")

# ---------------------------------------------------------------------
# 5. dump for the figure (analysis/heave_damping_figure.py)
import os
if os.environ.get('HD_DUMP'):
    rid = np.concatenate([np.full(len(r['phi']), i)
                          for i, r in enumerate(rows)])
    d_fit = []
    for r in rows:
        A = np.column_stack([r['om'], np.ones_like(r['om'])])
        c, *_ = np.linalg.lstsq(A, r['resid'], rcond=None)
        d_fit.append(c[0])
    np.savez(os.environ['HD_DUMP'],
             rid=rid,
             phi=np.concatenate([np.rad2deg(r['phi']) for r in rows]),
             om=np.concatenate([r['om'] for r in rows]),
             resid=np.concatenate([1e3 * r['resid'] for r in rows]),
             model=np.concatenate([1e3 * r['model'] for r in rows]),
             reg=np.concatenate([1e3 * r['reg'] for r in rows]),
             mdot=np.array([r['mdot'] for r in rows]),
             d_fit=np.array(d_fit),
             d_ideal=np.array([np.median(r['d_ideal']) for r in rows]),
             case=np.array([r['case'] for r in rows]),
             axis=np.array([r['ax'] for r in rows]))
    print(f"\ndumped -> {os.environ['HD_DUMP']}")
