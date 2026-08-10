#!/usr/bin/env python3
"""Anatomy of the dynamic-inversion residual.  RESULT: instrumentation.

This script was written to test whether the residual was the rotors'
own heave damping -- the static thrust map m = C_T sum b_i Omega_i^2 is
blind to the fact that a rotor at horizontal distance r_i from the
contact line climbs at omega r_i while the airframe tips, and loses
thrust doing so, which would put an omega-proportional term
omega * sum_i k_i r_i^2 into the balance.

THAT ACCOUNT IS RETRACTED.  It fitted well (eta = required/predicted
damping had median 0.96 against momentum theory, with the predicted
0.477 N.m/(rad/s) against 0.514 required) but it was absorbing an
omega-proportional ARTEFACT of two instrumentation errors, which has
the same shape.  Both are now fixed, and with them fixed the a-priori
damping term overshoots by +39 to +55 mN.m/deg -- there is nothing left
for it to explain.

The two errors were:

1. The Savitzky-Golay differentiator was applied to the SLICED window,
   putting the onset on its extrapolated left edge -- exactly where the
   physics says omega = omega_dot = 0.  Worth 1.23 N.m of fabricated
   offset on one measured run.  See analysis/rate_derivative.py.

2. The reported angular rate reads about 10% low.  omega / (d phi/dt)
   is 0.890 (IQR 0.868-0.907) over 137 runs, by three independent
   comparison methods, and mocap arbitrates against the gyro: the odom
   and mocap attitudes agree to 0.999 on roll while the rate sits at
   0.890 of the first and 0.904 of the second.  HD_GAIN divides it out.

Neither touches the identification.  The constrained cosh fit is
exactly scale-invariant in omega -- scaling the data by g scales C_1
and the baseline by g, leaves C_2 alone, and multiplies every residual
by g^2, so the onset argmin does not move -- and the calibrated K
absorbs g outright.  Both errors act only on J_P omega_dot, which the
identification never forms.

With both removed, and phi, omega, omega_dot taken from one
onset-anchored polynomial so the kinematics are self-consistent
(HD_DERIV=polyk:K):

  method                          slope   level ratio    RMS
  Savitzky-Golay 9, as it was     -42.0      0.77       229.8
  + gyro gain only                -25.5      1.00       216.6
  polynomial kinematics only       -5.5      0.93       207.1
  both, polynomial order 4         -3.4      1.07
  both, polynomial order 5        -13.3      1.13
  both, polynomial order 6         +1.4      1.06       203.5
  both, polynomial order 7        -11.7      1.09
  image-superposition model        -1.9      1.00

The claim is the BRACKET, not any single row.  Across orders 4-7 the
attitude gradient spans -13.3 to +1.4 mN.m/deg with the model's -1.9
inside it, and the level ratio sits at 1.06-1.13.  The inversion is
therefore no longer inconsistent with the parameter-free ground-effect
model in either level or gradient, where before it was 22x off in
gradient and 23% low in level.  Polynomial order is now the limiting
systematic, about +-7 mN.m/deg; quoting order 6 alone would be
choosing the order that matches.

On the gain being a single global factor: measured per axis it is
0.891 on roll and 0.890 on pitch, and per case/axis group it spans
0.880-0.907 with no structure, so one number is justified.  The check
was done twice, because comparing the reported p with d(euler roll)/dt
is not automatically the same quantity -- the vehicle turns about the
ground contact line, a world-fixed axis, which in a ZYX decomposition
couples the three Euler rates.  Deriving the body rate exactly from the
attitude instead, omega = 2 vec(conj(q) (x) qdot), gives the same
0.891 / 0.890, so the Euler route was not misleading here.

Separately, mocap and odometry attitudes agree to 0.999 on roll but
1.056 on pitch.  That is a discrepancy between the two ATTITUDE
sources and does not bear on the gain, since both of them place the
rate about 10% low.

THE 0.890 IS NOT DERIVABLE, AND THE CHOICE OF SOURCE MATTERS.  Three
candidate identities were tested and all three fail.  It is not the
Euler-vs-body-rate detour: recovering the body rate exactly from the
attitude, omega = 2 vec(conj(q) (x) qdot), returns the same 0.891 /
0.890.  It is not a misaligned tipping axis: the measured rotation axis
sits 2.4 deg (median) from the excited body axis, cos = 0.999, and
projecting onto the measured axis still gives 0.889.  And it cannot be
side-stepped by dropping the rate altogether: fitting the ATTITUDE with
the same onset-anchored polynomial (HD_DERIV=polyphi:K,
kinematics_from_phi) and differentiating twice gives

  polyphi:5   gradient  +22.8   level 193.9   1.16 x model
  polyphi:6             -90.7         200.7   1.20
  polyphi:7             -11.0         196.0   1.17
  polyk:6 + gain 0.890   +0.3         159.1   0.95

If the attitude were right and the rate simply 0.890 of it, these two
routes would agree.  They differ by 24% in level, so the scale factor
does not fully reconcile them -- J_P omega_dot involves the SECOND
derivative, and the relation between the two odometry outputs is not a
pure scale at that order.  The attitude route is also much noisier, as
two differentiations of a noisier signal must be (residual RMS 257
against 203).

So the honest level bracket spans BOTH kinematic sources, 0.95-1.20 of
the model, not the 0.95-0.99 the rate route alone suggests.  The rate
route is reported as primary because it differentiates once rather than
twice and its scale factor is measured against two independent attitude
sources -- but that is a choice, not a derivation, and it should be
stated as one.

Environment: HD_SAVGOL (width, default 9), HD_DERIV ('sg' | 'poly:K' |
'polyk:K' | 'polyphi:K'), HD_GAIN (rate divisor, default 1.0),
HD_DUMP (npz path).

Usage: PYTHONPATH=<stubs> python analysis/heave_damping.py
"""
import contextlib
import io
import os
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
from analysis.rate_derivative import (omega_dot, edge_margin,
                                      omega_dot_poly,
                                      kinematics_from_phi)

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

# Savitzky-Golay width for omega_dot, in samples at 100 Hz.  Not a free
# choice: the data carry a real ~10 Hz component (period 97 ms), so any
# window wider than that smooths away the very acceleration being
# measured -- 9 samples is 90 ms, right on it.  Overridable so the
# sensitivity can be swept (analysis/omega_dot_probe.py).
SAVGOL_W = int(os.environ.get('HD_SAVGOL', 9))

# HD_DERIV='poly:5' switches to the onset-anchored polynomial fit
# instead of Savitzky-Golay (analysis/rate_derivative.py).
DERIV = os.environ.get('HD_DERIV', 'sg')

# Scale applied to the reported rate.  The odometry angular_vel reads
# about 10% low against the derivative of the attitude, and mocap
# arbitrates in the attitude's favour: d(odom phi)/dt agrees with
# d(mocap phi)/dt to 0.999 on roll, while the gyro sits at 0.890 of the
# first and 0.904 of the second.  The identification does not care --
# the cosh fit is exactly scale-invariant in omega and the calibrated K
# absorbs the factor -- but the dynamic inversion does, because it puts
# J_P omega_dot and W z sin(phi) in the same balance.
GYRO_GAIN = float(os.environ.get('HD_GAIN', 1.0))

# HD_MAXPHI: truncate every window at a common tilt excursion [deg]
# before anything is fitted.  The polynomial kinematics fit the whole
# window and are then differentiated at tau = 0, so a window that runs
# out to 9 deg and one that stops at 2 deg do not give the onset the
# same treatment -- and the measured onset level tracks the window's
# total excursion almost linearly, which a static balance cannot do.
# Capping puts every run on the same footing.  0 disables.
MAXPHI = float(os.environ.get('HD_MAXPHI', 0))


def pivot_arms(axis, pos, lp):
    """Horizontal rotor arms about the contact line, tipping-positive."""
    ang = np.deg2rad(30 + 60 * np.arange(6))
    lx, ly = L_ARM * np.cos(ang), L_ARM * np.sin(ang)
    if axis == 'x':
        return ly + (lp if pos else -lp)
    return -(lx + (-lp if pos else lp))


rows = []
# Why the run count falls between the 140 bags on disk and the rows
# that reach the figure.  Each gate is counted so the drop is reported
# rather than inferred.
drop = dict(no_onset=0, short_window=0, savgol_too_short=0,
            filter_edge=0, no_ge_model=0)
n_bags = 0
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
    n_bags += len(bags)
    drop['no_onset'] += len(bags) - len(crits)
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
        if MAXPHI:
            over = np.flatnonzero(
                np.rad2deg(np.abs(phi_all[j:i1 + 1] - phi_all[j])) > MAXPHI)
            if len(over):
                i1 = j + int(over[0]) - 1
        if i1 - j < 15:
            drop['short_window'] += 1
            continue
        sl = slice(j, i1 + 1)
        tau = sig['t'][sl] - sig['t'][j]
        w = min(SAVGOL_W, len(tau) - (1 - len(tau) % 2))
        if w < 5:
            drop['savgol_too_short'] += 1
            continue

        phi_abs = s * phi_all[sl]
        phi_rel = s * (phi_all[sl] - phi_all[j])
        m = s * sig['moment'][sl]
        f = sig['f_col'][sl]
        # differentiate the FULL trace, then slice: slicing first would
        # put the onset on the filter's extrapolated left edge, exactly
        # where omega_dot must be zero (analysis/rate_derivative.py)
        dt = float(np.median(np.diff(sig['t'][:n])))
        om_full = s * sig['omega'][:n] / GYRO_GAIN
        # Bias from the PRE-ONSET MEAN, where the vehicle is at rest.
        # Using the single onset sample instead leaves that sample's
        # noise (scatter ~19 mrad/s) as a constant offset, which over a
        # 0.7 s window integrates to ~0.7 deg -- 11% of the excursion,
        # and exactly the size of the discrepancy it was blamed for.
        pre = slice(max(0, j - 100), j)
        bias = float(np.mean(om_full[pre])) if j >= 20 else om_full[j]
        if DERIV.startswith('polyphi'):
            # everything from the ATTITUDE, which mocap corroborates:
            # no rate scale factor is involved at all
            ph_fit, om, omd = kinematics_from_phi(
                tau, phi_rel, int(DERIV.split(':')[1]))
            phi_rel = ph_fit
            phi_abs = phi_abs[0] + ph_fit
        elif DERIV.startswith('poly'):
            om = om_full[sl] - bias
            ph_fit, om_fit, omd = omega_dot_poly(
                tau, om, int(DERIV.split(':')[1]))
            if DERIV.startswith('polyk'):
                # take the ATTITUDE from the same polynomial as well, so
                # phi, omega and omega_dot are one consistent kinematic
                # description; otherwise the oscillation is smoothed out
                # of J_P omega_dot and left in W z sin(phi)
                phi_rel = ph_fit
                phi_abs = phi_abs[0] + ph_fit
                om = om_fit
        else:
            omd = omega_dot(om_full, dt, w)[sl]
            om = om_full[sl]
            if not edge_margin(n, j, i1, w)['ok']:
                # only reachable with HD_DERIV=sg; the polynomial
                # derivatives have no filter edge to fall off
                drop['filter_edge'] += 1
                continue

        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) \
            else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest,
                        window=sl)
        if raw is None:
            drop['no_ge_model'] += 1
            continue

        # keep the five terms separately as well: which of them carries
        # the run-to-run scatter decides what the limitation actually is
        term = dict(inertia=j_p * omd, moment=-m, load=-f * lp,
                    grav_a=W * a * np.cos(phi_abs),
                    grav_z=-W * Z * np.sin(phi_abs))
        ge = sum(term.values())
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
        rows.append(dict(**{'t_' + k: v for k, v in term.items()},
                         lp=lp, arm_a=a, W=W, f=f, phi_abs=phi_abs,
                         case=case, ax=axname, bag=crit.bag_name,
                         tip='pos' if s > 0 else 'neg',
                         mdot=mdot, tau=tau,
                         phi=phi_rel, om=om, omd=omd, resid=resid,
                         reg=reg, d_ideal=d_ideal, arms=arms,
                         model=s * raw[sl]))
    print(f"  loaded {case}/{axname}", flush=True)

print(f"\n{len(rows)} of {n_bags} bags survive:")
for k, v in drop.items():
    if v:
        print(f"    -{v:3d}  {k.replace('_', ' ')}")
print()

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
             axis=np.array([r['ax'] for r in rows]),
             bag=np.array([r['bag'] for r in rows]),
             tip=np.array([r['tip'] for r in rows]),
             **{k: np.concatenate([1e3 * r[k] for r in rows])
                for k in ('t_inertia', 't_moment', 't_load', 't_grav_a',
                          't_grav_z')},
             # the arms and forces, so an arm substitution can be tried
             # without re-running the whole pipeline
             lp=np.array([r['lp'] for r in rows]),
             arm_a=np.array([r['arm_a'] for r in rows]),
             Wn=np.array([r['W'] for r in rows]),
             f_col=np.concatenate([r['f'] for r in rows]),
             phi_abs=np.concatenate([r['phi_abs'] for r in rows]))
    print(f"\ndumped -> {os.environ['HD_DUMP']}")
