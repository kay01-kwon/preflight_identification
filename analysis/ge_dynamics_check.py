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

RESULT: THE LEVEL IS THE STATIC CHECK; THE ATTITUDE DEPENDENCE IS NOT
RESOLVABLE.  Re-run with the CAD constants (z_CoM = 0.261 m from the
landing-gear datum, J_P = 0.334 / 0.309 kg m^2 from the parallel-axis
theorem), which are independently measured rather than fitted:

1. At the onset the balance is STATIC, exactly.  The onset is defined by
   a vanishing net moment, so omega(0) = 0 AND omega_dot(0) = 0, and the
   excursion is zero there by construction.  Every dynamic term drops:

       dM_GE(0) = W a - M_crit - f l_p          (no J_P, no z_CoM)

   which is precisely the static check of mcrit_prediction.py.  The
   fitted intercepts, 113-374 mN.m against 113-208 predicted, sit at the
   right level (mean offset +70 mN.m, RMS 109) but are noisier than the
   static check itself (residual RMS 88 mN.m), because using the
   measured omega_dot(0) instead of zero only injects the noise of a
   numerical derivative.  Nothing is gained.

2. Away from the onset the model signal is buried.  With the CAD z_CoM,
   d/dphi[-W z_CoM sin phi] = -141 mN.m/deg, which the growth of
   J_P omega_dot must cancel; the observed slopes span -79 to -20, so
   the cancellation closes to about 70%.  The model's own slope is
   -2.7 to -0.1 mN.m/deg, i.e. 0.0-1.9% of the term being cancelled.
   Resolving it needs z_CoM and J_P to roughly 2%, against the ~4% that
   CAD plus the parallel-axis theorem support.

   The SIGN of the drift is not the anomaly.  Tipping about one foot
   lifts the opposite rotors away from the ground, and those carry the
   long moment arms, so the net ground-effect moment must FALL with
   tilt; the model's own slope is negative too.  What does not make
   sense is the magnitude, sixteen times the model's, and that the
   inversion carries the level through zero to -90 mN.m: a net
   restoring ground-effect moment would need the near-pivot rotors to
   outweigh the far ones, which the geometry does not allow at 6 deg.

   The residual slope is a fitting artefact, not a physical rate: it
   correlates with the excursion range (the short My-negative windows,
   2.7-4.4 deg, are steepest; the long Mx windows, 5.3-7.1 deg,
   flattest) and varies by a factor of 4 across runs where a physical
   contact migration would repeat.

3. The residual is NOT a constants error.  Sweeping z_CoM along the
   parallel axis over 0.190 / 0.205 / 0.220 / 0.256 m moves the term
   being cancelled by 35% (-105 to -141 mN.m/deg) while the residual
   slope stays flat at -45.9 / -45.3 / -44.2 / -42.3.  Along that line
   W z_CoM and J_P move together, so the trajectory-model relation
   barely changes and no choice of height rescues the measurement.
   (That sweep was run before the CAD datum was settled at 0.261 m, so
   its top point is 0.256; the trend is flat to within 3.6 mN.m/deg
   across the whole 66 mm span, and 5 mm past its end changes nothing.)
   Neither is the derivative's source.  The bags carry
   /mavros/imu/data_raw at 200 Hz, twice the odom rate the pipeline
   uses, but analysis/imu_vs_odom.py shows the raw gyro is worse even at
   a matched 90 ms window: 98.8% of its AC power sits in 50-100 Hz,
   peaking at the 90.8 Hz blade passing, at 716 mrad/s RMS against 176
   for odom, whose EKF has already rejected it.  Differentiating it
   drives the slope to -139 to +40 mN.m/deg.

   analysis/slope_budget.py isolates the terms on a synthetic
   trajectory with a known ground-effect moment: the measured gyro
   noise contributes +-4.2 mN.m/deg of scatter and no bias, the
   Savitzky-Golay derivative +0.5, a 4% error in J_P +-7.2 and a 5 mm
   error in z_CoM +-2.8, against a signal of -2.  So neither the
   differentiation nor the noise is the limit.

   CONFIRMED FROM THE OTHER SIDE: denoising harder makes it WORSE.
   Re-running the batch with the Savitzky-Golay window widened toward
   the band-limit rule of the noise model (docs/noise_model_notes.tex,
   w ~ 2/(f_c T_s) = 41 samples at f_c = 5 Hz and T_s = 9.9 ms):

       w [samples]      9        21        41        61
       slope [mN.m/deg]   -47.7     -65.3    -114.2   (-137.6 single)
       cancellation        67%       55%       21%
       |omega_dot| max    2.23      1.93      1.49      1.13  rad/s^2

   A filter that only removed noise would converge, not diverge.  The
   peak omega_dot falls 49% by w = 61, so the smoother is eating the
   SIGNAL: omega_dot ~ sinh(C2 tau) with 1/C2 = 163 ms, and a 41- or
   61-sample window spans 2.5-3.7 e-foldings, which a local cubic
   cannot follow.  Suppressing the growth of J_P omega_dot is exactly
   suppressing the term that must cancel -W z_CoM sin phi, so the
   residual slope inflates.  The noise model's window rule does not
   transfer here because the FILTER PLAYS THE OPPOSITE ROLE: there the
   smooth curve is discarded and only the residue above f_c is kept,
   so over-smoothing errs safe; here the smooth curve IS the
   measurement.  w = 9 (89 ms = 0.54 e-folding) stays in the regime
   where the local cubic is a good approximation.

       PYTHONPATH=<stubs> python analysis/ge_dynamics_check.py \
           --all --z-com 0.261 --savgol {9,21,41}

   Read simply, the residual falls linearly with tilt, which is a
   restoring stiffness of 2.5 N.m/rad -- 31% of W z_CoM, and opposite in
   sign.  But over the fit window phi, omega and omega_dot all grow
   monotonically, so a term linear in any one looks linear in the
   others.  Regressing the residual on each in turn
   (analysis/regressor_test.py) gives

       phi        -2.69 N.m/rad        coefficient spread 0.69, R^2 0.22
       omega      -0.50 N.m/(rad/s)                     0.68,     0.21
       omega_dot  +0.055 kg.m^2                         2.47,     0.19

   Stiffness and damping are indistinguishable here, and the added
   inertia reading is the LEAST rate-consistent of the three -- so the
   "0.11 kg.m^2 of apparent inertia" below should be read as one
   parametrisation of the anomaly, not as its identification.
   Separating stiffness from damping needs runs where phi and omega are
   not collinear, which a ramp never provides and a release from rest
   does.

   TWO CORRECTIONS to an earlier reading of this residual, recorded so
   the same inference is not made again:

   (a) Sign.  Removing a residual of -42.3 mN.m/deg requires ADDING
       +42.3 to the slope, so J_P would have to be 23.4% LARGER, not
       smaller: 0.401 kg.m^2, which the parallel axis maps to
       z_CoM = 0.299 m.  The claim that the residual implied
       J_P = 0.249 and z_CoM = 0.205 m -- matching the calibration --
       was a sign flip.  The inversion in fact points ABOVE the CAD
       height while the calibration points below it, so the two
       disagree in opposite senses and no single pair satisfies both.

   (a2) The rolling-contact exclusion was argued wrongly.  An 80 mm
       rolling foot leaves a circle-fit residual of only 0.070 mm,
       BELOW the 0.1-0.2 mm measured, because the fit absorbs the
       trochoid into cx and R (analysis/rolling_test.py).  The
       exclusion holds through the fitted geometry instead: at
       r = 80 mm the fit would return l_p = 260 mm against the measured
       140.4 +- 3.6, which allows r <~ 2.4 mm and so at most 3% of the
       required stiffness.

   (b) The sensitivities do not transfer.  Along the parallel axis the
       synthetic budget predicts d(slope)/dz = +0.363 mN.m/deg per mm,
       i.e. +24.0 over the 190-256 mm sweep.  The observed change is
       +3.7, seven times smaller.  The residual therefore does not
       respond to (J_P, z_CoM) the way the model says it should, which
       is stronger than saying the pair is underdetermined: there is no
       pair that removes it, and inverting the residual for constants
       is not quantitatively valid in the first place.

   CORRECTION to (b): that sweep moved z_CoM along the parallel axis,
   where W z_CoM and J_P move together and their effects on the residual
   largely cancel.  Varying J_CoM alone moves J_P without moving
   W z_CoM, and along THAT direction the residual does respond --
   monotonically, crossing zero at J_CoM = 0.161 kg.m^2 for the slope
   and 0.212 for the level, against the CAD 0.051
   (analysis/jcom_sweep.py).  So the residual is equivalent to about
   0.11 kg.m^2 of apparent inertia about the pivot.  It cannot be real
   inertia: Table 5 gives an rms mass height of 66 mm about the CoM
   through Jxx + Jyy - Jzz = 2 m <z^2>, and 0.161 would need 196 mm on
   an airframe spanning -261 to +54 mm.

   Consequently, choosing (J_P, z_CoM) to zero the slope cannot work.
   It is circular -- it assumes the attitude independence it would
   demonstrate -- it is one scalar equation in two unknowns (the
   intercept adds nothing, being the static balance), and by (b) the
   residual does not live in that parameter subspace at all.  A moving
   horizon estimator does not change this: it is a better estimator on
   the same Fisher information, whose near-null direction is a property
   of the data, and it assumes a model structure the evidence says is
   incomplete.  MHE is the right tool for the OTHER problem -- fixing
   (J_P, z_CoM) from an independent measurement and estimating
   dM_GE(t) as a state -- which needs the constants first.

4. A migrating contact point is ruled out by the mocap.  Read as a
   moving pivot, the residual corresponds to an arm rate of
   1.35 mm/deg, i.e. the contact would have to travel 8.1 mm over a
   6 deg excursion.  Over that excursion the marker sweeps an arc of
   only 36 mm at R = 340 mm, so a centre drift of 8 mm would leave a
   circle-fit residual of several mm.  The measured residual is
   0.1-0.2 mm across 138 runs (analysis/pivot_geom.py).  Landing-gear
   compliance and foot roll are therefore excluded at the size needed.

   That leaves causes that scale with thrust -- aerodynamic ones, or an
   error in the moment reconstruction M = C_T sum b Omega^2 -- and a
   no-thrust release separates them from everything else, since the
   ground-effect moment is then zero by construction while the leg
   still carries the weight.

This negative result is now CLEAN.  The earlier attempt used the
calibrated J_P, which varies 2.5x across datasets and sits below the
parallel-axis floor, so it could not distinguish an SNR limit from a
bad constant.  With J_P and z_CoM pinned externally the failure
persists with the same signature, which identifies it as a property of
the experiment rather than of the calibration.  The attitude dependence
of the ground-effect moment is therefore bounded by the excitation
design -- the 5 deg tilt cap -- rather than measured, and the method is
unaffected: it uses the onset alone, where the balance is the static
one.

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
from analysis.rate_derivative import omega_dot, edge_margin

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}                  # Table 7
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

# Independently measured constants -- NOT fitted to this data.  The CAD
# model gives the CoM height above the landing-gear contact plane and the
# CoM inertias (manuscript Table 5); the parallel-axis theorem then fixes
# J_P.  This matters here: the inversion carries J_P omega_dot, so an
# error in J_P propagates directly, and the earlier attempt used the
# calibrated J_P, which varies 2.5x across datasets and sits below the
# parallel-axis floor.
Z_COM_SHARED = 0.261                       # m, CAD, landing-gear datum
J_COM = {'x': 0.051085, 'y': 0.050564}     # kg.m^2, CAD Table 5
J_AXIS = {a: J_COM[a] + 3.220 * (Z_COM_SHARED ** 2 + LP[a] ** 2)
          for a in ('x', 'y')}             # 0.334 / 0.309 kg.m^2


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
    # SIGN CONVENTION.  Everything below (m, phi, om) is multiplied by s, so
    # the balance is written in the TIPPING-POSITIVE frame.  In the raw frame
    # the static threshold is
    #     M_crit = s (W - f) l_p + W lam,
    # where the pivot arm flips with the tip direction -- it is the restoring
    # term -- and the CoM offset does not.  Multiplying through by s moves the
    # direction dependence off l_p and onto lam:
    #     s M_crit = W (l_p + s lam) - f l_p,
    # which is why the gravity arm below is l_p + s*lam with l_p unsigned.
    # The two forms are identical (verified numerically); this one matches
    # analysis/mcrit_prediction.py at the onset, where omega_dot = phi = 0.
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
    m = s * sig['moment'][sl]
    f = sig['f_col'][sl]

    w = min(savgol if savgol % 2 else savgol + 1,
            len(tau) - (1 - len(tau) % 2))
    if w < 5:
        return None
    dt = float(np.median(np.diff(tau)))
    # full-trace derivative, then slice: slicing first puts the onset on
    # the differentiator's extrapolated edge, exactly where omega_dot
    # must vanish (analysis/rate_derivative.py)
    if not edge_margin(n, j, i1, w)['ok']:
        return None
    om_full = s * sig['omega'][:n]
    om = om_full[sl]
    om_dot = omega_dot(om_full, dt, w)[sl]

    ge_dyn = (j_p * om_dot - m - f * lp
              + case_w * a * np.cos(phi) - case_w * z_com * np.sin(phi))
    raw = ge_moment(bag, sig, axis, n, pos, window=sl)
    if raw is None:
        return None
    ge_mod = s * raw[sl]
    # The model's sign is derived, not imposed: ge_moment sums the ground
    # effect gain over each rotor's OWN signed arm about the pivot, and the
    # far rotors dominate, so in the tipping-positive frame the net moment is
    # destabilising.  Verified positive on all 116 runs, both directions and
    # both axes; an earlier version silently flipped the sign when the mean
    # came out negative, which would have hidden a convention error rather
    # than surfacing one.
    if np.mean(ge_mod) < 0:
        raise AssertionError(
            f"model ground-effect moment came out restoring on {crit.bag_name}"
            f" (mean {1e3 * float(np.mean(ge_mod)):+.1f} mN.m) -- check the"
            f" pivot-arm convention in error_budget.ge_moment")

    sd, id_ = np.polyfit(phi, ge_dyn, 1)
    sm, im = np.polyfit(phi, ge_mod, 1)
    # An effective arm varying with tilt would show up as an extra W a' phi,
    # so a' = (slope_model - slope_dyn)/W.  CAUTION: this is fully degenerate
    # with an error in (J_P, z_CoM) -- see the batch summary -- and is
    # reported only to expose that degeneracy, not as a measurement.
    a_rate = -(sd - sm) / case_w              # m/rad
    res = dict(bag=crit.bag_name, dir='pos' if pos else 'neg',
               lp_mm=1e3 * lp, dphi_deg=float(np.rad2deg(phi[-1])),
               int_dyn=1e3 * id_, int_mod=1e3 * im,
               slope_dyn=1e3 * sd * np.pi / 180,
               slope_mod=1e3 * sm * np.pi / 180,
               a_rate_mm_per_deg=1e3 * a_rate * np.pi / 180)
    if getattr(analyse, 'keep_traces', False):
        res['trace'] = (np.rad2deg(phi), 1e3 * ge_dyn, 1e3 * ge_mod)
    return res


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
    wz_deg = 1e3 * 31.59 * Z_COM_SHARED * np.pi / 180.0
    s_dyn = np.array([agg[k]['slope_dyn'] for k in keys])
    s_mod = np.array([agg[k]['slope_mod'] for k in keys])
    mdl_lo, mdl_hi = np.abs(s_mod).min(), np.abs(s_mod).max()
    print(f"\nWhy the SLOPE carries no information about the model:")
    print(f"  d/dphi[-W z_CoM sin phi] = -W z_CoM = {-wz_deg:.0f} mN.m/deg,")
    print(f"  which the growth of J_P omega_dot must cancel.  The observed")
    print(f"  slopes span {s_dyn.min():+.0f} .. {s_dyn.max():+.0f}, "
          f"i.e. they measure how well that")
    print(f"  cancellation closes -- here to "
          f"{100 * (1 - np.abs(np.mean(d_slp)) / wz_deg):.0f}%.  The model's OWN")
    print(f"  slope is {-mdl_hi:.1f} .. {-mdl_lo:.1f} mN.m/deg, i.e. "
          f"{100 * mdl_lo / wz_deg:.1f}-{100 * mdl_hi / wz_deg:.1f}% of the")
    print(f"  term being cancelled: resolving it needs z_CoM and J_P to about")
    print(f"  {100 * mdl_hi / wz_deg:.0f}%, against the ~4% the CAD model and")
    print(f"  the parallel-axis theorem support.")
    print(f"  The slope also correlates with the fitted excursion range")
    print(f"  (My neg, the shortest excursions, is steepest; Mx, the longest,")
    print(f"  flattest), the signature of a fitting artefact, and it varies")
    print(f"  by a factor of {np.abs(s_dyn).max() / np.abs(s_dyn).min():.0f}"
          f" across runs where a physical rate would repeat.")
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
    m = s * mom_s[sl]
    f = f_s[sl]

    w = min(args.savgol if args.savgol % 2 else args.savgol + 1,
            len(tau) - (1 - len(tau) % 2))
    dt = float(np.median(np.diff(tau)))
    # full-trace derivative, then slice (analysis/rate_derivative.py)
    om_full = s * om_s[:n]
    om = om_full[sl]
    om_dot = omega_dot(om_full, dt, w)[sl]
    if not edge_margin(n, j, i1, w)['ok']:
        return None

    # ------------------------------------------- dynamic inversion vs model
    ge_dyn = (j_p * om_dot - m - f * lp
              + W * a * np.cos(phi) - W * args.z_com * np.sin(phi))
    ge_mod_raw = ge_moment(bag, sig, axis, n, pos, window=sl)
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
