"""Which attitude does each term see: absolute or relative?

The ground is not level.  Measured before the ramp, where the vehicle is
simply sitting there, the resting attitude is +0.5 deg in roll and
-1.5 deg in pitch, and it agrees between the two tip directions to
within 0.13 deg -- genuine ground slope, not a per-run artefact.

The two terms take opposite references, and the inversion originally had
both wrong:

  GRAVITY acts along true vertical, so it takes the ABSOLUTE attitude.
  Referring it to the resting plane is wrong by W z_CoM sin(phi_0), up
  to 260 mN.m at 1.8 deg -- larger than the whole ground-effect moment.
  Correcting it removes the level bias: the inversion sits +87.5 mN.m
  above the model with the relative angle and -16.1 with the absolute
  one.

  GROUND EFFECT is set by rotor height above the GROUND, so it takes the
  attitude referred to the resting plane.  This one barely matters:
  taking the heights along the ground normal instead of world z moves
  the model slope from -1.85 to -1.83 mN.m/deg and its level from 160.4
  to 160.1.  A uniform ground tilt raises some rotors and lowers others,
  and the image superposition sums over rotor PAIRS, so most of it
  cancels.

Neither reference touches the residual slope: -46.50 relative, -47.13
absolute, against a model slope near -1.8.

A second thing falls out.  Subtracting the resting attitude from the
onset attitude leaves the rotation already made before the onset, and
that rocking is direction-dependent as it must be: +-0.3 deg on roll but
+1.0 to +1.8 deg on pitch.  During it the vehicle transfers from several
feet onto one edge, so the pivot is not yet the single line the cosh
family assumes.  Pitch is the axis that is worse on every other measure
too, and this is a candidate cause.

None of this reaches the deliverable: cosh_onset_fit consumes omega
alone, and attitude enters the pipeline only through the tilt-cap gate.
"""
import contextlib, io, sys
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
J_CAD = {'x': 0.051085, 'y': 0.050564}; LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
          ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
          ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
          ('case_05','My'):-10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

print(f"{'case':<9}{'ax':<4}{'dir':<5}{'phi0':>7} | "
      f"{'RELATIVE (current)':>22} | {'ABSOLUTE gravity':>22} | {'model':>14}")
print(f"{'':18}{'[deg]':>7} | {'slope':>10}{'level':>12} | "
      f"{'slope':>10}{'level':>12} | {'slope':>7}{'level':>7}")
print('-' * 104)
agg = {}
for d in sorted(ROOT.glob('case_*/M[xy]')):
    ax = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    mass = MASS[case]; W = mass * G
    j_p = J_CAD[ax] + mass * (Z ** 2 + LP[ax] ** 2)
    c2f, kf = PNLS_CONSTANTS[(case, axname)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2f, ramp_gain=kf)
    by = {b.name: b for b in bags}
    res = {'pos': [], 'neg': []}
    for crit in crits:
        bag = by[crit.bag_name]
        s = 1.0 if crit.bag_name.startswith('pos') else -1.0
        roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
        phi_all = roll if ax == 'x' else pitch
        sig = cvp.prepare_signals(bag, ax)
        n = min(len(phi_all), len(sig['t']))
        _, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
        j = crit.onset_idx; i1 = min(i1, n - 1)
        if i1 - j < 15:
            continue
        sl = slice(j, i1 + 1)
        tau = sig['t'][sl] - sig['t'][j]
        phi_abs = s * phi_all[sl]                    # referred to vertical
        phi_rel = s * (phi_all[sl] - phi_all[j])     # referred to the ground
        phi0 = s * phi_all[j]
        m = s * sig['moment'][sl]; f = sig['f_col'][sl]
        dt = float(np.median(np.diff(tau)))
        w = min(9, len(tau) - (1 - len(tau) % 2))
        if w < 5:
            continue
        # full-trace derivative, then slice: slicing first puts the
        # onset on the differentiator's extrapolated edge
        # (analysis/rate_derivative.py)
        om_full = s * sig['omega'][:n]
        om = om_full[sl]
        omd = omega_dot(om_full, float(np.median(np.diff(
            sig['t'][:n]))), w)[sl]
        if not edge_margin(n, j, i1, w)['ok']:
            continue
        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        i0w, _ = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
        q_rest = np.median(bag.odom.quaternion[:max(20, i0w)], axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0)
        raw_g = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest)
        if raw is None or raw_g is None:
            continue
        gem = s * raw[sl]; gem_g = s * raw_g[sl]
        deg = np.rad2deg(phi_rel)
        out = {}
        for lab, ph in (('rel', phi_rel), ('abs', phi_abs)):
            ge = (j_p * omd - m - f * lp + W * a * np.cos(ph)
                  - W * Z * np.sin(ph))
            out[lab] = np.polyfit(deg, 1e3 * ge, 1)
        sm, im = np.polyfit(deg, 1e3 * gem, 1)
        smg, img = np.polyfit(deg, 1e3 * gem_g, 1)
        res['pos' if s > 0 else 'neg'].append(
            (np.degrees(phi0), out['rel'][0], out['rel'][1],
             out['abs'][0], out['abs'][1], sm, im, smg, img))
    for dirn in ('neg', 'pos'):
        v = np.array(res[dirn])
        if not len(v):
            continue
        mu = v.mean(axis=0)
        agg[(case, axname, dirn)] = mu
        print(f"{case:<9}{axname:<4}{dirn:<5}{mu[0]:7.2f} | "
              f"{mu[1]:10.2f}{mu[2]:12.1f} | {mu[3]:10.2f}{mu[4]:12.1f} | "
              f"{mu[5]:7.2f}{mu[6]:7.1f} |{mu[7]:7.2f}{mu[8]:7.1f}")
A = np.array(list(agg.values()))
print('-' * 104)
print(f"{'mean':<18}{A[:,0].mean():7.2f} | {A[:,1].mean():10.2f}{A[:,2].mean():12.1f}"
      f" | {A[:,3].mean():10.2f}{A[:,4].mean():12.1f} | "
      f"{A[:,5].mean():7.2f}{A[:,6].mean():7.1f} |{A[:,7].mean():7.2f}{A[:,8].mean():7.1f}")
print('  model columns: world-vertical heights | ground-normal heights')
for lab, si, ii, mi, mj in (('relative', 1, 2, 5, 6), ('absolute', 3, 4, 5, 6),
                            ('abs vs ground-normal model', 3, 4, 7, 8)):
    ds = A[:, si] - A[:, mi]; di = A[:, ii] - A[:, mj]
    print(f"  {lab:<9} minus model:  slope mean {ds.mean():+7.2f} RMS {np.sqrt((ds**2).mean()):6.2f}"
          f" | level mean {di.mean():+7.1f} RMS {np.sqrt((di**2).mean()):6.1f}")
