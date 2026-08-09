"""Could a smaller thrust coefficient close the ground-effect residual?

C_T scales both the applied moment and the collective thrust, so
lowering it raises the inverted dM_GE and its slope.  Two things rule it
out.

FIRST, C_T is not free.  M_crit scales linearly with it, so every
identified CoM offset scales with it too, and the load-cell truth pins
it: the least-squares optimal scale is 0.968, and a 1% change costs
0.018 mm of offset RMS.  Three per cent is about the room available.

SECOND, the residual needs far more than that, and needs a different
amount at every ramp rate:

    C_T scale     all     slow <0.2   mid .2-.6   fast >0.6
       1.00     -46.90      -48.63      -38.59      -55.58
       0.90     -42.96      -46.92      -35.19      -49.27
       0.60     -31.12      -41.80      -24.96      -30.34

A 40% cut recovers a third of the residual.  Extrapolated, the slope
would reach zero at a scale of -1.85 on the slow runs, -0.13 in the
middle and +0.12 on the fast ones.  Negative means that removing the
applied moment entirely would still not be enough.  The three bands
disagree completely, which is what a wrong hypothesis looks like.

The reason is direct: dm/dphi runs from 11 mN.m/deg on the slowest ramps
to 85 on the fastest, and the residual is 47.  On the slow runs the
whole moment term is smaller than what would have to be removed.
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
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
          ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
          ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
          ('case_05','My'):-10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}
CT_SCALES = (1.00, 0.95, 0.90, 0.80, 0.60)

cache = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    ax = 'x' if d.name == 'Mx' else 'y'
    case, axname = d.parent.name, d.name
    mass = MASS[case]; W = mass * G
    c2f, kf = PNLS_CONSTANTS[(case, axname)]
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2f, ramp_gain=kf)
    by = {b.name: b for b in bags}
    for crit in crits:
        bag = by[crit.bag_name]
        s = 1.0 if crit.bag_name.startswith('pos') else -1.0
        roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
        phi_all = roll if ax == 'x' else pitch
        sig = cvp.prepare_signals(bag, ax)
        n = min(len(phi_all), len(sig['t']))
        i0w, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
        j = crit.onset_idx; i1 = min(i1, n - 1)
        if i1 - j < 15:
            continue
        sl = slice(j, i1 + 1)
        tau = sig['t'][sl] - sig['t'][j]
        w = min(9, len(tau) - (1 - len(tau) % 2))
        if w < 5:
            continue
        phi_abs = s * phi_all[sl]
        deg = np.rad2deg(s * (phi_all[sl] - phi_all[j]))
        m = s * sig['moment'][sl]; f = sig['f_col'][sl]
        # full-trace derivative, then slice: slicing first puts the
        # onset on the differentiator's extrapolated edge
        # (analysis/rate_derivative.py)
        dtf = float(np.median(np.diff(sig['t'][:n])))
        om_full = s * sig['omega'][:n]
        om = om_full[sl]; omd = omega_dot(om_full, dtf, w)[sl]
        if not edge_margin(n, j, i1, w)['ok']:
            continue
        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest)
        if raw is None:
            continue
        grav = W * a * np.cos(phi_abs) - W * Z * np.sin(phi_abs)
        j_p = 0.051085 if ax == 'x' else 0.050564
        j_p += mass * (Z ** 2 + LP[ax] ** 2)
        mdot = abs(float(np.polyfit(tau, m, 1)[0]))
        cache.append((ax, j_p, omd, m, f * lp, grav, deg, s * raw[sl], mdot))
    print(f"  loaded {case}/{axname}", flush=True)

RATE_BINS = [(0.0, 0.2, 'slow  <0.2'), (0.2, 0.6, 'mid  0.2-0.6'),
             (0.6, 9.9, 'fast  >0.6')]
print(f"\n{'C_T scale':>10}{'all':>9} | " +
      ' | '.join(f"{lab:>13}" for _, _, lab in RATE_BINS))
print(f"{'':10}{'slope':>9} | " +
      ' | '.join(f"{'slope':>13}" for _ in RATE_BINS))
for e in CT_SCALES:
    ds = []
    per = {lab: [] for _, _, lab in RATE_BINS}
    for ax, j_p, omd, m, flp, grav, deg, gem, mdot in cache:
        ge = j_p * omd - e * m - e * flp + grav
        sd = np.polyfit(deg, 1e3 * ge, 1)[0]
        sm = np.polyfit(deg, 1e3 * gem, 1)[0]
        ds.append(sd - sm)
        for lo, hi, lab in RATE_BINS:
            if lo <= mdot < hi:
                per[lab].append(sd - sm)
    print(f"{e:10.2f}{np.mean(ds):9.2f} | " +
          ' | '.join(f"{np.mean(per[lab]):13.2f}" for _, _, lab in RATE_BINS))
print()
print('required C_T scale to zero the slope, per rate band:')
for lo, hi, lab in RATE_BINS:
    xs, ys = [], []
    for e in CT_SCALES:
        v = []
        for ax, j_p, omd, m, flp, grav, deg, gem, mdot in cache:
            if not (lo <= mdot < hi):
                continue
            ge = j_p * omd - e * m - e * flp + grav
            v.append(np.polyfit(deg, 1e3 * ge, 1)[0]
                     - np.polyfit(deg, 1e3 * gem, 1)[0])
        xs.append(e); ys.append(np.mean(v))
    xs, ys = np.array(xs), np.array(ys)
    o = np.argsort(ys)
    print(f"  {lab:14} -> C_T scale {np.interp(0.0, ys[o], xs[o]):.3f}")
