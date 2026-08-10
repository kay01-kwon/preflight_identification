"""Is the residual a stiffness, a damping, or an added inertia?

Over the fit window phi, omega and omega_dot all grow monotonically, so
a term linear in any of them looks linear in the others.  They are
different physics, though, so regress the residual on each and compare
how well it explains the runs -- and, decisively, whether the fitted
coefficient is CONSISTENT across ramp rates.  A real physical
coefficient must be; a proxy for the wrong variable will not be.
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

rows = []
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
        phi_rel = s * (phi_all[sl] - phi_all[j])
        m = s * sig['moment'][sl]; f = sig['f_col'][sl]
        # full-trace derivative, then slice (analysis/rate_derivative.py)
        dt = float(np.median(np.diff(sig['t'][:n])))
        om_full = s * sig['omega'][:n]
        om = om_full[sl]; omd = omega_dot(om_full, dt, w)[sl]
        if not edge_margin(n, j, i1, w)['ok']:
            continue
        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest,
                        window=sl)
        if raw is None:
            continue
        ge = (j_p * omd - m - f * lp + W * a * np.cos(phi_abs)
              - W * Z * np.sin(phi_abs))
        resid = ge - s * raw[sl]              # inversion minus model, N.m
        mdot = abs(float(np.polyfit(tau, m, 1)[0]))
        rows.append((mdot, phi_rel, om, omd, resid))
    print(f"  loaded {case}/{axname}", flush=True)

print(f"\n{len(rows)} runs\n")
NAMES = [('phi   (stiffness)', 1, 'N.m/rad'),
         ('omega (damping)', 2, 'N.m/(rad/s)'),
         ('omega_dot (inertia)', 3, 'kg.m^2')]
BINS = [(0.0, 0.2, 'slow'), (0.2, 0.6, 'mid'), (0.6, 9.9, 'fast')]
print(f"{'regressor':<22}{'unit':>13}{'all':>10}" +
      ''.join(f"{lab:>9}" for _, _, lab in BINS) + f"{'spread':>9}{'R^2':>7}")
for name, idx, unit in NAMES:
    coefs, r2s, per = [], [], {lab: [] for _, _, lab in BINS}
    for r in rows:
        x = r[idx]; y = r[4]
        A = np.column_stack([x, np.ones_like(x)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ c
        ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-18)
        coefs.append(c[0]); r2s.append(ss)
        for lo, hi, lab in BINS:
            if lo <= r[0] < hi:
                per[lab].append(c[0])
    coefs = np.array(coefs)
    mu = [np.mean(per[lab]) for _, _, lab in BINS]
    print(f"{name:<22}{unit:>13}{np.mean(coefs):10.3f}" +
          ''.join(f"{v:9.3f}" for v in mu) +
          f"{max(mu)/min(mu) if min(mu) != 0 else np.inf:9.2f}"
          f"{np.mean(r2s):7.2f}")
print("\n'spread' = fast/slow ratio of the fitted coefficient.")
print("A real physical coefficient should be near 1.")
