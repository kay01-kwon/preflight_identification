"""How much apparent inertia would close the ground-effect residual?

An earlier sweep varied z_CoM along the parallel axis and found the
residual slope almost unmoved, which was read as "no choice of
(J_P, z_CoM) removes it".  That reading was too broad.  Along the
parallel axis z_CoM moves W z_CoM and J_P together and their effects
largely cancel; varying J_CoM alone moves J_P WITHOUT moving W z_CoM,
and along that direction the residual responds strongly and
monotonically:

    J_CoM    J_P roll   slope resid   level resid
    0.051     0.3335        -46.84         -50.7
    0.080     0.3625        -34.49         -41.6
    0.120     0.4025        -17.45         -29.0
    0.200     0.4825        +16.63          -3.8
    0.300     0.5825        +59.22         +27.7

The slope crosses zero at J_CoM = 0.161 kg m^2 and the level at 0.212,
against the CAD value of 0.051 -- an extra 0.11 kg m^2 of apparent
inertia about the pivot.

It cannot be real inertia.  Table 5 fixes the mass distribution through
Jxx + Jyy - Jzz = 2 m <z^2>: the CAD inertias imply an rms mass height
of 66 mm about the CoM, 0.161 would need 196 mm, and the airframe only
spans -261 mm (feet) to +54 mm (rotor plane), so 196 mm would require
essentially all the mass at the two extremes.

So the residual is equivalent to about 0.11 kg m^2 of apparent inertia
that the mass distribution cannot supply.  That is a sharper statement
than "something unmodelled": it says how much, and in which term.  Air
added mass is roughly 0.008 by a wake-column estimate, an order too
small.

None of this reaches the deliverable, which reads the onset alone, where
omega_dot vanishes and J_P leaves the balance entirely.
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

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G, Z = 9.81, 0.261
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
          ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
          ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
          ('case_05','My'):-10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}
JCOMS = (0.051, 0.08, 0.12, 0.20, 0.30)

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
        om = s * sig['omega'][sl]; m = s * sig['moment'][sl]; f = sig['f_col'][sl]
        omd = savgol_filter(om, w, 2, deriv=1, delta=float(np.median(np.diff(tau))))
        piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
        lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[ax]
        a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
        q_rest = q_rest / np.linalg.norm(q_rest)
        raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest)
        if raw is None:
            continue
        base = -m - f * lp + W * a * np.cos(phi_abs) - W * Z * np.sin(phi_abs)
        cache.append((ax, mass, omd, base, deg, s * raw[sl]))
    print(f"  loaded {case}/{axname}", flush=True)

print(f"\n{'J_CoM':>7}{'J_P roll':>10}{'J_P pitch':>11}"
      f"{'slope resid':>13}{'level resid':>13}{'slope RMS':>11}")
for jc in JCOMS:
    ds, di = [], []
    for ax, mass, omd, base, deg, gem in cache:
        j_p = jc + mass * (Z ** 2 + LP[ax] ** 2)
        sd, id_ = np.polyfit(deg, 1e3 * (j_p * omd + base), 1)
        sm, im = np.polyfit(deg, 1e3 * gem, 1)
        ds.append(sd - sm); di.append(id_ - im)
    ds, di = np.array(ds), np.array(di)
    jr = jc + 3.220 * (Z ** 2 + LP['x'] ** 2)
    jp_ = jc + 3.220 * (Z ** 2 + LP['y'] ** 2)
    print(f"{jc:7.3f}{jr:10.4f}{jp_:11.4f}{ds.mean():13.2f}{di.mean():13.1f}"
          f"{np.sqrt((ds**2).mean()):11.2f}")
