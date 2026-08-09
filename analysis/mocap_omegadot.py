#!/usr/bin/env python3
"""Can mocap supply omega_dot better than the differentiated gyro?

The dynamic inversion's noise floor is J_P * sigma(omega_dot), and
omega_dot currently comes from Savitzky-Golay differentiation of the
odometry rate.  The bags also carry /S550/pose, a mocap pose whose
circle fit closes to 0.1-0.2 mm -- so its attitude may be far quieter
at the low frequencies that matter here, even though it needs TWO
differentiations instead of one.

This compares the two sources on the same runs and on the same
quantity.  The metric is the pre-onset standard deviation of
omega_dot, converted into the currency of the question:

    noise floor [mN.m] = J_P * sigma(omega_dot)

against a ground-effect moment of about 165 mN.m and an attitude
gradient of about -2.5 mN.m/deg.  Whichever source wins, the number
says directly whether the attitude dependence is reachable.

Usage: PYTHONPATH=<stubs> python analysis/mocap_omegadot.py [n_runs]
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

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
Z, LP = 0.261, {'x': 0.140, 'y': 0.110}
J_CAD = {'x': 0.051085, 'y': 0.050564}
MASS = 3.220
N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def euler_axis(quat, axis):
    roll, pitch = math_tools.quaternion_to_euler_vectorized(quat)
    return roll if axis == 'x' else pitch


rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    if len(rows) >= N_RUNS:
        break
    ax = 'x' if d.name == 'Mx' else 'y'
    j_p = J_CAD[ax] + MASS * (Z ** 2 + LP[ax] ** 2)
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, ax)
    by = {b.name: b for b in bags}
    for crit in crits:
        if len(rows) >= N_RUNS:
            break
        bag = by[crit.bag_name]
        sig = cvp.prepare_signals(bag, ax)
        t_od = sig['t']
        dt_od = float(np.median(np.diff(t_od)))
        j = crit.onset_idx
        if j < 40:
            continue

        # --- source 1: odom rate, one differentiation ----------------
        om_od = sig['omega']
        omd_od = savgol_filter(om_od, 9, 2, deriv=1, delta=dt_od)

        # --- source 2: mocap attitude, two differentiations ----------
        t_mc = cvp.align_mocap_time(bag)
        dt_mc = float(np.median(np.diff(t_mc)))
        phi_mc = euler_axis(bag.pose.quaternion, ax)
        if len(phi_mc) < 60:
            continue
        # match the odom effective window (9 samples at dt_od) so the
        # two are compared at the same bandwidth, not at whatever each
        # sensor's native rate happens to give
        w_mc = int(round(9 * dt_od / dt_mc))
        w_mc = max(5, w_mc + 1 - w_mc % 2)
        if w_mc >= len(phi_mc):
            continue
        omd_mc = savgol_filter(phi_mc, w_mc, 3, deriv=2, delta=dt_mc)

        # pre-onset window: the vehicle is still, so anything here is
        # noise.  Take it on each source's own time base.
        t_onset = crit.t[j]
        pre_od = omd_od[(t_od < t_onset) & (t_od > t_onset - 1.0)]
        pre_mc = omd_mc[(t_mc < t_onset) & (t_mc > t_onset - 1.0)]
        if len(pre_od) < 15 or len(pre_mc) < 15:
            continue
        rows.append(dict(case=d.parent.name, axis=d.name, bag=crit.bag_name,
                         j_p=j_p, dt_od=dt_od, dt_mc=dt_mc, w_mc=w_mc,
                         s_od=float(np.std(pre_od)),
                         s_mc=float(np.std(pre_mc))))
    print(f"  {d.parent.name}/{d.name}: {len(rows)} runs so far", flush=True)

if not rows:
    raise SystemExit("no runs with both sources")

s_od = np.array([r['s_od'] for r in rows])
s_mc = np.array([r['s_mc'] for r in rows])
j_p = np.array([r['j_p'] for r in rows])
print(f"\n{len(rows)} runs;  odom dt {np.median([r['dt_od'] for r in rows])*1e3:.1f} ms,"
      f"  mocap dt {np.median([r['dt_mc'] for r in rows])*1e3:.1f} ms"
      f"  (mocap window {int(np.median([r['w_mc'] for r in rows]))} samples)")
print(f"\n{'source':28}{'sigma(omega_dot)':>20}{'J_P sigma':>14}")
print(f"{'':28}{'[rad/s^2]':>20}{'[mN.m]':>14}")
for lab, s in (('odom rate, 1 deriv', s_od),
               ('mocap attitude, 2 deriv', s_mc)):
    print(f"{lab:28}{np.median(s):20.3f}{1e3*np.median(s*j_p):14.1f}")
print(f"\nratio mocap/odom: {np.median(s_mc)/np.median(s_od):.2f}")
print("\nReference: the ground-effect moment is about 165 mN.m and its")
print("attitude gradient about -2.5 mN.m/deg over a ~5 deg excursion,")
print("i.e. a 13 mN.m swing.  A noise floor above that cannot see it")
print("in a single run; averaging N runs divides the floor by sqrt(N).")
