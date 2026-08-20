#!/usr/bin/env python3
"""Can excluding the trailing samples rescue the GE dynamic check?  No.

Two separate questions, both answered here.

1. Is the differentiator extrapolating at the window end?  NO.  The
   pipeline differentiates the FULL bag trace and slices afterwards,
   and 335 samples follow the excitation window on this run, so every
   point inside the window has two-sided filter support.

2. Would excluding the last k samples remove the anomaly?  It moves
   it -- -89.7 -> -12.5 mN.m/deg at k = 20 for the 41-sample window --
   but at the cost of the measurement itself.  phi grows
   exponentially, so the trailing samples carry nearly all of the
   attitude range:

       k= 0   79 samples   phi 0 -> 6.93 deg
       k=10   69 samples   phi 0 -> 4.03 deg
       k=20   59 samples   phi 0 -> 2.11 deg
       k=41   38 samples   phi 0 -> 0.82 deg   slope -199.7 (no lever)

   The attitude dependence the check exists to measure lives in the
   tail, so trimming the tail is not a conservative choice: it is
   removing the signal.

Usage: PYTHONPATH=<stubs> python analysis/ge_edge_exclusion.py
"""
import contextlib, io, os, sys
from pathlib import Path
import numpy as np
_R = '/home/user/preflight_identification'
sys.path.insert(0, _R); sys.path.insert(0, _R + '/analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment, LP
from analysis.rate_derivative import omega_dot, omega_dot_poly
from analysis.ge_dynamics_check import MASS_KG, G, OFF_SIGN, OFF_MM, j_parallel

d = Path(_R + '/DataSet/exp/case_02/Mx'); axis = 'x'
case, axname = 'case_02', 'Mx'; W = MASS_KG[case]*G
off_truth = OFF_SIGN[axname]*OFF_MM[(case, axname)]*1e-3
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(d)
    c2, kg = cvp.estimate_rig_constants(bags, axis)
    crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=kg)
name = sorted(c.bag_name for c in crits if c.bag_name.startswith('pos'))[0]
crit = next(c for c in crits if c.bag_name == name)
bag = next(b for b in bags if b.name == name); s = 1.0
piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
lp = piv['pivot_abs']*1e-3 if not np.isnan(piv['pivot_abs']) else LP[axis]
arm = lp + s*off_truth
z = 0.261; j_p = j_parallel(axis, z, MASS_KG[case])
sig = cvp.prepare_signals(bag, axis)
roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
phi_all = roll if axis == 'x' else pitch
n = min(len(phi_all), len(sig['t']))
_, i1 = cvp.detect_excitation_window(sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
j = crit.onset_idx; i1 = min(i1, n-1); sl = slice(j, i1+1)
tau = sig['t'][sl]-sig['t'][j]; phi = s*(phi_all[sl]-phi_all[j])
m = s*sig['moment'][sl]; f = sig['f_col'][sl]
dt = float(np.median(np.diff(tau))); om_full = s*sig['omega'][:n]
ge_mod = s*ge_moment(bag, sig, axis, n, pos=True, window=sl)[sl]
sm, im = np.polyfit(phi, ge_mod, 1)

print(f"\n  {case}/{axname}/{name}, J_P = {j_p:.4f} (parallel), "
      f"window {len(tau)} samples / {tau[-1]:.2f} s")
print(f"  model: slope {1e3*sm*np.pi/180:+.1f} mN.m/deg, "
      f"intercept {1e3*im:.1f} mN.m\n")
print(f"  {'differentiator':<34}{'|om_dot|max':>12}{'slope':>10}{'intercept':>11}")
def row(lbl, od):
    ge = j_p*od - m - f*lp + W*arm*np.cos(phi) - W*z*np.sin(phi)
    sd, idd = np.polyfit(phi, ge, 1)
    print(f"  {lbl:<34}{np.abs(od).max():12.2f}"
          f"{1e3*sd*np.pi/180:10.1f}{1e3*idd:11.1f}")
for w, p in [(9,2),(21,2),(41,2)]:
    if w > len(tau): continue
    row(f"SG w={w:<3d} p={p}  ({1e3*w*dt:.0f} ms)",
        omega_dot(om_full, dt, w, p)[sl])
for o in ():
    row(f"anchored polynomial, order {o} (no window)",
        omega_dot_poly(tau, om_full[sl], order=o)[2])

# ---- how much data exists AFTER the excitation window ends? ----
print(f"\n  bag samples after i1: {n - 1 - i1}  "
      f"(excitation window {i1-j+1} samples, ends at index {i1} of {n})")

# ---- refit the slope with the last k samples excluded ----
print(f"\n  slope [mN.m/deg] with the last k samples dropped from the fit:")
print(f"  {'w':>4}{'half-width':>12}" + "".join(f"{f'k={k}':>9}"
      for k in (0, 5, 10, 20, 41)))
for w in (9, 21, 41):
    od = omega_dot(om_full, dt, w, 2)[sl]
    ge = j_p*od - m - f*lp + W*arm*np.cos(phi) - W*z*np.sin(phi)
    out = []
    for k in (0, 5, 10, 20, 41):
        if len(phi) - k < 12:
            out.append('    --'); continue
        e = len(phi) - k
        sd, _ = np.polyfit(phi[:e], ge[:e], 1)
        out.append(f"{1e3*sd*np.pi/180:9.1f}")
    print(f"  {w:4d}{(w-1)//2:12d}" + "".join(out))
sm2, _ = np.polyfit(phi, ge_mod, 1)
print(f"\n  model slope for reference: {1e3*sm2*np.pi/180:+.1f} mN.m/deg")

print(f"\n  phi range remaining after dropping the last k samples:")
for k in (0, 5, 10, 20, 41):
    if len(phi)-k < 2: continue
    e = len(phi)-k
    print(f"    k={k:<3d} n={e:3d}  phi 0 -> {np.rad2deg(phi[e-1]):.2f} deg"
          f"   (full range {np.rad2deg(phi[-1]):.2f})")
