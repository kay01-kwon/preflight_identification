#!/usr/bin/env python3
"""Does a signal-faithful differentiator remove the GE anomaly?  No.

Runs the dynamic ground-effect inversion on one run with every
differentiator available: Savitzky-Golay over 90-510 ms at orders 2-5,
and the windowless anchored polynomial of rate_derivative.omega_dot_poly.

Once the polynomial order is high enough to represent sinh(C2 tau)
inside the window (see analysis/sg_derivative_order.py), the fitted
slope stops depending on the differentiator:

    SG, orders 3-5, 90-510 ms      -37 to -43 mN.m/deg
    anchored polynomial, order 4-6 -23 to -30
    image-superposition model       -3.1

The window-driven steepening reported earlier (-42 -> -53 -> -90 at
9/21/41 samples) belongs to the deployed poly=2 alone.  The anomaly
does not: it survives every differentiator, which is what makes the
negative result robust rather than an artefact of differentiation.

Usage: PYTHONPATH=<stubs> python analysis/ge_differentiator_compare.py
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
for w, p in [(9,2),(9,3),(21,3),(41,3),(21,5),(41,5),(51,5)]:
    if w > len(tau): continue
    row(f"SG w={w:<3d} p={p}  ({1e3*w*dt:.0f} ms)",
        omega_dot(om_full, dt, w, p)[sl])
for o in (4, 5, 6):
    row(f"anchored polynomial, order {o} (no window)",
        omega_dot_poly(tau, om_full[sl], order=o)[2])
