#!/usr/bin/env python3
"""Empirical upper bound on the landing-gear compliance.

Below the tip-over threshold the rigid-pivot model predicts zero
rotation, so any systematic body rate while the ramp loads the gear is
gear (and structure) wind-up: with a compliance slope s = dphi/dM the
ramp drives a rate omega_gear = s * Mdot. Per run, s is estimated as

    s = (median omega over the sub-threshold ramp segment
         (15%..75% of the onset moment)  -  rest-bias median) / Mdot,

with the rest bias taken before the ramp starts. The campaign 95th
percentile of |s| is the reported compliance bound; from it follow
the stiffness lower bound k_gear = 1/s, the wind-up at break-away
s * M_crit, and the in-window rate contribution s * Mdot at the
fastest ramp -- the number to compare against the measured residual
floor and the vibration term of the RMSE bound.

Usage
-----
  PYTHONPATH=<stubs> python analysis/gear_compliance.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import critical_value_getter_piecewise as cvp          # noqa: E402
from utils.extractor import load_excitation_dataset    # noqa: E402
from analysis.pnls_constants import PNLS_CONSTANTS     # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'

slopes, mcrits = [], []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2_pn, k_pn = PNLS_CONSTANTS[(d.parent.name, d.name)]
        crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2_pn,
                                               ramp_gain=k_pn)
    by_bag = {b.name: b for b in bags}
    for c in crits:
        bag = by_bag[c.bag_name]
        mdot = cvp.commanded_ramp_rate(bag.name)
        if mdot is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            sig = cvp.prepare_signals(bag, axis)
        t, om, M = sig['t'], sig['omega'], sig['moment']
        i0, i1 = cvp.detect_excitation_window(
            M, moment_cap=cvp.MOMENT_CAP.get(axis))
        j_on = int(np.searchsorted(t, c.onset_time))
        # rest bias: up to 1 s of data just before the ramp starts
        rest = om[np.flatnonzero(t < t[i0] - 0.05)][-200:]
        if len(rest) < 20:
            continue
        b = float(np.median(rest))
        m_on = abs(float(c.onset_moment))
        seg = np.flatnonzero(
            (np.arange(len(t)) >= i0) & (np.arange(len(t)) < j_on)
            & (np.abs(M) >= 0.15 * m_on) & (np.abs(M) <= 0.75 * m_on))
        if len(seg) < 10:
            continue
        sgn = np.sign(float(np.median(M[seg])))
        s = sgn * (float(np.median(om[seg])) - b) / mdot   # rad/(N·m)
        slopes.append(s)
        mcrits.append(m_on)

s = np.array(slopes)
a = np.degrees(np.abs(s))                      # deg per N·m
q50, q95 = np.percentile(a, [50, 95])
mc = float(np.median(mcrits))
print(f"n = {len(s)} runs")
print(f"compliance slope |s| = dphi/dM: median {1e3*q50:.1f}, "
      f"95th pct {1e3*q95:.1f} mdeg/(N·m)  "
      f"(signed median {1e3*np.degrees(np.median(s)):+.1f})")
print(f"stiffness lower bound k_gear = 1/s95 = "
      f"{1.0/np.radians(q95):.0f} N·m/rad")
print(f"wind-up at break-away s95 * median|M_crit| ({mc:.2f} N·m): "
      f"{1e3*q95*mc:.1f} mdeg")
print(f"in-window rate contribution s95 * Mdot at 1.2 N·m/s: "
      f"{q95*1.2:.3f} deg/s")
print(f"  vs measured residual floor ~0.87 deg/s and the vibration "
      f"term 2.47 deg/s of the RMSE bound")
