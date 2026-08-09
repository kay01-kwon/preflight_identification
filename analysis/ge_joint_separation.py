#!/usr/bin/env python3
"""Separate the GE attitude gradient from rotor heave damping.

Every regression so far put phi, omega and omega_dot in ONE AT A TIME,
and over a single ramp window the three are collinear, so each absorbed
the others.  They are not collinear ACROSS ramp rates, and that is the
one lever this dataset has.

For the cosh trajectory phi ~ (1/6) C1 C2^2 tau^3 and omega ~ (1/2) C1
C2^2 tau^2, so omega = 3 phi / tau.  Reaching a given phi takes
tau ~ Mdot^(-1/3), so over the twelvefold rate range omega at FIXED phi
varies by 12^(1/3) = 2.3x.  A term proportional to phi and a term
proportional to omega are therefore distinguishable -- but only if they
are fitted together, and only if the between-run offsets are removed
first (they are nuisance parameters, one per run).

Model, per sample, with a free intercept per run:

    resid = a * phi + b * omega + c_run

  a  [mN.m/deg]      the attitude gradient -- what the GE model predicts
                     to be -2.5, and what we are trying to see
  b  [mN.m/(rad/s)]  rotor heave damping -- momentum theory predicts
                     -477, and it is 11-33x larger than the GE term
                     over this window, which is why it has to be in the
                     regression rather than left in the residual

Confidence intervals are bootstrapped over RUNS, not samples: samples
within a run are strongly correlated, so a sample bootstrap would
understate the interval by roughly the square root of the run length.

Usage: python analysis/ge_joint_separation.py hd.npz [n_boot]
"""
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else 'hd.npz'
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
GE_MODEL_SLOPE = -2.5          # mN.m/deg, image superposition
D_MOMENTUM = -477.0            # mN.m/(rad/s), momentum theory, eta = 1

d = np.load(SRC)
rid, phi, om, resid = d['rid'], d['phi'], d['om'], d['resid']
mdot = d['mdot']
runs = np.unique(rid)


def demean_by_run(x, ids, sel):
    """Remove each run's own mean -- the per-run intercepts, profiled out."""
    out = x.astype(float).copy()
    for i in sel:
        m = ids == i
        out[m] -= out[m].mean()
    return out


def fit(sel_runs, cols):
    """Least squares of resid on `cols`, with a free intercept per run."""
    m = np.isin(rid, sel_runs)
    y = demean_by_run(resid[m], rid[m], sel_runs)
    A = np.column_stack([demean_by_run(c[m], rid[m], sel_runs)
                         for c in cols])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def report(name, cols, labels, units):
    beta = fit(runs, cols)
    boot = np.empty((N_BOOT, len(cols)))
    rng = np.random.default_rng(0)
    for k in range(N_BOOT):
        boot[k] = fit(rng.choice(runs, len(runs), replace=True), cols)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"\n{name}")
    for j, (lab, un) in enumerate(zip(labels, units)):
        print(f"  {lab:26}{beta[j]:10.2f}   95% CI [{lo[j]:8.2f},"
              f" {hi[j]:8.2f}]   {un}")
    return beta, lo, hi


print(f"{len(runs)} runs, {len(phi)} samples;  "
      f"ramp rates {mdot.min():.2f}-{mdot.max():.2f} N.m/s "
      f"({mdot.max() / mdot.min():.0f}x)")

# how much independent leverage does the rate range actually give?
lev = []
for lo_, hi_ in ((2.5, 3.5), (3.5, 4.5)):
    s = (phi >= lo_) & (phi < hi_)
    if s.sum() > 200:
        q = np.percentile(om[s], [10, 90])
        lev.append((0.5 * (lo_ + hi_), q[0], q[1], q[1] / max(q[0], 1e-9)))
print(f"\nomega spread at fixed phi (the lever that separates the two):")
for c, a, b, r in lev:
    print(f"  phi ~ {c:.1f} deg :  omega p10-p90  {a:.2f}-{b:.2f} rad/s"
          f"   ({r:.1f}x)")
print(f"  collinearity |corr(phi, omega)| after de-meaning: "
      f"{abs(np.corrcoef(demean_by_run(phi, rid, runs), demean_by_run(om, rid, runs))[0, 1]):.3f}")

report("single regressor -- phi only (what was run before):",
       [phi], ['a  (attitude gradient)'], ['mN.m/deg'])
report("single regressor -- omega only:",
       [om], ['b  (heave damping)'], ['mN.m/(rad/s)'])
beta, lo, hi = report("JOINT -- phi and omega together:",
                      [phi, om],
                      ['a  (attitude gradient)', 'b  (heave damping)'],
                      ['mN.m/deg', 'mN.m/(rad/s)'])

print(f"\nGE model predicts a = {GE_MODEL_SLOPE:+.1f} mN.m/deg;"
      f"  momentum theory predicts b = {D_MOMENTUM:+.0f} mN.m/(rad/s)")
a_ok = lo[0] <= GE_MODEL_SLOPE <= hi[0]
b_ok = lo[1] <= D_MOMENTUM <= hi[1]
print(f"  a: model value {'INSIDE' if a_ok else 'OUTSIDE'} the 95% CI"
      f"   (CI half-width {0.5 * (hi[0] - lo[0]):.1f} mN.m/deg,"
      f" i.e. {abs(0.5 * (hi[0] - lo[0]) / GE_MODEL_SLOPE):.1f}x the signal)")
print(f"  b: theory value {'INSIDE' if b_ok else 'OUTSIDE'} the 95% CI")
print("\nA CI half-width comparable to |a| itself would mean the attitude")
print("gradient is measured, not merely bounded.  Wider than a few times")
print("|a| and the lever is too short, whatever the point estimate says.")
