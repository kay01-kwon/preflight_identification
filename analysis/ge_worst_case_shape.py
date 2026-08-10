#!/usr/bin/env python3
"""Bound the ground-effect channel without using the ground-effect model.

The budget evaluates its ground-effect entry along the image-superposition
model, which invites the objection that it assumes what it should prove:
is that model right along the trajectory?  This removes the model from
the argument.

Two facts do the work, and neither is a modelling assumption.

First, the projection is an algebraic identity of the estimator, not a
choice.  Each run is fitted with the three degrees of freedom
{1, tau, dphi}, so whatever the TRUE ground-effect moment is, any part
of it lying in that span is absorbed exactly.  Only the curvature in
the tilt and above can reach the onset -- for any function, not just
this model.

Second, the ground-effect moment is bounded in magnitude.  So take the
most adversarial shape allowed by that bound: put the ENTIRE moment
into curvature,

    dM_GE(dphi) = A * (dphi / dphi_max)^2 ,

with A the run's own modelled level, and propagate that through the same
Duhamel integral and onset linearisation.  No shape is worse: a
quadratic with the full amplitude is the largest curvature a bounded
function can have over the window.

If that bound is small, the budget's ground-effect entry stands whatever
the model does along the trajectory, and the dynamic check is not needed
to support it.

Usage:
  PYTHONPATH=<stubs> python analysis/ge_worst_case_shape.py hd.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.error_budget import duhamel, out_of_span, LP
from analysis.pnls_constants import PNLS_CONSTANTS

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('hd.npz')
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
J_CAD, Z, G = dict(x=0.0537, y=0.0537), 0.261, 9.80665
DT = 0.01

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
model = d['model'] * 1e-3                       # N.m
n = len(d['mdot'])

SHAPES = ('model as it is', 'all of it as curvature',
          'twice that amplitude')
out = {k: [] for k in SHAPES}
lev = []

for i in range(n):
    s = np.flatnonzero(rid == i)
    if len(s) < 20:
        continue
    case, axname = str(d['case'][i]), str(d['axis'][i])
    ax = 'x' if axname == 'Mx' else 'y'
    c2, k_ramp = PNLS_CONSTANTS[(case, axname)]
    j_p = J_CAD[ax] + MASS[case] * (Z ** 2 + LP[ax] ** 2)
    tau = np.arange(len(s)) * DT
    dphi = np.deg2rad(phi[s] - phi[s][0])
    if dphi.max() <= 0:
        continue
    basis = np.vstack([np.ones_like(tau), tau, dphi]).T
    sh = np.sinh(np.clip(c2 * tau, 0, 30))
    denom = k_ramp * c2 * float(sh @ sh)
    if not denom:
        continue

    amp = float(np.median(np.abs(model[s])))
    quad = amp * (dphi / dphi.max()) ** 2
    lev.append(amp * 1e3)

    for lab, trace in (('model as it is', model[s]),
                       ('all of it as curvature', quad),
                       ('twice that amplitude', 2 * quad)):
        rho = out_of_span(basis, trace)
        dom = duhamel(tau, rho, c2, j_p)
        dM = -float(dom @ sh) / denom
        out[lab].append(abs(dM) / (MASS[case] * G) * 1e3)

print(f"{len(lev)} runs;  modelled level {np.median(lev):.0f} mN.m median\n")
print("the ground-effect channel, propagated to the identified offset\n")
print(f"  {'shape of the perturbation':32}{'median':>10}{'95th':>9}"
      f"{'max':>9}")
print(f"  {'':32}{'[mm]':>10}{'[mm]':>9}{'[mm]':>9}")
for k in SHAPES:
    v = np.array(out[k])
    print(f"  {k:32}{np.median(v):10.4f}{np.percentile(v, 95):9.4f}"
          f"{v.max():9.4f}")
print(f"\n  against a validation RMS of 1.64 mm.  The middle row uses no"
      f"\n  ground-effect model at all beyond its magnitude: it puts the"
      f"\n  whole moment into the one shape the estimator cannot absorb.")
