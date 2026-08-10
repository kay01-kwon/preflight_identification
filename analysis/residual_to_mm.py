#!/usr/bin/env python3
"""Push the MEASURED dynamic residual through the budget, to millimetres.

analysis/ge_budget_robustness.py answers a hypothetical: how wrong would
the modelled ground-effect trace have to be before the identified offset
moved?  This answers the actual question.  Take the residual the dynamic
inversion leaves against that model -- run by run, as measured -- treat
every bit of it as an unmodelled perturbation, and propagate it through
the same three steps the error budget uses:

  1. project out the estimator's own per-run degrees of freedom
     {1, tau, dphi}, leaving rho;
  2. propagate with the Duhamel integral of the deviation dynamics,
     domega = (1/J_P) int cosh(C2 (tau-s)) rho(s) ds;
  3. linearise the onset argmin,
     dM_crit = -<domega, sinh(C2 tau)> / (K C2 ||sinh(C2 tau)||^2),
     and divide by the weight.

The residual is NOT a constant within a run -- projecting the span out
removes 61% of it on Mx and 71% on My, leaving 38.5 and 24.2 mN.m RMS
-- so this is not an argument that it cancels.  It is the statement
that what does not cancel is small once propagated.

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/residual_to_mm.py hd.npz
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
res = d['resid'] * 1e-3                    # N.m
n = len(d['mdot'])

rows = []
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
    basis = np.vstack([np.ones_like(tau), tau, dphi]).T

    rho = out_of_span(basis, res[s])
    dom = duhamel(tau, rho, c2, j_p)
    sh = np.sinh(np.clip(c2 * tau, 0, 30))
    denom = k_ramp * c2 * float(sh @ sh)
    dM = -float(dom @ sh) / denom if denom else np.nan       # N.m
    rows.append((axname, np.sqrt(np.mean(res[s] ** 2)) * 1e3,
                 np.sqrt(np.mean(rho ** 2)) * 1e3, dM * 1e3,
                 dM / (MASS[case] * G) * 1e3))

A = np.array([r[0] for r in rows])
raw = np.array([r[1] for r in rows])
rho_ = np.array([r[2] for r in rows])
dMc = np.array([r[3] for r in rows])
mm = np.array([r[4] for r in rows])

print(f"{len(rows)} runs, residual propagated as an unmodelled "
      f"perturbation\n")
print(f"  {'axis':6}{'n':>5}{'residual RMS':>15}{'after the span':>16}"
      f"{'dM_crit':>11}{'offset':>11}")
print(f"  {'':6}{'':5}{'[mN.m]':>15}{'[mN.m]':>16}{'[mN.m]':>11}{'[mm]':>11}")
for axn in ('Mx', 'My'):
    m = A == axn
    print(f"  {axn:6}{m.sum():5d}{np.median(raw[m]):15.1f}"
          f"{np.median(rho_[m]):16.1f}{np.median(np.abs(dMc[m])):11.3f}"
          f"{np.median(np.abs(mm[m])):11.4f}")
m = np.ones(len(A), bool)
print(f"  {'all':6}{m.sum():5d}{np.median(raw):15.1f}{np.median(rho_):16.1f}"
      f"{np.median(np.abs(dMc)):11.3f}{np.median(np.abs(mm)):11.4f}")
print(f"\n  worst single run: {np.max(np.abs(mm)):.4f} mm"
      f"   (95th percentile {np.percentile(np.abs(mm), 95):.4f})")
print(f"  the identification averages the two tip directions and the runs"
      f" within a\n  group, so the figure that matters is the median:"
      f" {np.median(np.abs(mm)):.4f} mm,\n  against a validation RMS of"
      f" 1.64 mm.")
