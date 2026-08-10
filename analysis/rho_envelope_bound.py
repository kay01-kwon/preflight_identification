#!/usr/bin/env python3
"""What an unmodelled moment of a given envelope can do, whatever its shape.

analysis/ge_worst_case_shape.py bounds the ground-effect channel by
taking the model's own amplitude and putting all of it into the one
shape the estimator cannot absorb.  That still borrows the amplitude
from a model.  This borrows nothing: fix an envelope
|rho(tau)| <= B and ask for the largest onset error ANY perturbation
inside it can produce.

The question has an exact answer, because every step from rho to the
identified offset is linear:

    domega = D rho,        D = the Duhamel operator with the cosh kernel
    dM_crit = -<domega, sinh(C2 tau)> / (K C2 ||sinh||^2)
            = <w, rho>,    w = -D^T sinh / (K C2 ||sinh||^2)

and the estimator absorbs its own span {1, tau, dphi} exactly, so only
P rho reaches the onset, P being the orthogonal projection off that
span.  Then

    dM_crit = <w, P x> = <P w, x>,     sup   = B ||P w||_1
                                    |x|_inf <= B

attained by the bang-bang perturbation x = B sgn(P w).  No smoothness,
no shape, no model: this is what the envelope alone permits.

That worst case is deliberately aligned with the kernel -- it never
changes sign except where the kernel does.  A perturbation that
wiggles does far less, because the Duhamel integral smooths it and the
correlation against sinh then cancels most of what is left; the table
below reports a few oscillation rates for scale.  Wiggling does not
threaten the cosh family: the propagation and the onset linearisation
are exact for any rho, and roughness only makes the bound looser than
it needs to be.  What would threaten the family is a forcing alive AT
the onset, which the onset conditions exclude (Sec. fid-transfer).

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/rho_envelope_bound.py hd.npz [envelope_mNm]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.error_budget import LP
from analysis.pnls_constants import PNLS_CONSTANTS

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('hd.npz')
B = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0        # mN.m
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
J_CAD, Z, G = dict(x=0.0537, y=0.0537), 0.261, 9.80665
DT = 0.01
FWIG = (0.0, 1.0, 2.0, 5.0)          # Hz; 0 = the bang-bang worst case

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
n = len(d['mdot'])

out = {f: [] for f in FWIG}
lens = []

for i in range(n):
    s = np.flatnonzero(rid == i)
    if len(s) < 20:
        continue
    case, axname = str(d['case'][i]), str(d['axis'][i])
    ax = 'x' if axname == 'Mx' else 'y'
    c2, k_ramp = PNLS_CONSTANTS[(case, axname)]
    j_p = J_CAD[ax] + MASS[case] * (Z ** 2 + LP[ax] ** 2)
    m = len(s)
    tau = np.arange(m) * DT
    dphi = np.deg2rad(phi[s] - phi[s][0])
    A = np.vstack([np.ones_like(tau), tau, dphi]).T
    sh = np.sinh(np.clip(c2 * tau, 0, 30))
    denom = k_ramp * c2 * float(sh @ sh)
    if not denom:
        continue
    lens.append(m * DT)

    # D as a matrix: the same trapezoidal quadrature of the same kernel
    # analysis/error_budget.duhamel applies, written out so its adjoint
    # is available.  Row 0 is zero (the integral is empty there).
    ti = tau[:, None] - tau[None, :]
    D = np.cosh(np.clip(c2 * ti, 0, 30)) * (ti >= 0) * (DT / j_p)
    D[np.tril_indices(m)] = D[np.tril_indices(m)]        # keep lower part
    D = np.tril(D)
    D[:, 0] *= 0.5
    for r in range(1, m):
        D[r, r] *= 0.5
    D[0, :] = 0.0

    w = -(D.T @ sh) / denom                              # dM = <w, rho>
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    pw = w - A @ coef                                    # P w, P symmetric

    for f in FWIG:
        if f == 0.0:
            x = np.sign(pw)                              # bang-bang optimum
        else:
            x = np.sign(np.sin(2 * np.pi * f * tau)) * np.sign(pw)
        dM = float(pw @ (B * 1e-3 * x))
        out[f].append(abs(dM) / (MASS[case] * G) * 1e3)  # mm

print(f"{len(lens)} runs, windows {np.median(lens):.1f} s median"
      f"  ({min(lens):.1f}-{max(lens):.1f})\n")
print(f"an unmodelled moment with |rho| <= {B:.0f} mN.m, no shape assumed\n")
print(f"  {'perturbation':34}{'median':>10}{'95th':>9}{'max':>9}")
print(f"  {'':34}{'[mm]':>10}{'[mm]':>9}{'[mm]':>9}")
for f in FWIG:
    v = np.array(out[f])
    lab = ('worst case (bang-bang)' if f == 0.0
           else f'the same envelope, sign flips at {f:g} Hz')
    print(f"  {lab:34}{np.median(v):10.4f}{np.percentile(v, 95):9.4f}"
          f"{v.max():9.4f}")
v0 = np.array(out[0.0])
print(f"\n  against a validation RMS of 1.64 mm, the worst case leaves a"
      f" factor of\n  {1.64 / np.median(v0):.1f} at the median and"
      f" {1.64 / v0.max():.1f} at the worst run.")
print(f"  the envelope that would reach 0.1 mm is"
      f" {0.1 / np.median(v0) * B:.0f} mN.m (median run),"
      f" {0.1 / v0.max() * B:.0f} (worst).")
print(f"  the envelope that would reach the validation RMS is"
      f" {1.64 / np.median(v0) * B:.0f} mN.m (median),"
      f" {1.64 / v0.max() * B:.0f} (worst).")
