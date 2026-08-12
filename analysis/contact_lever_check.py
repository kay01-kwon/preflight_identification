#!/usr/bin/env python3
"""Which lever does the static threshold actually need, and is it the one we have?

The per-direction threshold check over-predicts every group, and the
deficit is degenerate between a ground-effect moment and a contact-lever
displacement (analysis/static_attribution.py).  This script asks what the
geometry can say about the lever itself, and settles which part of any
lever error reaches the delivered offset.

Four checks, in order of what they rule out.

  1. THE CAD FOOTPRINT.  The skid measures 286.29 mm across and 250 mm
     fore-aft, so the half-widths are 143.14 mm (roll) and 125.00 mm
     (pitch).  The mocap circle fit returns 140.5 and 112.7.  Roll agrees
     to 2.7 mm; pitch is 12.3 mm inboard of the CAD edge, which is what a
     skid whose rail ends curve up should do.  Substituting the CAD
     half-widths makes the deficit WORSE -- pitch 69 to 173 mN.m -- so
     "the code uses the wrong nominal arm" is not the explanation, and
     the per-run mocap arm is the right one to use.

  2. A ROLLING FOOT.  A round foot of radius r rolls as the vehicle tips,
     so the contact migrates outboard and a circle fit over the arc is
     biased in the same direction -- which is the sign the deficit needs.
     The size is not: 0.44 mm at r = 5 mm and 0.87 at r = 10, against the
     ~19 mm required.  Rolling alone is an order of magnitude short.

  3. AN ELEVATED ROTATION CENTRE.  If the body rotated about a point c
     above the ground -- the reading a compliant leg would suggest -- a
     fit that pins the centre to the ground would return a biased arm.
     It would also return a LARGER one: c = 25 mm gives l_p = 154 mm
     against the 140.4 measured.  The measurement bounds c below ~10 mm,
     so the kinematic centre is at the foot and that reading is dead.
     Note that sqrt(R^2 - l_p^2) is insensitive to c and cannot be used
     for this; only l_p itself can.

  4. WHAT REACHES THE DELIVERABLE.  With M_+ = (W-f) l_+ + S_off and
     M_- = -(W-f) l_- + S_off, the pivot-free average is

         M_ff = (W-f)(l_+ - l_-)/2 + S_off ,

     so a lever error COMMON to both directions cancels identically and
     only its direction asymmetry survives.  That is the whole reason a
     19 mm lever discrepancy coexists with a 1.6 mm offset accuracy.
     The script reports both parts.

Reads mcrit_prediction.csv from argv[1].
"""
import csv
import sys

import numpy as np

SC = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = list(csv.DictReader(open(f'{SC}/mcrit_prediction.csv')))

GEO_HALF = {'Mx': 0.28629 / 2, 'My': 0.250 / 2}   # skid half-width, half-length
LP_NOMINAL = {'Mx': 0.140, 'My': 0.110}
CA, B = 0.0431, 0.04314
W_MINUS_F = 10.6                                   # N, dM/dl at the onset
WEIGHT = 31.59                                     # N, for mN.m -> mm of offset

by = {(r['case'], r['axis'], r['dir']): r for r in rows}
UNITS = sorted({(r['case'], r['axis']) for r in rows})


def col(name, axis=None, fn=float):
    return np.array([fn(r[name]) for r in rows
                     if axis is None or r['axis'] == axis])


print("1. the CAD footprint against the mocap arc\n")
print(f"  {'axis':6}{'CAD half':>10}{'nominal':>9}{'mocap fit':>11}"
      f"{'CAD - mocap':>13}{'as moment':>11}   [mm], [mN.m]")
for a in ('Mx', 'My'):
    mo = col('l_odom_mm', a).mean() * 1e-3
    gap = GEO_HALF[a] - mo
    print(f"  {a:6}{1e3 * GEO_HALF[a]:10.2f}{1e3 * LP_NOMINAL[a]:9.1f}"
          f"{1e3 * mo:11.1f}{1e3 * gap:13.1f}{1e3 * W_MINUS_F * gap:11.0f}")
r_geo = GEO_HALF['Mx'] / GEO_HALF['My']
r_moc = col('l_odom_mm', 'Mx').mean() / col('l_odom_mm', 'My').mean()
print(f"\n  roll/pitch ratio: CAD {r_geo:.3f}, mocap {r_moc:.3f}, "
      f"nominal {LP_NOMINAL['Mx'] / LP_NOMINAL['My']:.3f}")

print("\n  deficit if the CAD half-widths were used instead\n")
print(f"  {'axis':6}{'as fitted':>12}{'with CAD':>11}   [mN.m, median]")
for a in ('Mx', 'My'):
    cur, geo = [], []
    for r in rows:
        if r['axis'] != a:
            continue
        W, f = float(r['W']), float(r['f_onset'])
        M, sg = float(r['M_ident']), 1.0 if r['dir'] == 'pos' else -1.0
        for L, acc in ((float(r['l_odom_mm']) * 1e-3, cur), (GEO_HALF[a], geo)):
            acc.append(1e3 * abs(sg * (W - (1 + CA) * f) * L / (1 + B))
                       - 1e3 * abs(M))
    print(f"  {a:6}{np.median(cur):12.0f}{np.median(geo):11.0f}")
print("  -> larger arm, larger prediction, larger deficit: not the fix.")


def fit_cx(x, z):
    """The circle fit used by estimate_pivot_from_mocap: cz pinned to 0."""
    rhs = x ** 2 + z ** 2
    A = np.column_stack([2 * x, np.ones(len(x))])
    b, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx = b[0]
    R = np.sqrt(max(b[1] + cx ** 2, 0.0))
    return cx, R, float(np.std(np.sqrt((x - cx) ** 2 + z ** 2) - R))


print("\n2. a rolling foot of radius r, over an 8 deg arc\n")
print(f"  {'r [mm]':>8}{'contact bias [mm]':>20}{'fit residual [mm]':>20}")
for rr in (0.005, 0.010, 0.020, 0.050):
    p = np.linspace(0.0, np.deg2rad(8.0), 300)
    fx, fz = rr * p, np.full_like(p, rr)
    dx, dz = 0.0, 0.317 - rr
    x = fx + np.cos(p) * dx + np.sin(p) * dz
    z = fz - np.sin(p) * dx + np.cos(p) * dz
    cx, _, res = fit_cx(x, z)
    print(f"  {1e3 * rr:8.0f}{1e3 * cx:20.2f}{1e3 * res:20.4f}")
print("  -> right sign (outboard), an order of magnitude too small.")

print("\n3. an elevated rotation centre, bounded by the arm itself\n")
print(f"  {'c [mm]':>8}{'l_p fit [mm]':>14}{'vs 140.4 measured':>20}"
      f"{'residual [mm]':>15}")
for c in (0.0, 0.010, 0.025, 0.050, 0.119):
    p = np.linspace(0.0, np.deg2rad(8.0), 300)
    dx, dz = 0.140, 0.317 - c
    x = np.cos(p) * dx + np.sin(p) * dz
    z = c - np.sin(p) * dx + np.cos(p) * dz
    cx, _, res = fit_cx(x, z)
    lp = 0.140 - cx
    print(f"  {1e3 * c:8.0f}{1e3 * lp:14.1f}{1e3 * lp - 140.4:20.1f}"
          f"{1e3 * res:15.4f}")
print("  -> measured 140.4 +- 3.6 bounds c below ~10 mm: the kinematic")
print("     centre is at the foot, and the residual never reveals it.")

print("\n4. what reaches the delivered offset\n")
print(f"  {'case':9}{'ax':4}{'l_odom +':>10}{'l_odom -':>10}{'l+ - l-':>9}"
      f"{'sym resid':>11}{'d+ - d-':>10}   [mm], [mN.m], [mm]")
asym, sym = [], []
for c, a in UNITS:
    p, n = by[(c, a, 'pos')], by[(c, a, 'neg')]
    lp, ln = float(p['l_odom_mm']), float(n['l_odom_mm'])
    sy = 0.5 * (float(p['resid_interf_mNm']) + float(n['resid_interf_mNm']))
    asym.append(lp - ln)
    sym.append(sy)
    print(f"  {c:9}{a:4}{lp:10.1f}{ln:10.1f}{lp - ln:9.1f}{sy:11.0f}"
          f"{2 * sy * 1e-3 / W_MINUS_F * 1e3:10.1f}")
asym, sym = np.array(asym), np.array(sym)
for a, lab in (('Mx', 'roll '), ('My', 'pitch')):
    v = np.array([x for (c, ax), x in zip(UNITS, asym) if ax == a])
    print(f"\n  {lab} l+ - l-: {v.mean():+6.1f} mm mean, "
          f"{int(np.sum(v > 0))}/{len(v)} positive"
          f"   -- systematic per axis, not scatter")
print(f"\n  measured arm asymmetry |l+ - l-|   median {np.median(np.abs(asym)):5.1f} mm"
      f"   (already in the prediction)")
print(f"  surviving symmetric residual       median {np.median(np.abs(sym)):5.1f} mN.m"
      f"   -> d+ - d- = "
      f"{2 * np.median(np.abs(sym)) * 1e-3 / W_MINUS_F * 1e3:.1f} mm")
print(f"  as an offset                       "
      f"{np.median(np.abs(sym)) / WEIGHT:8.2f} mm"
      f"   against 1.43 mm measured between configurations")
print(f"\n  The common part of any lever error -- the 19.4 mm that H2 fits --")
print(f"  cancels identically in the pivot-free average.  Only the direction")
print(f"  asymmetry survives, and it is what sets the delivered accuracy.")
