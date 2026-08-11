#!/usr/bin/env python3
"""Where is the rotation centre?  Free the pivot height and let the arc say.

analysis/pivot_geom.py fits the marker's arc with the pivot height PINNED
to ground level (fit_circle_cz_fixed with cz = 0), so it measures the
horizontal arm l_p and assumes the answer to the question asked here.

The question comes from the calibration.  The amplitude constant returns
W z directly, K = 1/(W z), and over the ten configurations it implies

    h_eff = 1/(K W)  in  [0.094, 0.211] m

against a CAD CoM height of 0.261 m above the foot.  Two readings fit
that.  A landing-gear stiffness in PARALLEL with gravity would subtract,
Wz_eff = Wz - k_land; but it leaves the inertia at its foot-pivot value
and then C2 = sqrt(Wz_eff/J_P) comes out well BELOW what is identified,
so it cannot be the whole story.  A compliant leg that moves the
instantaneous centre UP the leg shrinks the arm, and shrinks Wz and J_P
together:

    Wz_eff = W h,     J_eff = J_CoM + m (h^2 + l_p^2),

which leaves C2^2 = W h / J_eff nearly flat in h -- 4.60 to 5.06 over
h = 0.10 to 0.26 m -- and that is exactly why C2 stays inside [3.5, 5.7]
while the identified Wz varies two-fold.  On this reading C2 is a weak
diagnostic and K carries the information.

The arc can decide, because estimate_pivot_from_mocap already accepts cz.
Sweeping it and taking the residual minimum asks the mocap where the
centre is, with nothing pinned.  Three outcomes:

  * minimum near h_eff (~0.10-0.17 m):  the compliant-leg reading is
    confirmed, the rigid parallel-axis bound (82) is not a bound, and
    the C2 ceiling in VI-E must come from the fits instead.
  * minimum near 0:  the foot really is the pivot and the low identified
    Wz needs another explanation.
  * flat residual:  the arc is too short to resolve cz, in which case
    the scan bounds nothing and should be reported as such.

The third is a live possibility: the excitation stops at a 10 deg tilt,
so the arc subtends little and the radial direction is poorly
conditioned.  The spread of the minimum across runs is printed for that
reason.

Usage: python analysis/pivot_height_scan.py
"""
import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT
from pnls_constants import PNLS_CONSTANTS

G, Z_CAD, J_COM = 9.80665, 0.261, 0.0537
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
CZ = np.linspace(0.0, 0.28, 29)


def h_eff(case, axis):
    """CoM height above the rotation centre implied by the amplitude gain."""
    _, k = PNLS_CONSTANTS[(case, 'Mx' if axis == 'x' else 'My')]
    return 1.0 / (k * MASS[case] * G)


print("cz swept with everything else free; the arc picks the minimum\n")
print(f"  {'case':9}{'ax':4}{'n':>3}{'h_eff (from K)':>16}"
      f"{'cz* (from arc)':>16}{'spread':>8}{'resid at cz*':>14}"
      f"{'at cz=0':>10}   [m], [mm]")
rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    case = d.parent.name
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    by_bag = {b.name: b for b in bags}
    best, r_best, r_zero = [], [], []
    for c in crits:
        res = []
        for cz in CZ:
            with contextlib.redirect_stdout(io.StringIO()):
                p = cvp.estimate_pivot_from_mocap(by_bag[c.bag_name],
                                                  c.onset_time, axis, cz=cz)
            res.append(p['residual'] if np.isfinite(p['pivot_abs'])
                       else np.nan)
        res = np.array(res, dtype=float)
        if np.all(~np.isfinite(res)):
            continue
        j = int(np.nanargmin(res))
        best.append(CZ[j])
        r_best.append(res[j])
        r_zero.append(res[0])
    if not best:
        continue
    best = np.array(best)
    rows.append((case, axis, best, np.array(r_best), np.array(r_zero)))
    print(f"  {case:9}{d.name:4}{len(best):3d}{h_eff(case, axis):16.3f}"
          f"{best.mean():16.3f}{best.std():8.3f}{np.mean(r_best):14.2f}"
          f"{np.mean(r_zero):10.2f}")

print()
allb = np.concatenate([r[2] for r in rows])
hs = np.array([h_eff(c, a) for c, a, *_ in rows])
cz_mean = np.array([r[2].mean() for r in rows])
rb = np.concatenate([r[3] for r in rows])
rz = np.concatenate([r[4] for r in rows])
print(f"  cz* pooled: mean {allb.mean():.3f} m, sd {allb.std():.3f}, "
      f"median {np.median(allb):.3f}")
print(f"  h_eff     : mean {hs.mean():.3f} m, range "
      f"[{hs.min():.3f}, {hs.max():.3f}]")
r_pool = np.corrcoef(cz_mean, hs)[0, 1]
print(f"  correlation cz* vs h_eff over the ten configurations: r = {r_pool:+.2f}")
# Guard against reading that correlation as evidence.  cz* takes very few
# distinct values here, and both it and h_eff differ systematically between
# the axes, so a pooled r is mostly the axis split.  Report the within-axis
# correlations and how much of the sweep range is being hit at its edge.
for ax, lbl in (('x', 'Mx'), ('y', 'My')):
    m = [i for i, r in enumerate(rows) if r[1] == ax]
    if len(m) > 2:
        print(f"    within {lbl}: r = "
              f"{np.corrcoef(cz_mean[m], hs[m])[0, 1]:+.2f}, "
              f"{len(set(np.round(cz_mean[m], 3)))} distinct cz*, "
              f"{int(np.sum(np.isclose(cz_mean[m], CZ[-1])))}/{len(m)} at the "
              f"sweep edge")
if len(set(np.round(cz_mean, 3))) <= 4 or np.any(np.isclose(cz_mean, CZ[-1])):
    print("    cz* is piling up on few values or on the sweep boundary, which")
    print("    is what a flat objective does; the pooled r is not evidence.")
print(f"  residual: {np.mean(rb):.2f} mm at cz*, {np.mean(rz):.2f} mm at "
      f"cz = 0  -> freeing the height buys "
      f"{100 * (1 - np.mean(rb) / np.mean(rz)):.0f}%")
if np.mean(rz) - np.mean(rb) < 0.05 * np.mean(rz):
    print("\n  The residual is flat in cz: the arc does not resolve the")
    print("  rotation-centre height, and this scan bounds nothing.  Report")
    print("  it as a null result rather than as support either way.")
print(f"\n  for reference, if the centre sat at cz then the implied inertia")
print(f"  J = J_CoM + m((z_CAD - cz)^2 + l_p^2) would be:")
for cz in (0.0, 0.10, 0.15):
    j = J_COM + 3.220 * ((Z_CAD - cz) ** 2 + LP['x'] ** 2)
    print(f"    cz = {cz:.2f} m -> J = {j:.3f} kg.m^2   (identified range"
          f" 0.10-0.28)" if cz == 0.0 else f"    cz = {cz:.2f} m -> J = {j:.3f}")
