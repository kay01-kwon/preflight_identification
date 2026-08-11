#!/usr/bin/env python3
"""Does the threshold deficit track the circle-fit residual?  A discriminating test.

The static threshold is defined at the onset, phi = 0, but the mocap
circle fit averages the arc that follows, phi = 0 to 6-9 deg.  If the
contact migrates outboard while the vehicle tips -- which soft rubber on
a skid should do, as the load redistributes toward the rail end -- then
the fitted arm exceeds the arm at the onset and the rigid threshold is
over-predicted.  That is the observed sign.

The size is bounded by something already measured.  Modelling the
migration as rolling with an effective radius r_eff, the contact moves
r_eff * phi and the marker traces a trochoid rather than a circle, so
the fit leaves a residual.  Over an 8 deg arc,

    r_eff = 100 mm -> arm bias  7.0 mm, residual 0.073 mm
    r_eff = 200 mm -> arm bias 14.0 mm, residual 0.146 mm
    r_eff = 278 mm -> arm bias 19.4 mm, residual 0.203 mm

and 19.4 mm is exactly the lever displacement that analysis/
static_attribution.py fits to the deficit.  The residuals actually
observed are 0.1 to 0.2 mm, so the mechanism is the right size at the
top of that range and about half of it at the bottom.

That is suggestive but not a test.  The test is this: if the deficit is
produced by contact migration, then configurations with a larger arc
residual should show a larger deficit, because both are driven by the
same r_eff.  A ground-effect moment predicts no such relation -- the
aerodynamics do not care how circular the arc is.  So the correlation
across the ten configurations discriminates between them, using data
already in hand.

Two cautions on reading the result.  The residual also contains mocap
noise, which dilutes any true correlation toward zero, so a null is
weak evidence against and a positive is stronger evidence for.  And with
ten configurations the standard error on r is about 0.33, so only a
large correlation means anything; the permutation p-value is reported
rather than a nominal one.

Usage: python analysis/arc_residual_test.py <dir with mcrit_prediction.csv>
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

SC = sys.argv[1] if len(sys.argv) > 1 else '.'
CA, B = 0.0431, 0.04314

rows = list(csv.DictReader(open(f'{SC}/mcrit_prediction.csv')))
by = {(r['case'], r['axis'], r['dir']): r for r in rows}


def deficit(case, axis):
    """Median |prediction| - |measured| over the two directions, mN.m."""
    out = []
    for d in ('pos', 'neg'):
        r = by[(case, axis, d)]
        W, f = float(r['W']), float(r['f_onset'])
        l, M = float(r['l_odom_mm']) * 1e-3, float(r['M_ident'])
        sg = 1.0 if d == 'pos' else -1.0
        out.append(1e3 * abs(sg * (W - (1 + CA) * f) * l / (1 + B))
                   - 1e3 * abs(M))
    return float(np.mean(out))


print("per-configuration arc residual against the threshold deficit\n")
print(f"  {'case':9}{'ax':4}{'n':>4}{'arc resid [mm]':>17}{'phi_end [deg]':>15}"
      f"{'deficit [mN.m]':>16}")
res, dfc, tag = [], [], []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    seen = {b.name: b for b in bags}
    rr = []
    for c in crits:
        with contextlib.redirect_stdout(io.StringIO()):
            p = cvp.estimate_pivot_from_mocap(seen[c.bag_name],
                                              c.onset_time, axis)
        if np.isfinite(p['pivot_abs']):
            rr.append(p['residual'])
    if not rr:
        continue
    v = float(np.mean(rr))
    g = deficit(d.parent.name, d.name)
    res.append(v)
    dfc.append(g)
    tag.append(f"{d.parent.name}/{d.name}")
    print(f"  {d.parent.name:9}{d.name:4}{len(rr):4d}{v:17.4f}{'':15}{g:16.0f}")

res, dfc = np.array(res), np.array(dfc)
r = float(np.corrcoef(res, dfc)[0, 1])
rng = np.random.default_rng(0)
null = np.array([abs(np.corrcoef(rng.permutation(res), dfc)[0, 1])
                 for _ in range(20000)])
p = (np.sum(null >= abs(r)) + 1) / 20001
print(f"\n  correlation over {len(res)} configurations: r = {r:+.2f}, "
      f"permutation p = {p:.3f}")
print(f"  (SE on r at n = {len(res)} is about {1 / np.sqrt(len(res) - 3):.2f})")

# A pooled r is not to be read on its own here.  Both the residual and the
# deficit differ systematically between the axes, so a positive pooled
# figure can be nothing more than that offset -- the same trap the cz sweep
# in pivot_height_scan.py sprang.  Print the within-axis correlations and
# the axis means beside it, and say so when they disagree in sign.
print(f"\n  the same, within each axis (read these first):")
signs = []
for a, lab in (('Mx', 'roll '), ('My', 'pitch')):
    m = [i for i, t in enumerate(tag) if t.endswith(a)]
    if len(m) > 2:
        rw = float(np.corrcoef(res[m], dfc[m])[0, 1])
        signs.append(rw)
        print(f"    {lab}: r = {rw:+.2f} (n = {len(m)}),"
              f"  mean resid {res[m].mean():.4f} mm,"
              f"  mean deficit {dfc[m].mean():+.0f} mN.m")
if len(signs) == 2 and signs[0] * signs[1] < 0:
    print(f"    The within-axis correlations have OPPOSITE signs, so the")
    print(f"    pooled r is the axis offset and not a relation.  With n = 5")
    print(f"    per axis this is what noise looks like; the test is a null.")

print(f"\n  arm bias implied by each configuration's residual, if the residual")
print(f"  were entirely contact migration (an upper bound, since it is not):")
print(f"  {'case/axis':16}{'resid [mm]':>12}{'r_eff [mm]':>12}"
      f"{'bias [mm]':>11}{'as moment [mN.m]':>18}")


def fit_cx(x, z):
    rhs = x ** 2 + z ** 2
    A = np.column_stack([2 * x, np.ones(len(x))])
    b, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx = b[0]
    R = np.sqrt(max(b[1] + cx ** 2, 0.0))
    return cx, float(np.std(np.sqrt((x - cx) ** 2 + z ** 2) - R))


def roll_arc(reff, pm=np.deg2rad(8.0), zm=0.317, n=400):
    q = np.linspace(0.0, pm, n)
    dz = zm - reff
    return reff * q + np.sin(q) * dz, reff + np.cos(q) * dz


grid = np.linspace(1e-3, 0.60, 400)
tab = np.array([[g, *fit_cx(*roll_arc(g))] for g in grid])
for t, v, g in zip(tag, res, dfc):
    j = int(np.argmin(np.abs(tab[:, 2] - v * 1e-3)))
    print(f"  {t:16}{v:12.4f}{1e3 * tab[j, 0]:12.0f}{1e3 * tab[j, 1]:11.1f}"
          f"{1e3 * 10.6 * tab[j, 1]:18.0f}")
