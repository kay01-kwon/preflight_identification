"""What the mocap circle fit actually measures, and what it bounds.

estimate_pivot_from_mocap fits the marker's arc about a pivot whose
HEIGHT is fixed at ground level (fit_circle_cz_fixed with cz = 0); only
the horizontal centre cx and the radius R are free.  So the fit measures
the pivot arm l_p, not the CoM height.

But it also reports the marker's own height, and the marker sits on the
airframe -- so the marker rest height is an upper bound on any body
point's height above the pivot, z_CoM included, up to wherever the
marker actually is.  This prints both.
"""
import contextlib, io, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

LP_USED = {'x': 0.140, 'y': 0.110}

print(f"{'case':<9}{'ax':<4}{'n':>4}{'l_p fit [mm]':>22}{'resid':>8}"
      f"{'R [mm]':>9}{'z_marker rest [mm]':>21}")
print('-' * 77)
agg = defaultdict(list)
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    by_bag = {b.name: b for b in bags}
    lp, res, rr, zm = [], [], [], []
    for c in crits:
        p = cvp.estimate_pivot_from_mocap(by_bag[c.bag_name], c.onset_time, axis)
        if np.isnan(p['pivot_abs']):
            continue
        lp.append(p['pivot_abs']); res.append(p['residual']); rr.append(p['R'])
        zm.append(float(np.mean(p['z_fit'][:3])))
    if not lp:
        continue
    lp, res, rr, zm = map(np.array, (lp, res, rr, zm))
    agg[axis].append((lp, rr, zm))
    print(f"{d.parent.name:<9}{d.name:<4}{len(lp):4d}"
          f"{lp.mean():10.1f} +- {lp.std():5.1f}{res.mean():8.1f}"
          f"{rr.mean():9.1f}{zm.mean():15.1f} +- {zm.std():4.1f}")

print()
for ax in 'xy':
    lp = np.concatenate([a[0] for a in agg[ax]])
    rr = np.concatenate([a[1] for a in agg[ax]])
    zm = np.concatenate([a[2] for a in agg[ax]])
    name = 'roll (Mx)' if ax == 'x' else 'pitch (My)'
    print(f"{name:<11} l_p fit  {lp.mean()/1e3:.4f} +- {lp.std()/1e3:.4f} m "
          f"(median {np.median(lp)/1e3:.4f}), used {LP_USED[ax]:.3f} m "
          f"-> {100*(LP_USED[ax]*1e3/lp.mean()-1):+.1f}%")
    print(f"{'':11} R        {rr.mean()/1e3:.4f} m   "
          f"marker rest height {zm.mean()/1e3:.4f} +- {zm.std()/1e3:.4f} m")
    print(f"{'':11} sqrt(R^2 - l_p^2) = {np.sqrt(np.maximum(rr**2-lp**2,0)).mean()/1e3:.4f} m"
          f"   (marker height implied by the circle fit)")
