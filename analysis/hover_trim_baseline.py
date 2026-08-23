#!/usr/bin/env python3
"""The in-flight baseline the comparison is missing: CoM offset from hover trim.

WHY. The estimator benchmark compares onset detectors against each
other, and the take-off results compare two disturbance observers.
Neither is an external baseline for the identification itself, which is
what a reader asks for: the online methods cited in the introduction
estimate the offset from flight data, so the fair question is what they
would deliver on this vehicle.

THE METHOD. It is the mainstream one and needs no new experiments. In
steady hover the vehicle is level and the collective thrust acts at the
geometric centre, so holding attitude against a centre-of-mass offset
costs a constant body moment,

    M_trim = -W p_off        =>        p_off = -M_trim / W ,

and M_trim is recoverable from the logged rotor speeds through the same
allocation of (1) that the pre-flight estimate uses. This is the
per-rotor-thrust route of the online identification literature,
evaluated here on the uncompensated take-off trials, where the vehicle
really is flying with the offset uncorrected.

WHAT IT COSTS, AND WHY THAT IS THE POINT. The estimate requires the
vehicle to be airborne and stably hovering with the offset
uncompensated -- which is the condition the pre-flight procedure exists
to avoid. The comparison is therefore not only of accuracy but of what
each method demands before it can be run, and both numbers belong in
the paper.

Only the uncompensated (wo_ff) trials are used: with the feedforward
moment applied the trim reflects the residual, not the offset.

Usage
-----
  PYTHONPATH=<stubs> python analysis/hover_trim_baseline.py [outdir]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

C_T = 1.3175e-7                  # N/rpm^2, table 9
K_M = 0.01569                    # N m/N,  table 9
ARM = 0.265                      # m,      table 9
G = 9.81
MASS = {'01': 3.066, '02': 3.220, '03': 3.220, '04': 3.220, '05': 3.220}
# load-cell truth, table 10 [mm]
TRUTH = {'01': (-11.45, -2.90), '02': (-9.90, -14.29), '03': (3.14, -5.26),
         '04': (2.40, 6.67), '05': (-10.89, 10.91)}
SETTLE = 2.0                     # s after lift-off before the hover window
RATE_CAP = 15.0                  # deg/s; samples above this are not hover
# A trial idles on the ground before take-off and after landing, at
# roughly a fifth of the weight, and those samples are not hover: a
# window selected by time alone reconstructs 0.61 W of collective
# thrust instead of 1.0 and reads the offset about half size. Hover is
# identified by the thrust actually carrying the vehicle.
THRUST_BAND = (0.90, 1.15)       # f / W

DATA = Path('DataSet/free_flight')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT.mkdir(parents=True, exist_ok=True)

# rotor geometry of (1): l_x,i = l cos(pi/6 + i pi/3), l_y,i = l sin(...)
# rotor index base checked against the load-cell truth: with i = 0..5
# the hover trim correlates with the delivered offset at r = 0.96,
# against 0.48 for i = 1..6, so this is the numbering the RPM message
# uses
i = np.arange(6)
LX = ARM * np.cos(np.pi / 6 + i * np.pi / 3)
LY = ARM * np.sin(np.pi / 6 + i * np.pi / 3)


def trim_offset(path, case):
    """(x_off, y_off) in mm from the mean hover moment, or None."""
    d = np.load(path)
    t, tr = d['odom/t'], d['rpm/t']
    W = MASS[case] * G
    T = C_T * d['rpm/rpm'].astype(np.float64) ** 2      # per-rotor thrust
    f = T.sum(axis=1)

    hit = np.flatnonzero(f >= W)
    if not hit.size:
        return None
    t_lo = float(tr[hit[0]])

    # settled hover: past the transient, and only while actually still
    rate = np.degrees(np.linalg.norm(
        d['odom/angular_vel'][:, :2].astype(np.float64), axis=1))
    keep_o = (t > t_lo + SETTLE) & (rate < RATE_CAP)
    if keep_o.sum() < 50:
        return None
    lo, hi = t[keep_o][0], t[keep_o][-1]
    keep_r = ((tr >= lo) & (tr <= hi)
              & (f >= THRUST_BAND[0] * W) & (f <= THRUST_BAND[1] * W))
    if keep_r.sum() < 100:
        return None

    # At equilibrium the rotor moment about the centre of mass vanishes,
    # so sum(l_i T_i) = p_off * f.  Normalising by the reconstructed
    # thrust rather than by W makes the estimate immune to an error in
    # C_T, which scales the moment and the thrust identically.
    fm = float(np.mean(f[keep_r]))
    return (1e3 * float(np.mean((T[keep_r] * LX).sum(axis=1))) / fm,
            1e3 * float(np.mean((T[keep_r] * LY).sum(axis=1))) / fm,
            float(tr[keep_r][-1] - tr[keep_r][0]))


rows = []
for p in sorted(DATA.glob('*/*/wo_ff*.npz')):
    case, ctrl, fn = p.relative_to(DATA).parts
    r = trim_offset(p, case)
    if r is None:
        print(f'  skipped {p.relative_to(DATA)}: no usable hover')
        continue
    x, y, dur = r
    rows.append(dict(case=case, controller=ctrl, run=fn[:-4],
                     hover_s=dur, x_mm=x, y_mm=y,
                     x_truth=TRUTH[case][0], y_truth=TRUTH[case][1],
                     x_err=x - TRUTH[case][0], y_err=y - TRUTH[case][1]))

if not rows:
    raise SystemExit('no usable hover segments')

with open(OUT / 'hover_trim_baseline.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

by = defaultdict(list)
for r in rows:
    by[r['case']].append(r)

print(f'{len(rows)} uncompensated trials, '
      f'{np.mean([r["hover_s"] for r in rows]):.0f} s of hover each\n')
print(f'{"case":<6}{"x truth":>9}{"x hover":>9}{"err":>7}'
      f'{"y truth":>10}{"y hover":>9}{"err":>7}')
errs = []
for case in sorted(by):
    g = by[case]
    x, y = np.mean([r['x_mm'] for r in g]), np.mean([r['y_mm'] for r in g])
    tx, ty = TRUTH[case]
    print(f'{case:<6}{tx:>9.2f}{x:>9.2f}{x - tx:>7.2f}'
          f'{ty:>10.2f}{y:>9.2f}{y - ty:>7.2f}')
    errs += [x - tx, y - ty]
e = np.abs(errs)
print(f'\nhover-trim baseline : {np.sqrt(np.mean(e ** 2)):.2f} mm RMS, '
      f'median {np.median(e):.2f}, max {e.max():.2f}')
print('pre-flight (paired) : 1.77 mm RMS  '
      '-- but obtained before the first take-off')
print(f"written to {OUT / 'hover_trim_baseline.csv'}")
