#!/usr/bin/env python3
"""Per-direction critical-moment check against manuscript Eqs. (7)/(14).

The static tip-over thresholds are
  M_{x,+} = (W - f) l_r + W y_off        M_{x,-} = -(W - f) l_l + W y_off
  M_{y,+} = (W - f) l_f - W x_off        M_{y,-} = -(W - f) l_b - W x_off
with every ingredient measured independently of the identified moments:
W and the CoM offsets from ground truth (manuscript Table 7), f the
measured collective thrust at the onset, and the pivot arms fitted from
the odometry/mocap position trace per run (circle fit about the contact
line, ``estimate_pivot_from_mocap``), averaged per case/axis/direction.

Because the arms are independent, the ground-effect question becomes a
genuine forward test: predictions are evaluated at gamma = 1 (measured
thrust as-is) and across the bracket gamma - 1 in [1.0, 4.2]% with the
arms held fixed.

Usage: PYTHONPATH=<stubs> python analysis/mcrit_prediction.py [outdir]
"""
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}          # Table 7
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}   # +W y_off (roll) / -W x_off (pitch)
GE_BAND = (0.010, 0.042)              # gamma - 1 bracket
GE_MID = 0.5 * (GE_BAND[0] + GE_BAND[1])

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    by_bag = {b.name: b for b in bags}
    key = (d.parent.name, d.name)
    W = MASS_KG[key[0]] * G
    S_off = OFF_SIGN[key[1]] * W * OFF_MM[key] * 1e-3
    by = defaultdict(lambda: ([], [], []))
    for c in crits:
        dirn = 'pos' if c.bag_name.startswith('pos') else 'neg'
        piv = cvp.estimate_pivot_from_mocap(by_bag[c.bag_name],
                                            c.onset_time, axis)
        by[dirn][0].append(c.onset_moment)
        by[dirn][1].append(c.onset_thrust)
        if not np.isnan(piv['pivot_abs']):
            by[dirn][2].append(piv['pivot_abs'] * 1e-3)
    for dirn in ('neg', 'pos'):
        M_bar = float(np.mean(by[dirn][0]))
        f_bar = float(np.mean(by[dirn][1]))
        arms = np.array(by[dirn][2])
        l_bar = float(np.mean(arms))
        sgn = +1.0 if dirn == 'pos' else -1.0
        pred = {g: sgn * (W - (1 + g) * f_bar) * l_bar + S_off
                for g in (0.0, GE_MID, *GE_BAND)}
        band = sorted((pred[GE_BAND[0]], pred[GE_BAND[1]]))
        rows.append(dict(case=key[0], axis=key[1], dir=dirn,
                         W=f"{W:.2f}", f_onset=f"{f_bar:.2f}",
                         l_odom_mm=f"{1e3 * l_bar:.1f}",
                         l_std_mm=(f"{1e3 * arms.std(ddof=1):.1f}"
                                   if len(arms) > 1 else ''),
                         n_piv=len(arms),
                         M_ident=f"{M_bar:+.4f}",
                         M_pred=f"{pred[0.0]:+.4f}",
                         M_pred_ge_mid=f"{pred[GE_MID]:+.4f}",
                         M_pred_ge_lo=f"{band[0]:+.4f}",
                         M_pred_ge_hi=f"{band[1]:+.4f}",
                         resid_mNm=f"{1e3 * (M_bar - pred[0.0]):+.1f}",
                         resid_ge_mid_mNm=
                         f"{1e3 * (M_bar - pred[GE_MID]):+.1f}",
                         in_band=str(band[0] <= M_bar <= band[1])))
    print(f"done {key[0]}/{key[1]}", flush=True)

with open(OUT / 'mcrit_prediction.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

hdr = (f"{'case':8} {'ax':3} {'dir':4} {'l_odom':>7} {'M_ident':>9} "
       f"{'pred(g=1)':>9} {'resid':>7} {'pred(mid)':>9} {'residGE':>8} "
       f"{'inB':>5}")
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['case']:8} {r['axis']:3} {r['dir']:4} {r['l_odom_mm']:>7} "
          f"{r['M_ident']:>9} {r['M_pred']:>9} {r['resid_mNm']:>7} "
          f"{r['M_pred_ge_mid']:>9} {r['resid_ge_mid_mNm']:>8} "
          f"{r['in_band']:>5}")

for lbl, col in (('gamma=1 ', 'resid_mNm'), ('GE mid  ', 'resid_ge_mid_mNm')):
    a = np.abs(np.array([float(r[col]) for r in rows]))
    print(f"{lbl}: |resid| median {np.median(a):.1f}, "
          f"p90 {np.percentile(a, 90):.1f}, max {a.max():.1f}, "
          f"RMS {np.sqrt(np.mean(a**2)):.1f} mN·m")
for ax in ('Mx', 'My'):
    a = np.array([float(r['l_odom_mm']) for r in rows if r['axis'] == ax])
    print(f"odom-fitted arm ({ax}) [mm]: mean {a.mean():.1f}, "
          f"std {a.std(ddof=1):.1f}, range [{a.min():.1f}, {a.max():.1f}]")
