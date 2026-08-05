#!/usr/bin/env python3
"""Per-direction critical-moment check against manuscript Eqs. (7)/(14).

The static tip-over thresholds are
  M_{x,+} = (W - f) l_r + W y_off        M_{x,-} = -(W - f) l_l + W y_off
  M_{y,+} = (W - f) l_f - W x_off        M_{y,-} = -(W - f) l_b - W x_off
with W and the CoM offsets known from ground truth (manuscript Table 7)
and f the measured collective thrust at the onset.  Two readings:

  forward  — predict M_crit per direction with the nominal symmetric arm
             (l_p = 0.140 m roll / 0.110 m pitch) and compare with the
             identified directional means; the residual budget is the
             ground-effect thrust band f_true = gamma*f,
             gamma - 1 in [1.0%, 4.2%] (bracket), which enters as
             -/+ (gamma-1) f l_p per direction, plus the arm tolerance
             (W - f) dl.
  inverse  — with the offset term pinned by truth, invert each
             directional mean for the effective arm
             l_hat = +/-(Mbar_dir - S_off)/(W - f_dir) and check physical
             plausibility and cross-case consistency.

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
ARM = {'Mx': 0.140, 'My': 0.110}      # nominal symmetric pivot arm [m]
GE_BAND = (0.010, 0.042)              # gamma - 1 bracket

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    key = (d.parent.name, d.name)
    W = MASS_KG[key[0]] * G
    S_off = OFF_SIGN[key[1]] * W * OFF_MM[key] * 1e-3
    lp = ARM[key[1]]
    by = defaultdict(lambda: ([], []))
    for c in crits:
        dirn = 'pos' if c.bag_name.startswith('pos') else 'neg'
        by[dirn][0].append(c.onset_moment)
        by[dirn][1].append(c.onset_thrust)
    for dirn in ('neg', 'pos'):
        M_bar = float(np.mean(by[dirn][0]))
        f_bar = float(np.mean(by[dirn][1]))
        sgn = +1.0 if dirn == 'pos' else -1.0
        pred = sgn * (W - f_bar) * lp + S_off
        # GE band: f_true = gamma*f shifts the prediction by
        # -sgn*(gamma-1)*f*lp  (thrust larger than measured -> smaller |M|)
        band = sorted(pred - sgn * g * f_bar * lp for g in GE_BAND)
        resid = M_bar - pred
        l_hat = sgn * (M_bar - S_off) / (W - f_bar)
        rows.append(dict(case=key[0], axis=key[1], dir=dirn,
                         W=f"{W:.2f}", f_onset=f"{f_bar:.2f}",
                         Wmf=f"{W - f_bar:.2f}",
                         M_ident=f"{M_bar:+.4f}", M_pred=f"{pred:+.4f}",
                         M_pred_ge_lo=f"{band[0]:+.4f}",
                         M_pred_ge_hi=f"{band[1]:+.4f}",
                         resid_mNm=f"{1e3 * resid:+.1f}",
                         in_band=str(band[0] <= M_bar <= band[1]),
                         l_hat_mm=f"{1e3 * l_hat:.1f}"))
    print(f"done {key[0]}/{key[1]}", flush=True)

with open(OUT / 'mcrit_prediction.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

hdr = (f"{'case':8} {'ax':3} {'dir':4} {'W-f':>6} {'M_ident':>9} "
       f"{'M_pred':>9} {'GE band':>19} {'resid':>7} {'inB':>4} "
       f"{'l_hat':>6}")
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['case']:8} {r['axis']:3} {r['dir']:4} {r['Wmf']:>6} "
          f"{r['M_ident']:>9} {r['M_pred']:>9} "
          f"[{r['M_pred_ge_lo']}, {r['M_pred_ge_hi']}] "
          f"{r['resid_mNm']:>7} {r['in_band']:>4} {r['l_hat_mm']:>6}")

res = np.array([float(r['resid_mNm']) for r in rows])
lh = {ax: [float(r['l_hat_mm']) for r in rows if r['axis'] == ax]
      for ax in ('Mx', 'My')}
print(f"\nresidual vs nominal-arm prediction [mN·m]: "
      f"median {np.median(np.abs(res)):.1f}, "
      f"p90 {np.percentile(np.abs(res), 90):.1f}, "
      f"max {np.max(np.abs(res)):.1f}")
for ax in ('Mx', 'My'):
    a = np.array(lh[ax])
    print(f"effective arm l_hat ({ax}) [mm]: mean {a.mean():.1f}, "
          f"std {a.std(ddof=1):.1f}, range [{a.min():.1f}, {a.max():.1f}] "
          f"(nominal {1e3 * ARM[ax]:.0f})")
