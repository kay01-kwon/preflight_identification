#!/usr/bin/env python3
"""Per-direction critical-moment check against manuscript Eqs. (7)/(14).

The static tip-over thresholds are
  M_{x,+} = (W - f) l_r + W y_off        M_{x,-} = -(W - f) l_l + W y_off
  M_{y,+} = (W - f) l_f - W x_off        M_{y,-} = -(W - f) l_b - W x_off
with every ingredient measured independently of the identified moments:
W and the CoM offsets from ground truth (manuscript Table 7), f the
measured collective thrust at the onset, and the pivot arms fitted from
the odometry/mocap position trace per run (circle fit about the contact
line, ``estimate_pivot_from_mocap``).  Each run is predicted with its
OWN arm and onset thrust; per-run predictions (and residuals) are then
aggregated per case/axis/direction.  The identified mean uses the same
valid-pivot run subset, so mean residual = mean(M) - mean(pred).

Because the arms are independent, the ground-effect question becomes a
genuine forward test.  The GE correction uses the FULL pivot-moment
decomposition of analysis/ge_linearity.py (both channels),
    Delta M_GE = sgn*a + b*M,   a = c_a * f * l,   at the onset
so the balance M(1+b) + sgn*a = T_rigid gives
    M_pred = (T_rigid - sgn*a) / (1 + b),  T_rigid = sgn*(W-f)l + S.
Coefficients at phi = 0 (attitude dependence < 0.5% over the measured
onset tilts), from `python analysis/ge_linearity.py --model ...`:
    single   c_a = 1.03%,  b = 1.026%   (lower bound, no interference)
    interf   c_a = 4.31%,  b = 4.314%   (pure image superposition over
             rotor-centre distances: mutual interference, no empirical
             constants -- no k matching, no fountain)
    garofano c_a = 11.67%, b = 4.155%   (upper bound, transferred consts)

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
GE = {'single': (0.0103, 0.01026),     # (c_a, b) at phi=0
      'interf': (0.0431, 0.04314),
      'garofano': (0.1167, 0.04155)}

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
    by = defaultdict(list)
    n_all = defaultdict(int)
    for c in crits:
        dirn = 'pos' if c.bag_name.startswith('pos') else 'neg'
        n_all[dirn] += 1
        piv = cvp.estimate_pivot_from_mocap(by_bag[c.bag_name],
                                            c.onset_time, axis)
        if not np.isnan(piv['pivot_abs']):
            by[dirn].append((c.onset_moment, c.onset_thrust,
                             piv['pivot_abs'] * 1e-3))
    for dirn in ('neg', 'pos'):
        runs = by[dirn]
        sgn = +1.0 if dirn == 'pos' else -1.0
        # per-run prediction with that run's own arm and onset thrust,
        # then aggregate: pred_dir = mean over runs (and identically for
        # the residuals, since the identified mean uses the same subset)
        per = {'none': [], 'single': [], 'interf': [],
               'garofano': []}
        for M_r, f_r, l_r in runs:
            T = sgn * (W - f_r) * l_r + S_off
            per['none'].append(T)
            for name, (ca, b) in GE.items():
                per[name].append((T - sgn * ca * f_r * l_r) / (1.0 + b))
        M_bar = float(np.mean([r[0] for r in runs]))
        f_bar = float(np.mean([r[1] for r in runs]))
        arms = np.array([r[2] for r in runs])
        pred = {k: float(np.mean(v)) for k, v in per.items()}
        band = sorted((pred['single'], pred['garofano']))
        res_runs = 1e3 * (np.array([r[0] for r in runs])
                          - np.array(per['none']))
        rows.append(dict(case=key[0], axis=key[1], dir=dirn,
                         W=f"{W:.2f}", f_onset=f"{f_bar:.2f}",
                         l_odom_mm=f"{1e3 * arms.mean():.1f}",
                         l_std_mm=(f"{1e3 * arms.std(ddof=1):.1f}"
                                   if len(arms) > 1 else ''),
                         n_piv=f"{len(runs)}/{n_all[dirn]}",
                         M_ident=f"{M_bar:+.4f}",
                         M_pred=f"{pred['none']:+.4f}",
                         M_pred_single=f"{pred['single']:+.4f}",
                         M_pred_interf=f"{pred['interf']:+.4f}",
                         M_pred_garofano=f"{pred['garofano']:+.4f}",
                         resid_mNm=f"{1e3 * (M_bar - pred['none']):+.1f}",
                         resid_run_std_mNm=
                         f"{res_runs.std(ddof=1):.1f}"
                         if len(runs) > 1 else '',
                         resid_single_mNm=
                         f"{1e3 * (M_bar - pred['single']):+.1f}",
                         resid_interf_mNm=
                         f"{1e3 * (M_bar - pred['interf']):+.1f}",
                         resid_garofano_mNm=
                         f"{1e3 * (M_bar - pred['garofano']):+.1f}",
                         in_bracket=str(band[0] <= M_bar <= band[1])))
    print(f"done {key[0]}/{key[1]}", flush=True)

with open(OUT / 'mcrit_prediction.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

hdr = (f"{'case':8} {'ax':3} {'dir':4} {'l_odom':>7} {'M_ident':>9} "
       f"{'pred:none':>9} {'resid':>7} {'r_sgl':>7} {'r_int':>7} "
       f"{'r_gar':>7} {'inBr':>5}")
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['case']:8} {r['axis']:3} {r['dir']:4} {r['l_odom_mm']:>7} "
          f"{r['M_ident']:>9} {r['M_pred']:>9} {r['resid_mNm']:>7} "
          f"{r['resid_single_mNm']:>7} {r['resid_interf_mNm']:>7} "
          f"{r['resid_garofano_mNm']:>7} {r['in_bracket']:>5}")

for lbl, col in (('no GE   ', 'resid_mNm'),
                 ('single  ', 'resid_single_mNm'),
                 ('interf  ', 'resid_interf_mNm'),
                 ('garofano', 'resid_garofano_mNm')):
    a = np.abs(np.array([float(r[col]) for r in rows]))
    print(f"{lbl}: |resid| median {np.median(a):.1f}, "
          f"p90 {np.percentile(a, 90):.1f}, max {a.max():.1f}, "
          f"RMS {np.sqrt(np.mean(a**2)):.1f} mN·m")
for ax in ('Mx', 'My'):
    a = np.array([float(r['l_odom_mm']) for r in rows if r['axis'] == ax])
    print(f"odom-fitted arm ({ax}) [mm]: mean {a.mean():.1f}, "
          f"std {a.std(ddof=1):.1f}, range [{a.min():.1f}, {a.max():.1f}]")
