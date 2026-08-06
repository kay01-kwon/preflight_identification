#!/usr/bin/env python3
"""Bilinear GE term along measured trajectories.

The affine (tangent) form of the ground-effect pivot moment,
    dM_GE = alpha_M M_cmd + alpha_f f l_p + (beta_M M_cmd + beta_f f l_p) dphi,
has a moment-proportional coefficient that is itself attitude dependent.
Writing M_cmd = M_crit + Mdot*tau, the part

    beta_M * (Mdot*tau) * dphi(tau)                              (bilinear)

is the only piece of the decomposition that is NOT in the absorbed span
{1, tau, dphi}: it is a product of two of them.  Everything else is
either constant (-> onset anchoring), proportional to tau (-> effective
Mdot), or proportional to dphi (-> effective anti-restoring gradient).

This script measures that term on every gate-passing run over the fit
window [onset, window end], using the measured moment and attitude
traces and beta_M = |d b/d phi| from analysis/ge_linearity.py:

    single 0.0122 /rad,  interf 0.0345 /rad,  garofano 0.0352 /rad.

It reports both the raw magnitude and the part that survives the
estimator's projection (out-of-span), which is what actually reaches
rho.

Usage: PYTHONPATH=<stubs> python analysis/ge_bilinear.py [outdir]
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    extract_piecewise_batch, detect_excitation_window, commanded_ramp_rate,
    MOMENT_CAP)

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
BETA_M = {'single': 0.0122, 'interf': 0.0345, 'garofano': 0.0352}  # 1/rad

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = extract_piecewise_batch(bags, axis)
    bag_by = {b.name: b for b in bags}
    for crit in crits:
        bag = bag_by[crit.bag_name]
        i0, i1 = detect_excitation_window(crit.moment,
                                          moment_cap=MOMENT_CAP.get(axis))
        j = crit.onset_idx
        if i1 - j < 10:
            continue
        roll, pitch = math_tools.quaternion_to_euler_vectorized(
            bag.odom.quaternion)
        phi = roll if axis == 'x' else pitch
        n = min(len(phi), len(crit.t))
        i1 = min(i1, n - 1)
        sl = slice(j, i1 + 1)
        tau = crit.t[sl] - crit.t[j]
        dM = crit.moment[sl] - crit.moment[j]          # = Mdot*tau
        dphi = phi[sl] - phi[j]
        A = np.vstack([np.ones_like(tau), tau, dphi]).T
        prod = dM * dphi                                # bilinear shape
        c, *_ = np.linalg.lstsq(A, prod, rcond=None)
        oos = prod - A @ c                              # out-of-span part
        rows.append(dict(
            case=d.parent.name, axis=d.name, bag=crit.bag_name,
            rate=commanded_ramp_rate(crit.bag_name) or np.nan,
            dM_end=f"{dM[-1]:+.3f}", dphi_end_deg=f"{np.rad2deg(dphi[-1]):+.2f}",
            **{f'raw_{k}_mNm': f"{1e3 * b * np.max(np.abs(prod)):.2f}"
               for k, b in BETA_M.items()},
            **{f'oos_{k}_mNm': f"{1e3 * b * np.max(np.abs(oos)):.2f}"
               for k, b in BETA_M.items()}))
    print(f'done {d.parent.name}/{d.name}', flush=True)

with open(OUT / 'ge_bilinear.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} runs; post-onset moment rise |Mdot*tau| max "
      f"{max(abs(float(r['dM_end'])) for r in rows):.3f} N·m, "
      f"excursion max {max(abs(float(r['dphi_end_deg'])) for r in rows):.2f} deg")
print(f"\n{'model':9} {'beta_M':>8} | {'raw peak [mN·m]':>26} | "
      f"{'out-of-span peak [mN·m]':>26}")
print(f"{'':9} {'[1/rad]':>8} | {'median':>8} {'p90':>8} {'max':>8} | "
      f"{'median':>8} {'p90':>8} {'max':>8}")
for k, b in BETA_M.items():
    raw = np.array([float(r[f'raw_{k}_mNm']) for r in rows])
    oos = np.array([float(r[f'oos_{k}_mNm']) for r in rows])
    print(f"{k:9} {b:8.4f} | {np.median(raw):8.2f} "
          f"{np.percentile(raw, 90):8.2f} {raw.max():8.2f} | "
          f"{np.median(oos):8.2f} {np.percentile(oos, 90):8.2f} "
          f"{oos.max():8.2f}")
