#!/usr/bin/env python3
"""
Post-onset ramp linearity RMSE
==============================
The ramp-execution entry of the coherent-forcing bound rho-bar must be
evaluated over the FIT window [onset, window end] where the deviation
equation lives -- not over the full excitation window, whose linearity
RMSE (up to ~21 mN.m on healthy runs; run gate 30 mN.m) includes the
ramp-start transition and long-span drift that never enter the fit.

Outputs (--output-dir): postonset_linearity_runs.csv + console summary.

Usage
-----
python analysis/postonset_linearity.py DataSet/exp
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import (
    extract_piecewise_batch, detect_excitation_window, commanded_ramp_rate)


def main():
    p = argparse.ArgumentParser(
        description="Ramp linearity RMSE over the post-onset fit segment.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root
    dirs = sorted(d for d in root.glob('case_*/M[xy]') if d.is_dir())
    if not dirs:
        raise SystemExit(f"no datasets under {root}")

    rows = []
    for d in dirs:
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            crits, _ = extract_piecewise_batch(bags, axis)
        for crit in crits:
            i0, i1 = detect_excitation_window(crit.moment)
            j = crit.onset_idx
            if i1 - j < 10:
                continue
            t, M = crit.t[j:i1 + 1], crit.moment[j:i1 + 1]
            a, b = np.polyfit(t, M, 1)
            ft, fM = crit.t[i0:i1 + 1], crit.moment[i0:i1 + 1]
            af, bf = np.polyfit(ft, fM, 1)
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=crit.bag_name,
                rate=commanded_ramp_rate(crit.bag_name) or np.nan,
                post_linrmse=float(np.std(M - (a * t + b))),
                full_linrmse=float(np.std(fM - (af * ft + bf)))))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'postonset_linearity_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    po = np.array([r['post_linrmse'] for r in rows]) * 1e3
    fu = np.array([r['full_linrmse'] for r in rows]) * 1e3
    print(f"\nall {len(rows)} gate-passing runs [mN.m]:")
    print(f"  post-onset linRMSE : median {np.median(po):.2f}  "
          f"p90 {np.percentile(po, 90):.2f}  max {po.max():.2f}")
    print(f"  full-window linRMSE: median {np.median(fu):.2f}  "
          f"p90 {np.percentile(fu, 90):.2f}  max {fu.max():.2f}")
    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r['post_linrmse'] * 1e3)
    print("\n  rate   n   post median / max [mN.m]")
    for k in sorted(g):
        v = np.array(g[k])
        print(f"  {k:4.2f} {len(v):3d}   {np.median(v):8.2f} / {v.max():8.2f}")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
