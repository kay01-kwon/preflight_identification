#!/usr/bin/env python3
"""
Cosh-fidelity certificate: zero-free-parameter fit residual
===========================================================
For every gate-passing run, after the full identification pipeline
(quality gates -> rig-constant estimation -> constrained cosh fit):

  * pre-onset noise floor = std of omega about its median, before onset;
  * post-onset fit RMSE   = RMSE of (omega - omega_pred) after onset;
  * NRMSE                 = post RMSE / post-onset peak excursion;
  * R^2 of the post-onset fit.

Because the constrained fit has NO free shape parameter per run
(C1 = K*Mdot with Mdot measured, C2 shared, baseline by continuity),
the post-onset curve is a prediction, not a fit: were any state-
dependent effect (ground effect, small-angle residual, rate dynamics)
deforming the response away from the cosh family, the normalized
residual would grow with excitation amplitude / ramp rate. Rate-flat
NRMSE is therefore the empirical certificate that the response stays
cosh-shaped.

Outputs (--output-dir): cosh_fidelity_runs.csv + console summary by rate.

Usage
-----
python analysis/cosh_fidelity.py DataSet/exp
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
        description="Constrained-cosh fit residual vs noise floor.")
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
            crits, fits = extract_piecewise_batch(bags, axis)
        for crit, pw in zip(crits, fits):
            if pw.get('model') != 'cosh' or 'omega_pred' not in pw:
                continue
            i0, i1 = detect_excitation_window(crit.moment)
            om = crit.omega[i0:i1 + 1]
            pred = pw['omega_pred']
            if len(pred) != len(om):
                continue
            j = pw['onset_idx']
            if j < 5 or len(om) - j < 5:
                continue
            pre = float(np.std(om[:j] - np.median(om[:j])))
            post = float(np.sqrt(np.mean((om[j:] - pred[j:]) ** 2)))
            base = float(np.median(om[:j]))
            span = float(np.max(np.abs(om[j:] - base)))
            ss_res = float(np.sum((om[j:] - pred[j:]) ** 2))
            ss_tot = float(np.sum((om[j:] - np.mean(om[j:])) ** 2))
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=crit.bag_name,
                rate=commanded_ramp_rate(crit.bag_name) or np.nan,
                pre_floor=pre, post_rmse=post,
                ratio=post / pre if pre > 0 else np.nan,
                nrmse_pct=100.0 * post / span if span > 0 else np.nan,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'cosh_fidelity_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    print(f"\n{'rate':>5} {'n':>3} {'post-RMSE':>10} {'ratio med':>9} "
          f"{'NRMSE% med':>10} {'R2 med':>7}")
    for rate in sorted(g):
        v = g[rate]
        po = np.array([r['post_rmse'] for r in v])
        ra = np.array([r['ratio'] for r in v])
        nr = np.array([r['nrmse_pct'] for r in v])
        r2 = np.array([r['r2'] for r in v])
        print(f"{rate:5.2f} {len(v):3d} {po.mean():10.4f} {np.median(ra):9.2f} "
              f"{np.median(nr):10.2f} {np.median(r2):7.4f}")
    nr = np.array([r['nrmse_pct'] for r in rows])
    r2 = np.array([r['r2'] for r in rows])
    print(f"\nall {len(rows)} runs: NRMSE median {np.median(nr):.2f}%  "
          f"p90 {np.percentile(nr, 90):.2f}%;  R2 median {np.median(r2):.4f}")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
