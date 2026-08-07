#!/usr/bin/env python3
"""
Small-angle residual evaluated along measured trajectories
==========================================================
The gravity moment about the pivot is an exact function of the measured
excitation-axis angle phi(t):  G(phi) = -W*arm*cos(phi) + W*z*sin(phi).
The estimator absorbs anything in span{1, tau, dphi}; only the
out-of-span part can deform the cosh shape. Because projection is
linear, a sign-convention-free bound per run over the fit window
[onset, window end] is

    |res(G)| <= W*arm*|res(cos phi)| + W*z*|res(sin phi)| .

This replaces the a-priori Lagrange bound (evaluated at the excursion
edge) with the measured coherent forcing, directly comparable to the
ground-effect projection residual (analysis/ge_trajectory.py).

Outputs (--output-dir): small_angle_runs.csv + console summary by rate.

Usage
-----
python analysis/small_angle_trajectory.py DataSet/exp
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
from utils import math_tools
from critical_value_getter_piecewise import (
    extract_piecewise_batch, detect_excitation_window, commanded_ramp_rate)


def main():
    p = argparse.ArgumentParser(
        description="Out-of-span small-angle residual along trajectories.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--weight', type=float, default=31.59, help="W [N]")
    p.add_argument('--z-com', type=float, default=0.30,
                   help="conservative z_CoM [m]")
    p.add_argument('--lp-roll', type=float, default=0.140)
    p.add_argument('--lp-pitch', type=float, default=0.110)
    p.add_argument('--offset-margin', type=float, default=0.020,
                   help="worst |CoM offset| added to the gravity arm [m]")
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    W, Z = args.weight, args.z_com
    ARM = dict(x=args.lp_roll + args.offset_margin,
               y=args.lp_pitch + args.offset_margin)
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
        bag_by = {b.name: b for b in bags}
        for crit in crits:
            bag = bag_by[crit.bag_name]
            i0, i1 = detect_excitation_window(crit.moment)
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
            ph = phi[sl]
            dphi = ph - ph[0]
            A = np.vstack([np.ones_like(tau), tau, dphi]).T

            def out_of_span(y):
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                return y - A @ c

            res = (W * ARM[axis] * np.abs(out_of_span(np.cos(ph)))
                   + W * Z * np.abs(out_of_span(np.sin(ph))))
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=crit.bag_name,
                rate=commanded_ramp_rate(crit.bag_name) or np.nan,
                dphi_max_deg=float(np.rad2deg(np.max(np.abs(dphi)))),
                res_rms=float(np.sqrt(np.mean(res ** 2))),
                res_max=float(res.max())))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'small_angle_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rr = np.array([r['res_rms'] for r in rows]) * 1e3
    rm = np.array([r['res_max'] for r in rows]) * 1e3
    dp = np.array([r['dphi_max_deg'] for r in rows])
    print(f"\nall {len(rows)} gate-passing runs [mN.m]:")
    print(f"  out-of-span residual: median RMS {np.median(rr):.2f}  "
          f"p90 {np.percentile(rr, 90):.2f}  max|.| {rm.max():.2f}")
    print(f"  fit-window excursion: median {np.median(dp):.2f} deg, "
          f"max {dp.max():.2f} deg")
    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r['res_max'] * 1e3)
    print("\n  rate   n   res_max median / max [mN.m]")
    for k in sorted(g):
        v = np.array(g[k])
        print(f"  {k:4.2f} {len(v):3d}   {np.median(v):8.2f} / {v.max():8.2f}")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
