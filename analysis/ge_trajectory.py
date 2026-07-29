#!/usr/bin/env python3
"""
Exact ground-effect moment along measured trajectories
======================================================
No Taylor expansion, no operating point. Per sample of every run:

    T_i(t)  = C_T * rpm_i(t)^2                      (measured rotor speeds)
    h_i(t)  = r3(t) . p_i^P                         (measured attitude; r3 =
              third row of R_wb, yaw-free), p_i^P = hub position w.r.t. the
              active pivot line with z = hub height
    dT_i    = T_i * R^2 / (16 h_i^2 - R^2)          (exact IGE model)
    dtau(t) = sum_i b_i * dT_i                      (moment about pivot axis)

The estimator has exactly three per-run degrees of freedom -- M_crit
(constant), C1 = K*Mdot (linear in tau) and C2 (linear in the measured
dphi) -- so projecting dtau onto the basis {1, tau, dphi(t)} is an
algebraic identity: anything in the span is absorbed, and only the
projection residual can perturb the identified onset. The regression is
restricted to the steady-collective portion of the excitation window
(the regime containing the identification fit window), excluding the
throttle-up transient.

Outputs (--output-dir): ge_trajectory_runs.csv and a console summary of
the GE level, the slope ratio slope/Mdot (the measured eta), and the
basis-projection residual, by commanded ramp rate.

Usage
-----
python analysis/ge_trajectory.py DataSet/exp
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
    prepare_signals, detect_excitation_window, commanded_ramp_rate)


def main():
    p = argparse.ArgumentParser(
        description="Exact GE moment along measured trajectories.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--arm', type=float, default=0.265)
    p.add_argument('--radius', type=float, default=0.127)
    p.add_argument('--hub-height', type=float, default=0.315)
    p.add_argument('--lp-roll', type=float, default=0.140)
    p.add_argument('--lp-pitch', type=float, default=0.110)
    p.add_argument('--c-t', type=float, default=1.3175e-7)
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    L, R, H = args.arm, args.radius, args.hub_height
    LP = dict(x=args.lp_roll, y=args.lp_pitch)
    ang = np.deg2rad(30 + 60 * np.arange(6))
    LX, LY = L * np.cos(ang), L * np.sin(ang)

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
        for bag in bags:
            sig = prepare_signals(bag, axis)
            t, M = sig['t'], sig['moment']
            i0, i1 = detect_excitation_window(M)
            win = slice(i0, i1 + 1)
            if i1 - i0 < 20:
                continue
            pos = bag.name.startswith('pos')
            lp = LP[axis]
            if axis == 'x':
                px, py = LX, LY + (lp if pos else -lp)
                arm = py
            else:
                px, py = LX + (-lp if pos else lp), LY
                arm = -px
            pP = np.vstack([px, py, np.full(6, H)])

            t0 = bag.odom.t[0]
            T6 = args.c_t * bag.rpm.rpm.astype(np.float64) ** 2
            Ti = np.vstack([np.interp(t, bag.rpm.t - t0, T6[:, j])
                            for j in range(6)]).T

            qw, qx, qy, qz = bag.odom.quaternion.T
            r31 = 2 * (qx * qz - qw * qy)
            r32 = 2 * (qy * qz + qw * qx)
            r33 = 1 - 2 * (qx * qx + qy * qy)
            n = min(len(r31), len(t))
            h_it = (np.outer(r31[:n], pP[0]) + np.outer(r32[:n], pP[1])
                    + np.outer(r33[:n], pP[2]))
            if np.any(h_it[win] <= R / 4 + 0.01):
                continue
            dT = Ti[:n] * R**2 / (16 * h_it**2 - R**2)
            dtau = dT @ arm

            # steady-collective portion only (excludes throttle-up transient)
            fcw = sig['f_col'][win]
            steady = np.where(fcw >= 0.95 * np.median(fcw))[0]
            if len(steady) < 20:
                continue
            j0 = int(steady[0])
            tw = t[win][j0:]
            yw = dtau[win][j0:][:len(tw)]
            tau = tw - tw[0]
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                bag.odom.quaternion[:n])
            phi = (roll if axis == 'x' else pitch)[win][j0:][:len(tw)]
            dphi = phi - phi[0]

            A2 = np.vstack([np.ones_like(tau), tau]).T
            A3 = np.vstack([np.ones_like(tau), tau, dphi]).T
            r2 = yw - A2 @ np.linalg.lstsq(A2, yw, rcond=None)[0]
            c3, *_ = np.linalg.lstsq(A3, yw, rcond=None)
            r3v = yw - A3 @ c3
            mdot = float(np.polyfit(tw, M[win][j0:][:len(tw)], 1)[0])
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=bag.name,
                rate=commanded_ramp_rate(bag.name) or np.nan,
                ge_mean=float(np.mean(yw)),
                ge_span=float(yw.max() - yw.min()),
                aff_rms=float(np.std(r2)),
                res_rms=float(np.std(r3v)),
                res_max=float(np.max(np.abs(r3v))),
                slope_ratio_pct=float(c3[1] / mdot * 100.0)))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'ge_trajectory_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def arr(k):
        return np.array([r[k] for r in rows])

    print(f"\nall {len(rows)} runs  [mN.m]")
    print(f"  GE level          : median {np.median(arr('ge_mean'))*1e3:6.2f}  "
          f"range [{arr('ge_mean').min()*1e3:.2f}, {arr('ge_mean').max()*1e3:.2f}]")
    print(f"  {{1,t}} residual    : median RMS {np.median(arr('aff_rms'))*1e3:.3f}")
    print(f"  {{1,t,dphi}} resid  : median RMS {np.median(arr('res_rms'))*1e3:.3f}  "
          f"max|.| {arr('res_max').max()*1e3:.3f}")
    print(f"  slope / Mdot      : median {np.median(arr('slope_ratio_pct')):+.2f}%  "
          f"IQR [{np.percentile(arr('slope_ratio_pct'),25):+.2f}, "
          f"{np.percentile(arr('slope_ratio_pct'),75):+.2f}]%")
    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    print("\n  rate   n   res RMS med   res max   slope/Mdot med")
    for rate in sorted(g):
        v = g[rate]
        rr = np.array([x['res_rms'] for x in v]) * 1e3
        rm = np.array([x['res_max'] for x in v]) * 1e3
        sr = np.array([x['slope_ratio_pct'] for x in v])
        print(f"  {rate:4.2f} {len(v):3d}   {np.median(rr):9.3f} {rm.max():9.3f}"
              f"   {np.median(sr):+8.2f}%")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
