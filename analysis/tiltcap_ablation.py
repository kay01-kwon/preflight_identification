#!/usr/bin/env python3
"""
Window ablation: identification with a relative tilt cap
========================================================
Truncates every run's signals at the first sample where the excitation-
axis Euler angle departs more than --cap degrees from its pre-excitation
rest value, then runs the unchanged full pipeline (gates -> rig
constants -> constrained cosh). Inside the capped window the small-angle
model is unquestionable; comparing M_crit / M_ff against the full-window
baseline therefore measures how much the wider window (up to ~9 deg of
absolute tilt) influences the identification.

On the reference dataset a 3-deg cap changes the per-direction critical
moments by <= 0.022 N.m (within run-to-run scatter) and M_ff by
<= 0.012 N.m (~0.4 mm of CoM offset).

Outputs (--output-dir): tiltcap_mcrit.csv; with --baseline pointing to a
CSV with columns case,axis,dir,mean (full-window means), also prints the
side-by-side comparison and the M_ff differences.

Usage
-----
python analysis/tiltcap_ablation.py DataSet/exp --cap 3
python analysis/tiltcap_ablation.py DataSet/exp --baseline full_mcrit.csv
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
import critical_value_getter_piecewise as cv


def main():
    p = argparse.ArgumentParser(
        description="Identification with a relative-tilt-capped window.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--cap', type=float, default=3.0,
                   help="relative tilt cap [deg] (default 3)")
    p.add_argument('--baseline', default=None,
                   help="CSV with full-window means (case,axis,dir,mean)")
    p.add_argument('--weight', type=float, default=31.59,
                   help="W [N] for the CoM-equivalent of the M_ff shift")
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    cap_rad = np.deg2rad(args.cap)
    orig_prepare = cv.prepare_signals

    def capped_prepare(bag, axis, **kw):
        sig = orig_prepare(bag, axis, **kw)
        roll, pitch = math_tools.quaternion_to_euler_vectorized(
            bag.odom.quaternion)
        angle = roll if axis == 'x' else pitch
        n = min(len(angle), len(sig['t']))
        i0, _ = cv.detect_excitation_window(sig['moment'][:n])
        rest = float(np.median(angle[:max(1, i0)]))
        over = np.where(np.abs(angle[:n] - rest) > cap_rad)[0]
        over = over[over > i0]
        cut = int(over[0]) if len(over) else n
        return {k: (v[:cut] if isinstance(v, np.ndarray) else v)
                for k, v in sig.items()}

    cv.prepare_signals = capped_prepare

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root
    dirs = sorted(d for d in root.glob('case_*/M[xy]') if d.is_dir())
    if not dirs:
        raise SystemExit(f"no datasets under {root}")

    results = []
    for d in dirs:
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            crits, _ = cv.extract_piecewise_batch(bags, axis)
        by_dir = defaultdict(list)
        for c in crits:
            by_dir['pos' if c.bag_name.startswith('pos') else 'neg'].append(
                c.onset_moment)
        for dirn, vals in sorted(by_dir.items()):
            v = np.array(vals)
            results.append(dict(case=d.parent.name, axis=d.name, dir=dirn,
                                n=len(v), mean=float(v.mean()),
                                std=float(v.std(ddof=1)) if len(v) > 1 else 0.0))
        print(f"  identified {d} (cap {args.cap:g} deg)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'tiltcap_mcrit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'axis', 'dir', 'n',
                                          'mean', 'std'])
        w.writeheader()
        w.writerows(results)
    print(f"Tables -> {out}")

    if not args.baseline:
        return
    base = {}
    for r in csv.DictReader(open(args.baseline)):
        base[(r['case'], r['axis'], r['dir'])] = float(r['mean'])
    print(f"\n== M_crit: {args.cap:g}-deg cap vs full window [N.m] ==")
    mff_c, mff_f = defaultdict(dict), defaultdict(dict)
    for r in results:
        b = base.get((r['case'], r['axis'], r['dir']), np.nan)
        print(f"{r['case']:8} {r['axis']:3} {r['dir']:4} {r['n']:2d} "
              f"{r['mean']:9.4f} {b:9.4f} {r['mean']-b:+8.4f}")
        mff_c[(r['case'], r['axis'])][r['dir']] = r['mean']
        mff_f[(r['case'], r['axis'])][r['dir']] = b
    print(f"\n== M_ff = (pos+neg)/2 ==")
    for k in sorted(mff_c):
        c, fb = mff_c[k], mff_f[k]
        if len(c) == 2 and len(fb) == 2:
            mc = 0.5 * (c['pos'] + c['neg'])
            mf = 0.5 * (fb['pos'] + fb['neg'])
            print(f"{k[0]:8} {k[1]:3} {mc:9.4f} {mf:9.4f} {mc-mf:+8.4f} "
                  f"{(mc-mf)/args.weight*1e3:+7.2f} mm")


if __name__ == '__main__':
    main()
