#!/usr/bin/env python3
"""
Angular acceleration at the excitation-moment peak
==================================================
Per run, the angular acceleration at the instant the excitation moment
peaks (window edge) -- the signal level against which the combined
worst-case model-error bound on the angular acceleration is compared.
alpha is a local linear fit of omega over +/-0.06 s around the peak
sample (a first-order Savitzky-Golay derivative; averaging over the
window makes the value slightly conservative while omega accelerates).

Outputs (--output-dir): alpha_at_peak_runs.csv + console summary.

Usage
-----
python analysis/alpha_at_peak.py DataSet/exp
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
    prepare_signals, detect_excitation_window)


def main():
    p = argparse.ArgumentParser(
        description="Angular acceleration at the moment peak, per run.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--half-window', type=float, default=0.06,
                   help="half width of the local slope fit [s]")
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
        for bag in bags:
            sig = prepare_signals(bag, axis)
            t, om, M = sig['t'], sig['omega'], sig['moment']
            i0, i1 = detect_excitation_window(M)
            tp = t[i1]
            m = (t >= tp - args.half_window) & (t <= tp + args.half_window)
            if m.sum() < 4:
                continue
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=bag.name,
                M_peak=float(M[i1]),
                alpha=float(np.polyfit(t[m], om[m], 1)[0])))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'alpha_at_peak_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    g = defaultdict(list)
    for r in rows:
        dirn = 'neg' if r['bag'].startswith('neg') else 'pos'
        g[(r['case'], r['axis'], dirn)].append(abs(r['alpha']))
    print(f"\n{'case':8} {'axis':4} {'dir':4} {'n':>2} "
          f"{'min|a|':>8} {'med|a|':>8} {'max|a|':>8}  [rad/s^2]")
    overall = 1e9
    for k in sorted(g):
        v = np.array(g[k])
        overall = min(overall, v.min())
        print(f"{k[0]:8} {k[1]:4} {k[2]:4} {len(v):2d} "
              f"{v.min():8.3f} {np.median(v):8.3f} {v.max():8.3f}")
    print(f"\noverall minimum |alpha| at moment peak: {overall:.3f} rad/s^2")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
