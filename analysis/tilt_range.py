#!/usr/bin/env python3
"""
Measured tilt range over the excitation window
==============================================
Justifies the tilt range used by the model-error bounds empirically:
per run, reports the rest tilt (pre-excitation median, i.e. the tilt at
which the onset occurs) and the tilt at the excitation-window end (the
largest tilt the identification fit ever sees). Onset-free (window
detection only), so it covers every run quickly.

Context: the excitation terminates on a 5-deg absolute attitude
threshold; the sloped-surface rest attitude and the rotor spin-down
sweep (growing with ramp rate) put the absolute window-edge tilt above
that nominal threshold, which is why the bounds must be evaluated over
the measured range rather than the termination setting.

Outputs (--output-dir): tilt_range_runs.csv + console summary.

Usage
-----
python analysis/tilt_range.py DataSet/exp
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
    prepare_signals, detect_excitation_window, _tilt_deg)


def main():
    p = argparse.ArgumentParser(
        description="Measured tilt range over the excitation window.")
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
        for bag in bags:
            sig = prepare_signals(bag, axis)
            tilt = _tilt_deg(bag.odom.quaternion)
            n = min(len(tilt), len(sig['moment']))
            i0, i1 = detect_excitation_window(sig['moment'][:n])
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=bag.name,
                rest_tilt=float(np.median(tilt[:max(1, i0)])),
                end_tilt=float(tilt[min(i1, n - 1)]),
                win_max_tilt=float(np.max(tilt[i0:min(i1 + 1, n)]))))
        print(f"  assessed {d} ({len(rows)} runs)")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'tilt_range_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rest = np.array([r['rest_tilt'] for r in rows])
    wmax = np.array([r['win_max_tilt'] for r in rows])
    print(f"\nall {len(rows)} runs [deg]:")
    print(f"  rest tilt      : median {np.median(rest):.2f}  "
          f"p90 {np.percentile(rest, 90):.2f}  max {rest.max():.2f}")
    print(f"  window max tilt: median {np.median(wmax):.2f}  "
          f"p90 {np.percentile(wmax, 90):.2f}  max {wmax.max():.2f}")
    g = defaultdict(list)
    for r in rows:
        g[(r['axis'], r['bag'][:3])].append(r['end_tilt'] - r['rest_tilt'])
    print("\n  axis dir   n   departure median / max [deg]")
    for k in sorted(g):
        v = np.array(g[k])
        print(f"  {k[0]:4} {k[1]:4} {len(v):3d}   {np.median(v):8.2f} / {v.max():8.2f}")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
