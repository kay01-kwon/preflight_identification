#!/usr/bin/env python3
"""The measured angular-rate residual, put next to the Sec. VI-E bound.

Sec. VI-E bounds |e_omega(tau_end)|, the deviation AT THE WINDOW END,
relative to the nominal rate there.  It predicts that the relative
figure worsens as the ramp slows -- 4.7% at 1.20 N.m/s rising to 35.9%
at 0.10 -- not because the disturbance grows but because the moment
swept before the tilt cap is reached collapses as Mdot^(2/3) while the
unstable plant is given more e-foldings to amplify whatever is there.

That is a bound over the admissible box.  This script measures what the
runs actually do, in the same currency, so the two can be compared:

  endpoint    |omega - omega_pred| at the last post-onset sample,
              divided by the nominal peak.  This is the quantity VI-E
              bounds, and the only one for which the comparison is
              apples to apples.
  window RMS  RMS of the same residual over the post-onset window,
              divided by the same peak -- the NRMSE that
              cosh_fidelity.py reports.  Always smaller, since the
              deviation is concentrated at the end.
  noise floor pre-onset std, so that a residual can be told apart from
              the gyro.

The fit has no free shape parameter per run (C1 = K*Mdot with Mdot
measured, C2 shared across the configuration, baseline by continuity),
so the post-onset curve is a prediction rather than a fit and the
residual is meaningful.

Usage: python analysis/rate_residual.py [DataSet/exp]
"""
import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critical_value_getter_piecewise import (commanded_ramp_rate,
                                             detect_excitation_window,
                                             extract_piecewise_batch)
from utils.extractor import load_excitation_dataset

# Sec. VI-E, roll, evaluated at the exact window of (110).  Interpolated
# in log Mdot for the rates the sweep uses that VI-E does not tabulate.
BOUND = {0.10: 35.9, 0.45: 10.3, 1.20: 4.7}


def bound_at(rate):
    xs = np.log(sorted(BOUND))
    ys = [BOUND[r] for r in sorted(BOUND)]
    return float(np.interp(np.log(rate), xs, ys))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp')
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
            base = float(np.median(om[:j]))
            res = om[j:] - pred[j:]
            span = float(np.max(np.abs(om[j:] - base)))
            if span <= 0:
                continue
            # The endpoint is taken as the mean of the last three samples,
            # so a single noisy sample cannot carry the statistic.
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=crit.bag_name,
                rate=commanded_ramp_rate(crit.bag_name) or np.nan,
                span=span,
                end_pct=100.0 * abs(float(np.mean(res[-3:]))) / span,
                rms_pct=100.0 * float(np.sqrt(np.mean(res ** 2))) / span,
                floor_pct=100.0 * float(np.std(om[:j] - base)) / span))
        print(f"  assessed {d} ({len(rows)} runs so far)")

    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)

    print(f"\n  measured residual against the VI-E bound, by ramp rate\n")
    print(f"  {'rate':>6}{'n':>4}{'peak':>9}{'endpoint %':>21}"
          f"{'window RMS %':>15}{'noise %':>9}{'VI-E bound %':>14}")
    print(f"  {'':6}{'':4}{'[rad/s]':>9}{'median':>11}{'p90':>10}"
          f"{'median':>15}{'median':>9}{'':>14}")
    for rate in sorted(g):
        v = g[rate]
        pk = np.array([r['span'] for r in v])
        en = np.array([r['end_pct'] for r in v])
        rm = np.array([r['rms_pct'] for r in v])
        fl = np.array([r['floor_pct'] for r in v])
        print(f"  {rate:6.2f}{len(v):4d}{np.median(pk):9.3f}"
              f"{np.median(en):11.2f}{np.percentile(en, 90):10.2f}"
              f"{np.median(rm):15.2f}{np.median(fl):9.2f}"
              f"{bound_at(rate):14.1f}")

    en = np.array([r['end_pct'] for r in rows])
    rm = np.array([r['rms_pct'] for r in rows])
    print(f"\n  {len(rows)} runs.  endpoint median {np.median(en):.2f}%,"
          f" p90 {np.percentile(en, 90):.2f}%, max {en.max():.2f}%;"
          f"  window RMS median {np.median(rm):.2f}%")

    slow = [r['end_pct'] for r in rows if r['rate'] <= 0.30]
    fast = [r['end_pct'] for r in rows if r['rate'] >= 0.65]
    print(f"  slow half (Mdot <= 0.30) endpoint median"
          f" {np.median(slow):.2f}%,  fast half (>= 0.65)"
          f" {np.median(fast):.2f}%")

    worst = max(rows, key=lambda r: r['end_pct'])
    print(f"  worst single run: {worst['case']}/{worst['axis']}"
          f" {worst['bag']} at {worst['end_pct']:.1f}%")

    out = root / 'rate_residual_runs.csv'
    import csv
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  per-run table -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
