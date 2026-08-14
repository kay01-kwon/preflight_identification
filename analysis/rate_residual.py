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

The bound is printed twice.  The a-priori column is VI-E as published,
set at the 10-degree tilt cap.  The runs stop at 4.5 to 5.6 degrees, so
that column overstates the disturbance by roughly the square of the
ratio and is not the honest comparison; the realised column re-solves
the window from each rate's measured peak and evaluates the same bound
there.  Expect the measured residual to sit near the realised bound at
the slowest ramp and well above it at the fastest -- the bound covers
only the modelled forcing, and what dominates in practice is rate-flat.

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
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critical_value_getter_piecewise import (commanded_ramp_rate,
                                             detect_excitation_window,
                                             extract_piecewise_batch)
from utils.extractor import load_excitation_dataset

# Sec. VI-D box, roll, for re-deriving the VI-E bound alongside.
W, G, Z, ARM = 31.59, 9.81, 0.30, 0.160
BETA_M, J_CAD = 0.03446, 0.0537
WZ = W * Z
J_LO = (W / G) * (Z ** 2 + ARM ** 2)
C2 = np.sqrt(WZ / J_LO)


def _lam(u):
    return np.sinh(u) - u


def _r_phi(x):
    A = (np.sinh(2 * x) / 4 - x / 2 - 2 * x * np.cosh(x)
         + 2 * np.sinh(x) + x ** 3 / 3)
    return A / (x * _lam(x) ** 2)


def _r_ge(x):
    return (x * np.cosh(x) - np.sinh(x) - x ** 3 / 3) / (x ** 2 * _lam(x))


def bound_at(rate, peak=None, cap_deg=10.0):
    """The VI-E relative rate bound.

    With `peak` given, the window is the one the runs actually reached,
    inferred from the measured peak rate through omega = C1(cosh x - 1);
    this is the comparison that means something, because the a-priori
    figure is set at the tilt cap and the runs stop well short of it.
    With `peak` omitted, the a-priori figure at the cap is returned.
    """
    c1 = rate / WZ
    if peak is None:
        phi = np.deg2rad(cap_deg)
        x = brentq(lambda t: _lam(t) - phi * WZ * C2 / rate, 1e-9, 40.0)
        d_m = (6 * phi * (J_CAD + J_LO) * rate ** 2) ** (1 / 3)
        peak = c1 * (np.cosh(x) - 1.0)
    else:
        x = brentq(lambda t: c1 * (np.cosh(t) - 1.0) - peak, 1e-9, 40.0)
        phi = (c1 / C2) * _lam(x)
        d_m = rate * x / C2
    rho = 0.5 * W * ARM * phi ** 2 * _r_phi(x) + BETA_M * phi * d_m * _r_ge(x)
    return 100.0 * rho * np.sinh(x) / (J_LO * C2) / peak, np.rad2deg(phi)


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
    print(f"  {'rate':>6}{'n':>4}{'peak':>9}{'phi_end':>9}{'endpoint %':>21}"
          f"{'RMS %':>8}{'noise %':>9}{'bound @cap':>12}{'@realised':>11}")
    print(f"  {'':6}{'':4}{'[rad/s]':>9}{'[deg]':>9}{'median':>11}{'p90':>10}"
          f"{'median':>8}{'median':>9}{'%':>12}{'%':>11}")
    for rate in sorted(g):
        v = g[rate]
        pk = np.array([r['span'] for r in v])
        en = np.array([r['end_pct'] for r in v])
        rm = np.array([r['rms_pct'] for r in v])
        fl = np.array([r['floor_pct'] for r in v])
        cap, _ = bound_at(rate)
        real, phi = bound_at(rate, peak=float(np.median(pk)))
        print(f"  {rate:6.2f}{len(v):4d}{np.median(pk):9.3f}{phi:9.2f}"
              f"{np.median(en):11.2f}{np.percentile(en, 90):10.2f}"
              f"{np.median(rm):8.2f}{np.median(fl):9.2f}"
              f"{cap:12.1f}{real:11.1f}")
    print("\n  The runs stop well short of the 10-degree cap, so the a-priori"
          "\n  column is not the one to compare against; the realised column"
          " is.")

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
