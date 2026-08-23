#!/usr/bin/env python3
"""Does the fixed-threshold baseline's showing depend on its two knobs?

A benchmark that rests on an arbitrary k is not a benchmark, so the
question a reader will ask -- why three sigma, why three samples? --
has to be answered with a sweep rather than with a convention. This
evaluates the detector over a grid of (k, n_hold) and reports the same
two quantities the estimator table reports: the within-configuration
coefficient of variation of the onset moment, and the delivered CoM
offset against the load-cell truth.

The two knobs pull against each other, which is why neither can be set
by taste alone. Lowering k or n_hold fires earlier, on noise as often
as on motion; raising either waits for the rate to build, which is
exactly the rate-dependent lag that breaks the invariance the
calibration selects for. If the detector's standing in the table is
the same across the grid, the choice of k = 3 and n_hold = 3 carries
no weight and can be reported as a convention; if it is not, the
sweep is the honest thing to publish instead.

The quiet segment is taken up to a coarse onset seed shared by every
detector in the benchmark, as in the CUSUM. That hands the baseline
information the proposed estimator also has, so it is an assist to the
baseline rather than a handicap.

Usage
-----
  PYTHONPATH=<stubs> python analysis/threshold_sensitivity.py [outdir]
"""
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))

from utils.extractor import load_excitation_dataset          # noqa: E402
from analysis.pelt_crosscheck import _window                 # noqa: E402

ROOT = Path('DataSet/exp')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT.mkdir(parents=True, exist_ok=True)

G = 9.81
MASS = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
        'case_04': 3.220, 'case_05': 3.220}
SIGN = {'Mx': +1.0, 'My': -1.0}
TRUTH = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
         ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
         ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
         ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
         ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}

K_GRID = [2.0, 2.5, 3.0, 4.0, 5.0]
HOLD_GRID = [1, 3, 5, 10]


def onset_moment(omega, moment, guess, direction, k, n_hold):
    """The detector, inlined so the knobs can be swept cheaply."""
    base = omega[:max(guess, 5)]
    mu0 = float(np.median(base))
    sigma = float(np.std(base)) + 1e-9
    over = direction * (omega - mu0) > k * sigma
    run = 0
    for t in range(len(omega)):
        run = run + 1 if over[t] else 0
        if run >= n_hold:
            return float(moment[t - n_hold + 1])
    return float(moment[-1])


# ---- one pass over the bags; the windows are what costs -------------
cache = defaultdict(list)
for d in sorted(ROOT.glob('case_*/M[xy]')):
    case, ax = d.parent.name, d.name
    axis = 'x' if ax == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    for bag in bags:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                base, i0, i1, win, guess, direction = _window(bag, axis)
        except Exception as exc:
            print(f'  skipped {bag.name}: {exc}', flush=True)
            continue
        cache[(case, ax)].append(
            (base.omega[win], base.moment[win], guess, direction,
             'pos' if direction > 0 else 'neg'))
    print(f'cached {case}/{ax}  ({len(cache[(case, ax)])} runs)', flush=True)

# ---- sweep ----------------------------------------------------------
rows = []
print(f'\n{"k":>5}{"n_hold":>8}{"CV mean":>10}{"CV worst":>10}'
      f'{"RMS [mm]":>10}{"med":>8}{"max":>8}')
for k in K_GRID:
    for n_hold in HOLD_GRID:
        cvs, errs = [], []
        for (case, ax), runs in cache.items():
            per = defaultdict(list)
            for omega, moment, guess, direction, dirn in runs:
                per[dirn].append(
                    onset_moment(omega, moment, guess, direction, k, n_hold))
            if len(per) < 2:
                continue
            for dirn, v in per.items():
                v = np.asarray(v)
                cvs.append(abs(np.std(v, ddof=1) / np.mean(v)))
            mff = 0.5 * (np.mean(per['pos']) + np.mean(per['neg']))
            off = SIGN[ax] * 1e3 * mff / (MASS[case] * G)
            errs.append(off - TRUTH[(case, ax)])
        c, e = 100 * np.array(cvs), np.abs(np.array(errs))
        rows.append(dict(k=k, n_hold=n_hold, cv_mean=c.mean(),
                         cv_worst=c.max(), rms=np.sqrt(np.mean(e ** 2)),
                         med=np.median(e), max=e.max()))
        print(f'{k:>5.1f}{n_hold:>8d}{c.mean():>10.1f}{c.max():>10.1f}'
              f'{np.sqrt(np.mean(e ** 2)):>10.2f}{np.median(e):>8.2f}'
              f'{e.max():>8.2f}')

with open(OUT / 'threshold_sensitivity.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

best = min(rows, key=lambda r: r['rms'])
print(f"\nbest delivered offset over the grid: k = {best['k']}, "
      f"n_hold = {best['n_hold']}  ->  {best['rms']:.2f} mm RMS, "
      f"CV {best['cv_mean']:.1f}%")
print('for reference, the proposed estimator is 1.65 mm at CV 2.9%')
print(f"written to {OUT / 'threshold_sensitivity.csv'}")
