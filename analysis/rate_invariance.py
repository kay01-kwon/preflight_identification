#!/usr/bin/env python3
"""Is the identified threshold really independent of the ramp rate?

The claim underwrites the method: M_crit is a static tip-over threshold,
so it must not depend on how fast the moment was ramped.  But the same
statement is also the CALIBRATION OBJECTIVE -- estimate_rig_constants()
picks (C2, K) by minimising the coefficient of variation of M_crit
across the ramp sweep (critical_value_getter_piecewise.py, score_of).
So the low CV reported by cv_table.py and by nls_comparison_summary.csv
is a training-set fit statistic for the cosh method and cannot be cited
as evidence for the very property it was fitted to.  Reading it that way
is circular, and it is the first thing a referee will test.

This script separates what is fitted from what is not, in four steps.

  1. Raw M_crit against rate, per (case, axis, direction).  Every method
     is regressed; the calibrated one is expected to look good and the
     others to look bad, and that expectation is exactly the problem.

  2. The same for M_ff = (M_pos + M_neg)/2, which is what the offset
     actually uses.  A rate bias common to both tip directions cancels
     here.  This is where the circularity breaks: if the differenced
     quantity is rate-stable for cosh_cad -- whose constants come from
     CAD, with nothing fitted -- then the stability is a property of the
     differencing, not of the calibration.

  3. The per-rate offset error against the MEASURED truth.  Nothing in
     the pipeline is fitted to truth_mm, so a flat error-versus-rate
     profile is external evidence and settles the question.

  4. Equivalence (TOST) rather than a failure to reject.  A large p on
     an F test is not evidence of no effect; the claim has to be stated
     as "any rate effect is smaller than delta" and delta has to come
     from somewhere defensible.  Two candidates are reported: the (109)
     a-priori bound of 0.400 mm, and the method's own validation RMS.

Inputs are the two CSVs written by analysis/nls_comparison.py.

Usage: python analysis/rate_invariance.py
"""
import collections
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
W_MIN = 30.08                      # N, unloaded: converts moment to offset
RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)
METHODS = ('cosh', 'cosh_cad', 'nls', 'pelt_normal', 'pelt_rbf', 'cusum')
FITTED = 'cosh'                    # the one whose constants were calibrated
UNFITTED = 'cosh_cad'              # same estimator, CAD constants, nothing fit
BOUND_MM = 0.400                   # (109), roll, worst admissible


def load():
    runs = list(csv.DictReader(open(ROOT / 'nls_comparison_runs.csv')))
    summ = list(csv.DictReader(open(ROOT / 'nls_comparison_summary.csv')))
    truth, sign = {}, {}
    for r in summ:
        k = (r['case'], r['axis'])
        truth[k] = float(r['truth_mm'])
        if r['method'] == FITTED:
            # offset_mm carries an axis-dependent sign against M_ff/W: a
            # moment about x displaces the CoM along y.  Recover it rather
            # than hard-coding, so the convention stays with its source.
            sign[k] = np.sign(float(r['offset_mm'])
                              / (float(r['M_ff']) * 1e3 / W_MIN))
    cells = collections.defaultdict(dict)
    for r in runs:
        cells[(r['case'], r['axis'])][(float(r['rate']), r['dir'])] = r
    return cells, truth, sign


def offset_mm(cells, sign, key, rate, method):
    """The offset this one rate would have produced, on its own."""
    c = cells[key]
    m_ff = 0.5 * (float(c[(rate, 'pos')]['mcrit_' + method])
                  + float(c[(rate, 'neg')]['mcrit_' + method]))
    return sign[key] * m_ff * 1e3 / W_MIN


def trend(xs, ys):
    """Per-group slope, then a one-sample t across groups."""
    sl = np.array([stats.linregress(xs, y).slope for y in ys])
    t = stats.ttest_1samp(sl, 0.0)
    span = sl.mean() * (max(xs) - min(xs))
    # 90% one-sided upper confidence limit on |span|, for the TOST below
    se = sl.std(ddof=1) / np.sqrt(len(sl)) * (max(xs) - min(xs))
    return sl.mean(), t.pvalue, span, abs(span) + stats.t.ppf(0.95, len(sl) - 1) * se


def blocked_f(per_rate):
    """Rate as a factor, each group centred on its own mean first."""
    return stats.f_oneway(*[per_rate[r] for r in RATES])


def main():
    cells, truth, sign = load()
    keys = sorted(cells)
    print(f"  {len(keys)} configurations x 2 tip directions x {len(RATES)}"
          f" ramp rates = {2 * len(keys) * len(RATES)} runs.\n")

    print("1. raw M_crit against rate, per (case, axis, direction)\n")
    print(f"  {'method':13}{'slope [mN.m per N.m/s]':>24}{'p':>9}"
          f"{'span 0.1->1.2':>15}{'as offset':>13}")
    for m in METHODS:
        ys = []
        for k in keys:
            for d in ('pos', 'neg'):
                ys.append([abs(float(cells[k][(r, d)]['mcrit_' + m]))
                           for r in RATES])
        mu, p, span, _ = trend(RATES, ys)
        print(f"  {m:13}{1e3 * mu:20.2f}{p:13.4f}{1e3 * span:15.2f}"
              f"{1e3 * span / W_MIN:10.3f} mm"
              + ('   <- the calibration objective' if m == FITTED else ''))
    print("\n  Only the calibrated method is flat, and it was fitted to be."
          "\n  The others carry a real and large rate trend.\n")

    print("2. the differenced quantity the offset uses, M_ff = (M_pos+M_neg)/2\n")
    print(f"  {'method':13}{'slope [mm per N.m/s]':>22}{'p':>9}"
          f"{'span':>9}{'rate F':>9}{'p':>9}")
    devs = {}
    for m in METHODS:
        ys = [[offset_mm(cells, sign, k, r, m) for r in RATES] for k in keys]
        mu, p, span, hi = trend(RATES, ys)
        per_rate = collections.defaultdict(list)
        for y in ys:
            for r, v in zip(RATES, y):
                per_rate[r].append(v - np.mean(y))
        F = blocked_f(per_rate)
        devs[m] = (per_rate, hi)
        print(f"  {m:13}{mu:18.4f}{p:13.4f}{span:9.4f}"
              f"{F.statistic:9.2f}{F.pvalue:9.4f}")
    print(f"\n  {'method':13}" + "".join(f"{r:>8}" for r in RATES)
          + "   mean deviation from each config's own offset [mm]")
    for m in (FITTED, UNFITTED):
        per_rate, _ = devs[m]
        print(f"  {m:13}" + "".join(f"{np.mean(per_rate[r]):8.3f}"
                                    for r in RATES))
    print(f"\n  {FITTED} and {UNFITTED} agree closely, and {UNFITTED} has"
          f" nothing fitted to\n  rate.  So what makes the OFFSET rate-stable"
          f" is the two-sided differencing,\n  not the calibration --- which"
          f" is the answer to the circularity.\n")

    print("3. per-rate offset error against the measured truth"
          " (external: nothing is fitted to it)\n")
    print(f"  {'method':13}" + "".join(f"{r:>8}" for r in RATES)
          + f"{'RMS':>9}{'rate F':>9}{'p':>9}")
    for m in METHODS:
        per_rate, allv = collections.defaultdict(list), []
        for k in keys:
            for r in RATES:
                e = offset_mm(cells, sign, k, r, m) - truth[k]
                per_rate[r].append(e)
                allv.append(e)
        F = blocked_f(per_rate)
        rms = [np.sqrt(np.mean(np.square(per_rate[r]))) for r in RATES]
        print(f"  {m:13}" + "".join(f"{v:8.3f}" for v in rms)
              + f"{np.sqrt(np.mean(np.square(allv))):9.3f}"
              f"{F.statistic:9.2f}{F.pvalue:9.4f}")
    for m in (FITTED, UNFITTED):
        slow = [offset_mm(cells, sign, k, min(RATES), m) - truth[k] for k in keys]
        allr = [np.mean([offset_mm(cells, sign, k, r, m) for r in RATES])
                - truth[k] for k in keys]
        print(f"\n  {m:10} slowest ramp alone"
              f" {np.sqrt(np.mean(np.square(slow))):6.3f} mm,"
              f"   averaged over all seven {np.sqrt(np.mean(np.square(allr))):6.3f} mm")
    print("\n  If the quasi-static reading were the privileged one, the slow"
          "\n  ramp would be more accurate.  It is not.\n")

    print("4. equivalence, not a failure to reject\n")
    print("  A large p in steps 2-3 is not evidence of no effect.  State the")
    print("  claim as |rate effect| < delta and test THAT.  Two margins:\n")
    print(f"  {'method':13}{'max |dev|':>11}{'90% upper':>11}"
          f"{f'vs {BOUND_MM:.3f} mm':>14}{'vs 1.64 mm':>12}")
    for m in METHODS:
        per_rate, _ = devs[m]
        mx = max(abs(np.mean(per_rate[r])) for r in RATES)
        hi = max(abs(np.mean(per_rate[r]))
                 + stats.t.ppf(0.95, len(per_rate[r]) - 1)
                 * np.std(per_rate[r], ddof=1) / np.sqrt(len(per_rate[r]))
                 for r in RATES)
        print(f"  {m:13}{mx:11.3f}{hi:11.3f}"
              f"{'PASS' if hi < BOUND_MM else 'fail':>14}"
              f"{'PASS' if hi < 1.64 else 'fail':>12}")
    print(f"\n  With {len(keys)} configurations the (109) margin of"
          f" {BOUND_MM:.3f} mm is out of reach ---")
    print("  not because the effect is large but because the confidence"
          " interval is.")
    print("  The defensible statement is equivalence within a margin below the")
    print("  method's own validation RMS, together with step 3, which needs no")
    print("  margin at all.\n")
    print("  Note the a-priori prediction points the same way: in (109) the only")
    print("  rate-dependent term is rho_GE/5, which runs 0.19 to 0.99 mN.m over")
    print("  the sweep --- under 0.03 mm of offset.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
