#!/usr/bin/env python3
"""What sub-sample onset refinement buys, measured on the campaign.

The onset sweep evaluates its cost only at data samples, so the located
onset is short of the true one by up to half a step -- 5 ms at 100 Hz,
which is 6.0 mN.m of threshold at the fastest ramp.  The cost is a
smooth function of the continuous onset (the linearised form is exactly
quadratic in it), so a parabola through the three costs around the grid
minimum has its vertex at the sub-sample offset.  That is now returned
as onset_t alongside the integer onset_idx.

This reprocesses all 140 runs both ways and compares them on the
quantity that is reported: the half-sum M_off = (M_+ + M_-)/2, whose
spread across the seven ramp rates is the model-free accuracy measure of
Sec. VIII.

The result is a null, and it is worth recording as one.  The refinement
works -- on a synthetic onset placed deliberately between samples it
takes the error from 5.00 ms to 0.26 ms, and the corrections it applies
to the campaign are real and near-uniformly distributed, median 0.232
samples against the 0.25 a flat distribution would give.  But the
measured spread does not improve: the per-rate changes run -3.1% to
+3.2%, which is scatter on ten configurations, and the median over all
rates moves 0.5225 to 0.5260 mm.

The arithmetic says why, and it was available beforehand.  The grid
contributes Ts Mdot/sqrt(12) per direction, or 0.081 mm to the half-sum
at the fastest ramp against 0.422 measured -- under 4% of the variance,
so removing it can improve the RMS by at most 2%, which ten
configurations cannot resolve.  The earlier expectation of a modest gain
was optimistic by exactly that margin.

Keep the refinement anyway.  It costs three lines, it removes a term
from the budget analytically rather than leaving it to be argued about,
and it becomes the binding term if the ramp rate is pushed higher or the
logging rate lowered.  What it is not is a way to make the present
numbers better.

Usage: python analysis/subsample_gain.py [DATASET_ROOT]
"""
import contextlib
import collections
import io
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W_MIN = 30.08
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '.subsample_cache.pkl')


def collect(root):
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as fh:
            return pickle.load(fh)
    import critical_value_getter_piecewise as cvp
    from pnls_constants import PNLS_CONSTANTS
    from utils.extractor import load_excitation_dataset
    from pathlib import Path
    rows = []
    for (case, ad), (c2, k) in sorted(PNLS_CONSTANTS.items()):
        axis = 'x' if ad == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(Path(root) / case / ad)
        for bag in bags:
            rate = cvp.commanded_ramp_rate(bag.name)
            if rate is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, mom = sig['t'][w], sig['omega'][w], sig['moment'][w]
            md = float(np.polyfit(t, mom, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                    c2_fixed=c2, moment_floor=0.0,
                                    ramp_gain=k, ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            rows.append(dict(
                case=case, axis=ad, rate=rate, sign=int(np.sign(md)),
                mdot=abs(md),
                m_grid=float(mom[j]),
                m_ref=float(np.interp(pw['onset_t'], t, mom)),
                frac=float(pw['onset_frac'])))
        print(f"  {case}/{ad} done ({len(rows)} runs)")
    with open(CACHE, 'wb') as fh:
        pickle.dump(rows, fh)
    return rows


def spreads(rows, key):
    """Half-sum spread per configuration, and per rate against own mean."""
    hs = collections.defaultdict(dict)
    for r in rows:
        hs[(r['case'], r['axis'])].setdefault(
            round(r['rate'], 2), {})[r['sign']] = r[key]
    per_cfg, per_rate = [], collections.defaultdict(list)
    fast = []
    for cfg, per in hs.items():
        v = {rt: 0.5 * (d[1] + d[-1]) for rt, d in per.items()
             if 1 in d and -1 in d}
        if len(v) < 5:
            continue
        mu = np.mean(list(v.values()))
        per_cfg.append(1e3 * np.std(list(v.values()), ddof=1) / W_MIN)
        f = [x for rt, x in v.items() if rt >= 0.65]
        if len(f) >= 3:
            fast.append(1e3 * np.std(f, ddof=1) / W_MIN)
        for rt, x in v.items():
            per_rate[rt].append(1e3 * (x - mu) / W_MIN)
    return np.array(per_cfg), np.array(fast), per_rate


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'DataSet/exp'
    rows = collect(root)
    fr = np.array([r['frac'] for r in rows])
    print(f"\n  {len(rows)} runs.  sub-sample correction, in samples:")
    print(f"    median |frac| {np.median(np.abs(fr)):.3f},"
          f"  p90 {np.percentile(np.abs(fr), 90):.3f},"
          f"  max {np.abs(fr).max():.3f}")
    print(f"    as time at 100 Hz: median {10*np.median(np.abs(fr)):.2f} ms,"
          f"  p90 {10*np.percentile(np.abs(fr), 90):.2f} ms")
    print(f"    (a flat distribution over [-0.5, 0.5] would give 0.25)")

    print(f"\n  the reported half-sum spread, both ways\n")
    print(f"  {'':>22}{'grid':>10}{'refined':>10}{'change':>10}")
    print(f"  {'':>22}{'mm':>10}{'mm':>10}{'':>10}")
    cg, fg, rg = spreads(rows, 'm_grid')
    cr, fr_, rr = spreads(rows, 'm_ref')
    for nm, a, b in (('median, all 7 rates', np.median(cg), np.median(cr)),
                     ('worst, all 7 rates', cg.max(), cr.max()),
                     ('median, Mdot >= 0.65', np.median(fg), np.median(fr_)),
                     ('worst, Mdot >= 0.65', fg.max(), fr_.max())):
        print(f"  {nm:>22}{a:10.4f}{b:10.4f}{100*(b/a-1):9.1f}%")

    print(f"\n  per-rate error of the half-sum against each"
          f" configuration's own mean\n")
    print(f"  {'Mdot':>6}{'grid RMS':>11}{'refined':>11}{'change':>10}"
          f"{'grid p90':>11}{'refined':>10}")
    print(f"  {'N m/s':>6}{'mm':>11}{'mm':>11}{'':>10}{'mm':>11}{'mm':>10}")
    for rt in sorted(rg):
        a, b = np.array(rg[rt]), np.array(rr[rt])
        ra, rb = np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean())
        print(f"  {rt:6.2f}{ra:11.4f}{rb:11.4f}{100*(rb/ra-1):9.1f}%"
              f"{np.percentile(np.abs(a), 90):11.4f}"
              f"{np.percentile(np.abs(b), 90):10.4f}")

    print(f"\n  what the grid alone should have cost, Ts Mdot/sqrt(12) per")
    print(f"  direction and /sqrt(2) into the half-sum:\n")
    print(f"  {'Mdot':>6}{'predicted':>12}{'observed':>12}")
    print(f"  {'N m/s':>6}{'mm':>12}{'mm':>12}")
    for rt in sorted(rg):
        pred = 1e3 * 0.010 * rt / np.sqrt(12) / np.sqrt(2) / W_MIN
        a, b = np.array(rg[rt]), np.array(rr[rt])
        obs = np.sqrt(max((a ** 2).mean() - (b ** 2).mean(), 0.0))
        print(f"  {rt:6.2f}{pred:12.4f}{obs:12.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
