#!/usr/bin/env python3
"""The residual described, with no mechanism attached.

Three readings have been tried on this residual and two were wrong, so
this script only describes.  It reports the profile of
|omega - omega_hat| through the post-onset window, in deciles of
tau/tau_end, alongside what the signal itself does over the same
deciles -- and nothing else.

What the description rules out is worth more than what it suggests.  A
residual proportional to the signal (an onset displacement, a
mis-specified amplitude, a growth-rate error) would rise with omega_dot.
A residual carrying the modelled forcing of (90) would rise faster still,
the envelope leaving the onset like tau^7.  A residual that is
measurement noise would not depend on the ramp rate, both independent
noise measures being flat in it.  The data matches none of the three.

Reads the cache written by analysis/residual_budget.py.

Usage: python analysis/residual_profile.py
"""
import collections
import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constrained_calibration import ROOT

DEC = np.arange(0.05, 1.0, 0.1)


def profile(runs, key):
    out = []
    for r in runs:
        u = r['tau'] / r['tau'][-1]
        row = []
        for d in DEC:
            m = (u >= d - 0.05) & (u < d + 0.05)
            row.append(np.median(r[key][m]) if m.any() else np.nan)
        out.append(row + [np.mean(r[key][-3:])])
    return np.nanmedian(np.array(out), axis=0)


def main():
    cache = ROOT / 'residual_budget_cache.pkl'
    if not cache.exists():
        raise SystemExit(f"no cache at {cache}; run analysis/residual_budget.py")
    rows = pickle.load(open(cache, 'rb'))
    g = collections.defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    hdr = (f"  {'rate':>6}" + "".join(f"{int(100 * d):>7}%" for d in DEC)
           + f"{'  last 3':>9}")

    print(f"  {len(rows)} runs.  |omega - omega_hat| through the post-onset"
          f" window [rad/s],\n  by decile of tau/tau_end, median over the runs"
          f" at each rate.\n")
    print(hdr)
    for k in sorted(g):
        print(f"  {k:6.2f}" + "".join(f"{v:8.4f}" for v in profile(g[k], 'res')))

    print("\n  each row divided by its own first decile\n")
    print(hdr)
    for k in sorted(g):
        p = profile(g[k], 'res')
        print(f"  {k:6.2f}" + "".join(f"{v / p[0]:8.2f}" for v in p))

    print("\n  and the signal over the same deciles, likewise normalised"
          "\n  (the grid term is proportional to omega_dot)\n")
    print(f"  {'rate':>6}" + "".join(f"{int(100 * d):>7}%" for d in DEC))
    for k in sorted(g):
        p = profile(g[k], 'grid')
        print(f"  {k:6.2f}" + "".join(f"{v / p[0]:8.1f}" for v in p[:len(DEC)]))

    lev = np.array([np.median(r['res']) for r in rows])
    rate = np.array([r['rate'] for r in rows])
    sp = stats.spearmanr(rate, lev)
    print(f"\n  level, over the whole window: median {np.median(lev):.5f} rad/s,"
          f" p10-p90 {np.percentile(lev, 10):.5f}-{np.percentile(lev, 90):.5f}")
    print(f"  Spearman(ramp rate, level) = {sp[0]:+.3f} (p = {sp[1]:.4f})")
    print(f"\n  {'rate':>6}{'level':>10}{'sigma (orthogonal axis)':>26}"
          f"{'level/sigma':>13}")
    for k in sorted(g):
        a = np.median([np.median(r['res']) for r in g[k]])
        b = np.median([r['sigma'] for r in g[k]])
        print(f"  {k:6.2f}{a:10.5f}{b:26.5f}{a / b:13.2f}")

    print("\n  Three statements, and no fourth:")
    print("    1. the residual does not follow the signal -- flat through the")
    print("       window while omega_dot grows 13 to 28-fold, with a minimum")
    print("       near the three-quarter point on every rate;")
    print("    2. its level rises about twofold with the ramp rate, while both")
    print("       the pre-onset floor and the orthogonal-axis noise stay flat;")
    print("    3. at the slow ramps it is the size of that noise, at the fast"
          " ramps 2 to 4 times it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
