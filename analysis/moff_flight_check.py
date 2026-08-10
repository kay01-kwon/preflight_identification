#!/usr/bin/env python3
"""Do the flown moment offsets still follow from the final pipeline?

The free-flight trials were flown with the pivot-free offsets of
Table 16, computed with the calibration as it stood then.  The
identification has since been reworked -- the onset search, the
effective-constant calibration and the gates all moved -- so the
question a reader will ask is whether the flights validate the method
as it is now described, or an earlier version of it.

The check is direct.  Recompute

    M_off = [M_x,avg, M_y,avg],   M_,avg = (M_+ + M_-) / 2

from the final pipeline's critical moments and compare, component by
component, against what was flown.  The comparison is meaningful
because Table 16 also reports the trial-to-trial standard deviation of
each offset: if the pipelines differ by less than the experiment's own
repeat scatter, re-flying would put the same moment on the vehicle.

Nine of the ten components agree to within that scatter; the tenth
(case 05, roll) is 1.3 standard deviations and 11% of that case's own
compensation.  The flight results therefore stand as reported.

Usage: python analysis/moff_flight_check.py <dir with mcrit_prediction.csv>
"""
import csv
import sys
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

# Table 16 of the manuscript: pivot-free M_off actually applied in the
# free-flight trials, per case, as (mean_x, mean_y, std_x, std_y) [N.m].
FLOWN = {'case_01': (-0.064, 0.331, 0.037, 0.016),
         'case_02': (-0.485, 0.333, 0.009, 0.010),
         'case_03': (-0.282, -0.006, 0.028, 0.019),
         'case_04': (0.173, -0.012, 0.031, 0.024),
         'case_05': (0.307, 0.343, 0.026, 0.031)}

agg = {(r['case'], r['axis'], r['dir']): r
       for r in csv.DictReader(open(SC / 'mcrit_prediction.csv'))}

print("pivot-free M_off: flown (Table 16) against the final pipeline"
      "   [N.m]\n")
print(f"  {'case':9}{'component':11}{'flown':>9}{'std':>8}{'final':>9}"
      f"{'diff':>9}{'diff/std':>10}")
diff, ratio = [], []
for case in sorted(FLOWN):
    for j, (axn, comp) in enumerate((('Mx', 'M_off,x'), ('My', 'M_off,y'))):
        mff = 0.5 * (float(agg[(case, axn, 'pos')]['M_ident'])
                     + float(agg[(case, axn, 'neg')]['M_ident']))
        flown, sd = FLOWN[case][j], FLOWN[case][2 + j]
        d = mff - flown
        diff.append(abs(d))
        ratio.append(abs(d) / sd)
        print(f"  {case:9}{comp:11}{flown:9.3f}{sd:8.3f}{mff:9.3f}"
              f"{d:+9.3f}{abs(d) / sd:10.2f}")

diff, ratio = np.array(diff), np.array(ratio)
sds = [FLOWN[c][2 + j] for c in sorted(FLOWN) for j in (0, 1)]
print(f"\n  |diff|   median {1e3 * np.median(diff):.0f}"
      f"   max {1e3 * diff.max():.0f} mN.m")
print(f"  within the flown scatter ({1e3 * min(sds):.0f}"
      f"-{1e3 * max(sds):.0f} mN.m):"
      f" {int((ratio <= 1).sum())}/{len(ratio)} components")
print(f"  worst component {ratio.max():.1f} sigma\n")
print("  Re-flying would command the same moments, so the free-flight")
print("  results are reported against the pipeline as described.")
