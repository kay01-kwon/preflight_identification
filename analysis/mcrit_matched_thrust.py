#!/usr/bin/env python3
"""The matched-thrust reading of the static GE prediction, verified.

The pipeline's prediction (analysis/mcrit_prediction.py)

    M_pred = (T_rigid - sgn c_a f l) / (1 + b)

is identical to the two-equation system in which the theoretical
threshold is defined AT MATCHED TOTAL THRUST -- the reference balance
carries the same GE-augmented thrust moment (1 + c_a) f l as the
experiment, so only the moment channel separates the two:

    (1 + b) M_crit + (1 + c_a) f l = W (l + p_off)      [experiment]
    M_theory       + (1 + c_a) f l = W (l + p_off)      [reference]
    =>  M_crit = M_theory / (1 + b).

This script loads mcrit_prediction.csv, reproduces M_pred_interf from
the user-form expression (identity to < 0.1 mN.m, limited only by the
CSV's group-mean rounding), and quantifies what the matching is worth:
dropping the c_a term from the reference shifts every directional
prediction by c_a f l/(1+b) = 90-127 mN.m and degrades the
interference residual median from 59 to 147 mN.m.

Usage: python analysis/mcrit_matched_thrust.py <dir with mcrit_prediction.csv>
"""
import csv
import sys
from pathlib import Path

import numpy as np

CA, B = 0.0431, 0.04314            # interference model, phi* = 0


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    rows = list(csv.DictReader(open(d / 'mcrit_prediction.csv')))
    print(f"{'case/ax/dir':<20}{'M_ident':>9}{'pred(csv)':>10}"
          f"{'matched':>9}{'unmatched':>11}{'shift[mNm]':>11}")
    mx, res_m, res_u = 0.0, [], []
    for r in rows:
        f, l = float(r['f_onset']), float(r['l_odom_mm']) * 1e-3
        sgn = 1.0 if r['dir'] == 'pos' else -1.0
        T = float(r['M_pred'])                 # rigid, no GE
        csvp = float(r['M_pred_interf'])
        user = (T - sgn * CA * f * l) / (1.0 + B)
        unm = T / (1.0 + B)                    # thrust channel dropped
        mi = float(r['M_ident'])
        mx = max(mx, abs(user - csvp))
        res_m.append(abs(mi - user))
        res_u.append(abs(mi - unm))
        print(f"{r['case']+'/'+r['axis']+'/'+r['dir']:<20}{mi:9.3f}"
              f"{csvp:10.3f}{user:9.3f}{unm:11.3f}"
              f"{1e3*(unm-user)*sgn:11.1f}")
    print(f"\n  identity: max |matched-form - pipeline| = {1e3*mx:.2f} mN.m")
    print(f"  |resid| median: matched {1e3*np.median(res_m):.1f} "
          f"vs unmatched {1e3*np.median(res_u):.1f} mN.m")
    return 0


if __name__ == '__main__':
    sys.exit(main())
