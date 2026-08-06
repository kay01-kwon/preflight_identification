#!/usr/bin/env python3
"""Can the static check attribute the threshold deficit to ground effect?

The per-direction static check (analysis/mcrit_prediction.py) shows the
rigid no-GE prediction over-predicting every directional threshold.  Two
physically distinct explanations produce the same signature:

  H1  ground effect: the thrust channel a = c_a f l reduces the moment
      needed to tip, plus a moment-proportional channel b M;
  H2  contact geometry: the normal-force resultant acts a distance dl
      INBOARD of the kinematic pivot circle (finite, compliant contact
      patch), i.e. the static lever is l - dl.

Both scale as force x length, so they are degenerate in the constant
part.  The only discriminating structure is the moment-proportional
channel of H1, which grows with |M_crit| (a 4.3x range across groups)
while H2 does not.  This script fits both and reports whether the data
can tell them apart.

Reads mcrit_prediction.csv from argv[1].
"""
import csv
import sys

import numpy as np

SC = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = list(csv.DictReader(open(f'{SC}/mcrit_prediction.csv')))

M = np.array([abs(float(r['M_ident'])) for r in rows])
f = np.array([float(r['f_onset']) for r in rows])
W = np.array([float(r['W']) for r in rows])
l = np.array([float(r['l_odom_mm']) * 1e-3 for r in rows])
sgn = np.array([1.0 if r['dir'] == 'pos' else -1.0 for r in rows])
# magnitude deficit |M_ident| - |M_pred(no GE)|  (negative = over-predicted)
d = sgn * np.array([float(r['resid_mNm']) for r in rows]) * 1e-3
n = len(d)


def report(name, X, names, dof):
    c, *_ = np.linalg.lstsq(X, d, rcond=None)
    r = d - X @ c
    rms = np.sqrt(np.mean(r ** 2))
    se = np.sqrt(np.sum(r ** 2) / (n - dof)
                 * np.diag(np.linalg.pinv(X.T @ X)))
    txt = ',  '.join(f"{nm} = {v:.4g} +- {s:.2g}"
                     for nm, v, s in zip(names, c, se))
    print(f"{name:34} {txt}   |  residual RMS {1e3 * rms:5.1f} mN·m")
    return rms


print(f"n = {n} direction groups; magnitude deficit "
      f"{1e3 * d.min():.0f} .. {1e3 * d.max():.0f} mN·m "
      f"(mean {1e3 * d.mean():.1f}, std {1e3 * d.std(ddof=1):.0f})\n")

report('H1 ground effect (c_a, b free)', np.column_stack([-f * l, -M]),
       ['c_a', 'b'], 2)
report('H1 with model coefficients', np.column_stack([-(0.0431 * f * l
                                                        + 0.0431 * M)]),
       ['scale'], 1)
report('H2 contact-lever offset', np.column_stack([-(W - f)]), ['dl [m]'], 1)
report('H1+H2 (degenerate constant)', np.column_stack([-np.ones(n), -M]),
       ['const [N·m]', 'b'], 2)

sig = 0.0431 * M
print(f"\nb-channel signal across groups: {1e3 * sig.min():.0f} .. "
      f"{1e3 * sig.max():.0f} mN·m, spread {1e3 * (sig.max() - sig.min()):.0f}"
      f" — group-to-group scatter of the deficit is "
      f"{1e3 * d.std(ddof=1):.0f} mN·m")
print("=> the discriminating channel sits below the scatter: the check "
      "bounds the\n   combined ground-contact budget but cannot attribute "
      "it to either cause.")
