#!/usr/bin/env python3
"""Fit the contact lever from the balance, in the time domain.

analysis/lever_solve.py asked the same question with amplitude spectra
and got 262 mm on My, longer than the 265 mm arm, which is impossible:
adding |m| and |f| l_p ignores their relative phase.  Solve it properly
instead.  The pivot balance, with the gravity arm written a = l_p + off,

    0 = J_P wdot - m - f l_p + W (l_p + off) cos(phi)
        - W z sin(phi) - dM_GE

is linear in l_p, so least squares over each run's window gives

    l_p = <num, den> / <den, den>,
    num = dM_GE + m - J_P wdot + W z sin(phi) - W off cos(phi),
    den = W cos(phi) - f              (about 10.5 N here)

Fitted below 3 deg, per run, the answer is physical everywhere:

    class     assumed   fitted (p25 / median / p75)
    Mx/pos      138.9      137 / 145 / 151
    Mx/neg      141.2      130 / 134 / 143
    My/pos      118.9       90 / 101 / 106
    My/neg      106.5      126 / 131 / 139

all inside or near the landing-gear footprint, and nothing near 262.

What the fit then says is that the direction asymmetry of the lever is
not the one mocap reports.  On My mocap gives pos 118.9 and neg 106.5,
the positive direction longer by 12 mm; the balance wants pos 101 and
neg 131, the negative direction longer by 30.  Converting the
antisymmetric residual at the balance's own sensitivity, 240 mN.m /
10.5 N = 23 mm on My, and comparing group by group against the
fitted-minus-mocap half-difference gives corr -0.975 and slope -1.11 --
equal and opposite, which is what cancelling it requires.

So the antisymmetric residual is the contact lever's direction
asymmetry, about +-20 mm of it on My and +-5 on Mx, of which mocap
captures roughly a quarter.  This is a consistency statement, not a
proof: it shows the residual has the size and the group-by-group
pattern of a lever error, and that the lever it implies is physically
possible.  It does not change the deliverable either way, since an
antisymmetric term cancels in the pivot-free average the identification
forms.

Usage:
  HD_DERIV=bwk:3 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/lever_fit.py hd.npz
"""
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('hd.npz')
PHI_MAX = 3.0                 # deg; fit where both directions have data

d = np.load(SRC)
rid, phi = d['rid'], d['phi']
res = d['resid']
W = d['Wn'][rid]
f = d['f_col']
ca = np.cos(d['phi_abs'])
off = (d['arm_a'] - d['lp'])[rid]

num = ((d['model'] - d['t_inertia'] - d['t_moment'] - d['t_grav_z']) * 1e-3
       - W * off * ca)
den = W * ca - f
grp = np.array([f"{c}/{a}/{t}"
                for c, a, t in zip(d['case'], d['axis'], d['tip'])])
CASES = sorted(set(d['case']))


def fit(group):
    m = (grp[rid] == group) & (phi < PHI_MAX)
    v = [float(num[m & (rid == i)] @ den[m & (rid == i)]
               / (den[m & (rid == i)] @ den[m & (rid == i)]))
         for i in np.unique(rid[m])]
    return np.array(v) * 1e3


def mocap(group):
    c, a, t = group.split('/')
    m = (d['case'] == c) & (d['axis'] == a) & (d['tip'] == t)
    return np.median(d['lp'][m]) * 1e3


def onset(group, y):
    m = (grp[rid] == group) & (phi >= 0) & (phi < 0.4)
    return np.median(y[m])


print(f"contact lever, least squares below {PHI_MAX:g} deg  [mm]\n")
print(f"  {'class':10}{'n':>4}{'mocap':>9}{'p25':>8}{'median':>9}{'p75':>8}")
for axn in ('Mx', 'My'):
    for tp in ('pos', 'neg'):
        v = np.concatenate([fit(f'{c}/{axn}/{tp}') for c in CASES])
        mk = np.median([mocap(f'{c}/{axn}/{tp}') for c in CASES])
        print(f"  {axn + '/' + tp:10}{len(v):4d}{mk:9.1f}"
              f"{np.percentile(v, 25):8.0f}{np.median(v):9.0f}"
              f"{np.percentile(v, 75):8.0f}")

print(f"\ndoes that asymmetry account for the antisymmetric residual?\n")
print(f"  {'group':16}{'anti/2':>9}{'sens':>8}{'implied':>10}"
      f"{'fitted - mocap':>17}")
print(f"  {'':16}{'[mN.m]':>9}{'[N]':>8}{'[mm]':>10}{'[mm]':>17}")
X, Y = [], []
for c in CASES:
    for axn in ('Mx', 'My'):
        k = f'{c}/{axn}'
        a2 = 0.5 * (onset(k + '/pos', res) - onset(k + '/neg', res))
        m = (d['axis'][rid] == axn) & (phi >= 0) & (phi < 0.4)
        s_ = float(np.median(den[m]))
        implied = a2 * 1e-3 / s_ * 1e3
        got = 0.5 * ((np.median(fit(k + '/pos')) - mocap(k + '/pos'))
                     - (np.median(fit(k + '/neg')) - mocap(k + '/neg')))
        X.append(implied)
        Y.append(got)
        print(f"  {k:16}{a2:9.0f}{s_:8.2f}{implied:10.1f}{got:17.1f}")
X, Y = np.array(X), np.array(Y)
print(f"\n  corr {np.corrcoef(X, Y)[0, 1]:+.3f}   slope "
      f"{np.polyfit(X, Y, 1)[0]:+.2f}"
      f"   (-1 = the residual is exactly that lever error)")
