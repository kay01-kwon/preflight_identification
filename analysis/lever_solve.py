#!/usr/bin/env python3
"""The lever the rigid-body balance demands, axis by axis.

analysis/rate_band_check.py shows the inertial moment the gyro implies
matching the moment actually present below 3 Hz -- but only on Mx.  On
My the ratio is 1.8-2.0 even there, in every group.  This inverts that
comparison: instead of reporting the ratio, solve

    J_P omega_dot  =  m  +  f l_p

for the l_p that would close it in the 1-3 Hz band, and set that beside
the value assumed.

    Mx: demanded 140.9 mm (IQR 104.7-167.6), assumed 140.0, excess +0.9
    My: demanded 262.0 mm (IQR 199.0-307.9), assumed 110.0, excess +152.0

Mx closes with the lever it was given.  My demands 2.4 times its own,
consistently: all five My/neg groups ask for 263-321 mm and all five
My/pos for 140-221, while Mx scatters about 140 with no systematic
excess.

That single discrepancy accounts for three separate anomalies -- the
band ratio, the antisymmetric residual appearing on My and not Mx
(analysis/ge_dynamic_symmetry.py, +7.6 mm against -1.1), and its
sensitivity of 37.2 mN.m per mm of pivot asymmetry where a pivot shift
should give 10.2.

Two honest caveats.  The static check fits its lever from odometry,
gets 112.7 mm on My, and predicts the threshold to 59 mN.m; if the true
lever were 262 that check should fail, so the two are probably not the
same quantity -- odometry fits the kinematic centre of rotation, this
solves for the moment arm of the collective, and a finite rolling
contact separates them.  And the solve adds amplitude spectra without
phase, so the absolute 262 carries an error the Mx/My contrast does
not, both axes being computed the same way.

Usage:
  PYTHONPATH=<stubs> python analysis/lever_solve.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from analysis.error_budget import LP

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
J_CAD, Z = dict(x=0.0537, y=0.0537), 0.261
BAND = (1.0, 3.0)


def spectrum(x, dt):
    x = np.asarray(x, float)
    k = np.arange(len(x))
    x = x - np.polyval(np.polyfit(k, x, 3), k)
    return np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 4


out = {}
for case in sorted(p.name for p in ROOT.glob('case_*') if p.is_dir()):
    for axname in ('Mx', 'My'):
        d = ROOT / case / axname
        if not d.exists():
            continue
        ax = 'x' if axname == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            crits, _ = cvp.extract_piecewise_batch(bags, ax)
        by = {b.name: b for b in bags}
        j_p = J_CAD[ax] + MASS[case] * (Z ** 2 + LP[ax] ** 2)
        for crit in crits:
            bag = by[crit.bag_name]
            sig = cvp.prepare_signals(bag, ax)
            n = len(sig['t'])
            dt = float(np.median(np.diff(sig['t'][:n])))
            j = crit.onset_idx
            k0, k1 = max(0, j - 150), min(n, j + 250)
            f = np.fft.rfftfreq(k1 - k0, dt)
            xw = spectrum(sig['omega'][k0:k1], dt)
            xm = spectrum(sig['moment'][k0:k1], dt)
            xf = spectrum(sig['f_col'][k0:k1], dt)
            b = (f >= BAND[0]) & (f < BAND[1])
            fc = f[b][np.argmax(xw[b])]
            need = j_p * (2 * np.pi * fc) * xw[b].max()
            # need = xm + xf * lp   ->   lp that closes the balance
            lp_req = (need - xm[b].max()) / max(xf[b].max(), 1e-9)
            key = f'{case}/{axname}/{"pos" if crit.bag_name.startswith("pos") else "neg"}'
            out.setdefault(key, []).append(lp_req)

print("lever the 1-3 Hz balance demands, against the one assumed  [mm]\n")
print(f"  {'group':22}{'n':>4}{'lever demanded':>17}{'assumed LP':>13}"
      f"{'excess':>9}")
for k in sorted(out):
    v = np.array(out[k]) * 1e3
    lp = LP['x' if '/Mx/' in k else 'y'] * 1e3
    print(f"  {k:22}{len(v):4d}{np.median(v):17.1f}{lp:13.1f}"
          f"{np.median(v) - lp:+9.1f}")
for axn in ('Mx', 'My'):
    v = np.concatenate([out[k] for k in out if f'/{axn}/' in k]) * 1e3
    lp = LP['x' if axn == 'Mx' else 'y'] * 1e3
    print(f"\n  {axn}: demanded {np.median(v):.1f} mm (IQR "
          f"{np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}), "
          f"assumed {lp:.1f}, excess {np.median(v) - lp:+.1f}")
