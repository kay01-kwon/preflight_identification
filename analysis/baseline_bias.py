#!/usr/bin/env python3
"""Does the fitted pre-onset baseline bias the identified threshold?

The two-segment cost carries a constant C for the pre-onset level, shared
with the post-onset rise.  A natural worry is that C must be driven to
zero.  It must not: what enters the threshold is C - b, the error of C
against the TRUE pre-onset level b, and a large b is harmless as long as
C tracks it.  The synthetic check below shows a true baseline of
0.2 rad/s moving the onset by nothing at all when C = b, while a
mismatch of 0.001 rad/s moves it measurably.

That is also why the estimator fits C rather than fixing it, and why it
uses a median (BASELINE_STAT): the requirement is an unbiased estimate
of the pre-onset level under heavy-tailed gyro noise, not a small
number.

The script reports three things:

  - the gain of C - b onto the identified threshold, in closed form and
    checked against direct minimisation.  Keeping C inside the square
    gives delta t_c = <e_w - C, chi>/||chi||^2, and <1, chi> does not
    vanish because chi is of one sign, so

        dM_crit,C = Wz (C - b) (cosh x - 1) / ( C2 [ sinh 2x /4 - x/2 ] ).

    The Wz there must be the one the setup implies, Mdot/C1; substituting
    a different vehicle's value is what once made this look wrong;
  - the size of C actually fitted across the runs, which bounds |C - b|
    conservatively, since C estimates b from the pre-onset samples;
  - how far C drifts when the candidate onset moves by one sample, the
    quantity dC/dt_c that the stationarity condition of Sec. VI-E drops.

Usage: python analysis/baseline_bias.py [case-glob]     (default case_*)
Runtime is dominated by the onset sweep; a single case takes ~1 min.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import ROOT

# A representative window for the synthetic gain: C_2 and the window
# length sit inside the box of Sec. VI-D, C_1 at the fastest ramp.
C1, C2, TSTAR, TB, MDOT, W_MIN = 0.30, 5.0, 0.40, 1.00, 1.20, 30.08

# The rig's own constants, for converting a fitted baseline into offset:
# the gain rises as the window shortens, so the fastest ramp is the worst.
RIG = {'roll, slowest': (9.477, 5.045, 5.18), 'roll, fastest': (9.477, 5.045, 2.99),
       'pitch, fastest': (9.477, 5.415, 2.99)}


def gain_of(wz, c2, x):
    return wz * (np.cosh(x) - 1) / (c2 * (np.sinh(2 * x) / 4 - x / 2))


def onset_shift(b_true, c_fit, amp=1e-4):
    """Onset displacement when the fitted baseline is c_fit and the truth b."""
    rise = lambda t, tc: C1 * (np.cosh(C2 * max(t - tc, 0.0)) - 1.0)
    e = lambda t: amp * (np.sinh(C2 * max(t - TSTAR, 0.0))
                         - C2 * max(t - TSTAR, 0.0)) ** 2
    y = lambda t: b_true + rise(t, TSTAR) + e(t)
    J = lambda tc: quad(lambda t: (y(t) - c_fit - rise(t, tc)) ** 2,
                        tc, TB, limit=200)[0]
    return minimize_scalar(J, bounds=(TSTAR - 0.08, TSTAR + 0.08),
                           method='bounded', options={'xatol': 1e-13}).x - TSTAR


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else 'case_*'

    print("  what matters is C - b, not C\n")
    print(f"  {'b_true':>9}{'C_fit':>9}{'onset shift [s]':>18}{'vs C = b':>12}")
    base = onset_shift(0.0, 0.0)
    for bt, cf in ((0.0, 0.0), (0.05, 0.05), (0.20, 0.20),
                   (0.0, 0.001), (0.0, 0.005), (0.0, 0.020)):
        s = onset_shift(bt, cf)
        print(f"  {bt:9.3f}{cf:9.3f}{s:18.5e}{s - base:12.3e}")

    gains = [MDOT * (onset_shift(0.0, d) - base) / d
             for d in (0.001, 0.005, 0.020)]
    g = float(np.mean(gains))
    x = C2 * (TB - TSTAR)
    wz = MDOT / C1                     # the Wz this synthetic actually implies
    closed = gain_of(wz, C2, x)
    print(f"\n  gain of C - b onto the threshold")
    print(f"    measured by direct minimisation   {g:.4f} N.m per rad/s"
          f"   (spread {np.ptp(gains):.4f})")
    print(f"    closed form (D.11b)               {closed:.4f}")
    print(f"      Wz (cosh x - 1) / ( C2 [ sinh 2x / 4 - x/2 ] ),  Wz = Mdot/C1"
          f" = {wz:.2f}, x = {x:.2f}")
    print(f"    NOTE the Wz must be the one the setup implies; using another"
          f" vehicle's\n         value here is what made an earlier check"
          f" appear to fail.")

    print(f"\n  the baseline actually fitted, over {pat}\n")
    C, dC = [], []
    for d in sorted(ROOT.glob(f'{pat}/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            crits, _ = cvp.extract_piecewise_batch(bags, axis)
            seen = {b.name: b for b in bags}
        for c in crits:
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(seen[c.bag_name], axis)
            om = sig['omega']
            # The sweep averages over the EXCITATION WINDOW up to the
            # candidate, not over the whole record before it: a longer,
            # quieter stretch gives a baseline four times smaller and is
            # not the quantity the estimator forms.
            i0, _ = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            j = c.onset_idx
            if j - i0 < 8 or j + 2 >= len(om):
                continue
            C.append(abs(cvp._baseline_of(om[i0:j])))
            dC.append(abs(cvp._baseline_of(om[i0:j + 1])
                          - cvp._baseline_of(om[i0:j - 1])) / 2)

    C, dC = np.array(C), np.array(dC)
    print(f"  {'':26}{'median':>10}{'p90':>10}{'max':>10}")
    for lab, v in (('|C| [rad/s]', C), ('|dC| per sample [rad/s]', dC)):
        print(f"  {lab:26}{np.median(v):10.5f}{np.percentile(v, 90):10.5f}"
              f"{v.max():10.5f}")

    print(f"\n  {len(C)} runs.  |C - b| <= |C| conservatively, since C is an"
          f" estimate of b, so\n  at the rig's own gain the threshold bias is"
          f" at most\n")
    print(f"  {'':18}{'gain':>9}{'median |C|':>14}{'worst |C|':>14}")
    for lab, (wz, c2, x) in RIG.items():
        gg = gain_of(wz, c2, x)
        a, b = gg * np.median(C), gg * C.max()
        print(f"  {lab:18}{gg:9.4f}{1e3 * a / W_MIN:11.3f} mm"
              f"{1e3 * b / W_MIN:11.3f} mm")
    print(f"\n  and dC/dt_c, the term Sec. VI-E drops, is under"
          f" {dC.max():.1e} rad/s per sample.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
