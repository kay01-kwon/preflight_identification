#!/usr/bin/env python3
"""A model-curve-free noise anchor: three generic estimators, none
knowing the cosh family, cross-checking the deployed RMS(n_hi).

The deployed anchor subtracts the fitted branch before the 5 Hz split
(fit-robust: 1.4% median sensitivity to which family member).  For a
reader uneasy that the method's own curve appears at all, three
cosh-blind alternatives handle the truncation leakage generically:

  poly4   Legendre detrend, degree 4, then the brick-wall split
  dct     DCT-II brick-wall split (even extension kills the
          periodicity jump the DFT split trips over)
  sg      Savitzky-Golay residual (local cubic), then the split

Against the deployed anchor over 140 runs the medians are 0.97 / 1.19
/ 0.97 -- independent confirmation that the cosh subtraction is not
flattering the method -- but each has runs where generic smoothing
absorbs genuine disturbance (one run all three under-read by 38%).
The per-run maximum of the three, inflated 1.10x, is a valid
model-free anchor: swapped into (19b), coverage 140/140, worst used
0.94, mean used 0.39 (against the deployed anchor's 0.50-0.52).  The
deployed anchor stays -- tighter, and confirmed by the neutral ones.

Usage: python analysis/neutral_anchor.py
"""
import pickle
import os
import sys

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failing_runs import split, FC
from rms_check import measure, PHI_BOX, amplitude_best
from tight_rms_bound import enrich
from fit_quality_bound import rho_bar
from kernel_free_bound import model_term

HERE = os.path.dirname(os.path.abspath(__file__))
INFLATE = 1.10


def main():
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    ksup = max(d['kimp'] for d in rows)
    mult = np.sqrt(1 + ksup ** 2)
    hi = lambda v: float(np.sqrt(np.mean(v ** 2)))
    rat = {'poly4': [], 'dct': [], 'sg': [], 'max': []}
    for d in rows:
        om = np.asarray(d['om'], float)
        dt, N = d['dt'], len(d['om'])
        _, rf = amplitude_best(d['tau'], om, d['c2'])
        anchor = hi(split(rf, dt)[1])
        t = np.linspace(-1, 1, N)
        co = np.polynomial.legendre.legfit(t, om, 4)
        p4 = hi(split(om - np.polynomial.legendre.legval(t, co), dt)[1])
        c = dct(om - om.mean(), norm='ortho')
        c[np.arange(N) / (2.0 * N * dt) <= FC] = 0.0
        dc = hi(idct(c, norm='ortho'))
        win = max(int(round(2.0 / (FC * dt))) | 1, 7)
        win = min(win, N - 1 if (N - 1) % 2 else N - 2)
        sg = hi(split(om - savgol_filter(om, win, 3), dt)[1])
        for k, v in (('poly4', p4), ('dct', dc), ('sg', sg),
                     ('max', max(p4, dc, sg))):
            rat[k].append(v / anchor)
        d['neu'] = INFLATE * max(p4, dc, sg)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        de, dpre = model_term(d, rb)
        d['m'] = de + dpre
    for k in rat:
        v = np.array(rat[k])
        print(f"  {k:>6} / deployed: median {np.median(v):.3f}, "
              f"IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}")
    ins = sum(1 for d in rows
              if d['rms_min'] <= d['m'] + np.rad2deg(d['neu']) * mult)
    used = [d['rms_min'] / (d['m'] + np.rad2deg(d['neu']) * mult)
            for d in rows]
    print(f"\n  {INFLATE} x max-of-three as the (19b) anchor: "
          f"coverage {ins}/140, worst used {max(used):.2f}, "
          f"mean used {np.mean(used):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
