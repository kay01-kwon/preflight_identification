#!/usr/bin/env python3
"""Cutoff sensitivity of the hardware RMSE-bound validation.

Re-runs the deployed free fit once per run, then sweeps the
brick-wall split frequency f_c and re-scores the final single-curve
cap (box-worst envelope + max n_hi * sqrt(1 + kappa_b^2), kappa_b
held at the deployed 1.31): per cutoff, the median and maximum
n_hi, the noise term, the coverage count and the worst used
fraction. Result: over the 2-6 Hz plateau the validation is
unchanged (140/140 throughout) and lowering f_c only enlarges the
noise term, i.e. makes the cap more conservative.

Usage
-----
  PYTHONPATH=<stubs> python analysis/fc_sweep_final.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import critical_value_getter_piecewise as cvp          # noqa: E402
from utils.extractor import load_excitation_dataset    # noqa: E402
from bound_final_figs import hw_capline                # noqa: E402

KB = 1.31
CUTS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

runs = []
for d in sorted((Path(__file__).resolve().parents[1]
                 / 'DataSet' / 'exp').glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    for bag in bags:
        md = cvp.commanded_ramp_rate(bag.name)
        if md is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            crit, pw = cvp.extract_piecewise(bag, axis, model='cosh',
                                             cosh_c2=None,
                                             ramp_gain=None)
            sig = cvp.prepare_signals(bag, axis)
        i0, i1 = cvp.detect_excitation_window(
            sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
        om = sig['omega'][i0:i1 + 1]
        pred = pw['omega_pred']
        n = min(len(om), len(pred))
        r = om[:n] - pred[:n]
        dt = float(np.median(np.diff(sig['t'][i0:i0 + n])))
        runs.append((md, r, dt))
    print('done', d, flush=True)

caps_env = {}
print(f"\n{'fc':>4} {'med n_hi':>9} {'max n_hi':>9} {'N term':>7} "
      f"{'inside':>8} {'worst use':>10}")
for fc in CUTS:
    nhis, res_l, rates = [], [], []
    for md, r, dt in runs:
        rr = r - r.mean()
        F = np.fft.rfft(rr)
        f = np.fft.rfftfreq(len(rr), d=dt)
        F[f <= fc] = 0.0
        hi = np.fft.irfft(F, n=len(rr))
        nhis.append(np.degrees(np.sqrt(np.mean(hi ** 2))))
        res_l.append(np.degrees(np.sqrt(np.mean(r ** 2))))
        rates.append(md)
    nhis, res_l, rates = map(np.array, (nhis, res_l, rates))
    N = nhis.max() * np.sqrt(1 + KB ** 2)
    cap = np.array([caps_env.setdefault(v, hw_capline(v))
                    for v in rates]) + N
    use = res_l / cap
    print(f"{fc:4.0f} {np.median(nhis):9.2f} {nhis.max():9.2f} "
          f"{N:7.2f} {int((use <= 1).sum()):>4}/{len(use)} "
          f"{use.max():10.2f}")
