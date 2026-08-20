#!/usr/bin/env python3
"""The RMS-of-w variant of the model term: how much tighter than sup.

(17) caps |delta_e(tau)| pointwise by w(tau) = (rd2 B2 + rd1 B1)/Wz,
and (20) uses sup w -- valid since RMS <= sup.  But the comparison
solution is concentrated in the last e-foldings, so its RMS over the
window is genuinely smaller: RMS(delta_e) <= RMS(w), computed here
exactly on a grid.  Asymptotically RMS(B2) -> sqrt(1/(108 x)) and
RMS(B1) -> sqrt(1/(16 x)) -- a 1/sqrt(x) dilution the sup ignores.

Campaign result: the model term tightens by 1.43-1.69x; with the
declared noise constant N_n = 3 N_med dominating the cap, per-rate
mean used moves only 0.36-0.41 -> 0.38-0.43 (worst run 0.76),
coverage stays 140/140.  The sup form stays the deployed one (margin
for vehicles beyond this campaign); this variant quantifies the
remaining conservatism of the model term itself.

Usage: python analysis/rms_w_variant.py
"""
import os, pickle, sys
import numpy as np
from scipy.signal import savgol_filter
sys.path.insert(0, '/home/user/preflight_identification/analysis')
from failing_runs import split, FC
from fit_quality_bound import ARMS, W, BETA_M, rho_bar
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from kernel_free_bound import SHAPE_SAFETY, model_term

def B2(u, x):
    return (np.sinh(u + x) / np.sinh(x)
            + np.exp(-2 * x) * np.sinh(-u) / np.sinh(x)
            - np.exp(2 * u)) / 3.0

def B1(u, x):
    return (x / 2.0) * np.sinh(u + x) / np.sinh(x) \
        - (u + x) * np.exp(u) / 2.0

HERE = '/home/user/preflight_identification/analysis'
with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
    rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))

_hi = lambda v: float(np.sqrt(np.mean(v ** 2)))
_sg = []
for d in rows:
    _om = np.asarray(d['om'], float)
    _w = max(int(round(2.0 / (FC * d['dt']))) | 1, 7)
    _w = min(_w, len(_om) - 1 if (len(_om) - 1) % 2 else len(_om) - 2)
    _sg.append(_hi(split(_om - savgol_filter(_om, _w, 3), d['dt'])[1]))
N_n = 3.0 * float(np.rad2deg(np.median(_sg)))
for d in rows:
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
    de_sup, dpre = model_term(d, rb)          # current sup version
    tau, c2, k = d['tau'], d['c2'], d['k']
    x = c2 * tau[-1]; wz = 1.0 / k
    jp0 = 1.0 / (k * c2 ** 2)
    om = k * d['md_full'] * (np.cosh(min(x, 30.0)) - 1.0) \
        + rb * np.sinh(min(x, 30.0)) / (jp0 * c2)
    rd2 = SHAPE_SAFETY * (W * ARMS[d['axis']] * PHI_BOX) * om
    rd1 = abs(BETA_M) * (d['md_full'] * PHI_BOX + d['dm_win'] * om)
    u = np.linspace(-x, 0.0, 4001)
    wtot = (rd2 * B2(u, x) + rd1 * B1(u, x)) / wz
    de_rms = float(np.rad2deg(np.sqrt(np.mean(wtot ** 2))))
    d['cap_sup'] = de_sup + dpre + N_n
    d['cap_rms'] = de_rms + dpre + N_n
    d['ms'], d['mr'] = de_sup, de_rms

rates = sorted({d['rate'] for d in rows})
print(f"{'Mdot':>6}{'model sup':>11}{'model RMSw':>12}{'ratio':>8}"
      f"{'used sup':>10}{'used RMSw':>11}")
for rt in rates:
    v = [d for d in rows if d['rate'] == rt]
    ms = np.mean([d['ms'] for d in v]); mr = np.mean([d['mr'] for d in v])
    us = np.mean([d['rms_min'] / d['cap_sup'] for d in v])
    ur = np.mean([d['rms_min'] / d['cap_rms'] for d in v])
    print(f"{rt:6.2f}{ms:11.3f}{mr:12.3f}{ms/mr:8.2f}{us:10.3f}{ur:11.3f}")
ins_s = sum(1 for d in rows if d['rms_min'] <= d['cap_sup'])
ins_r = sum(1 for d in rows if d['rms_min'] <= d['cap_rms'])
wu = max(d['rms_min'] / d['cap_rms'] for d in rows)
print(f"\ncoverage: sup {ins_s}/140, RMS-of-w {ins_r}/140; "
      f"worst-run used, RMS-of-w: {wu:.3f}")
# asymptotic factors for the doc
for x in (1.9, 2.8, 3.7):
    u = np.linspace(-x, 0, 20001)
    m2s = np.abs(B2(u, x)).max(); m2r = np.sqrt(np.mean(B2(u, x) ** 2))
    m1s = np.abs(B1(u, x)).max(); m1r = np.sqrt(np.mean(B1(u, x) ** 2))
    print(f"x={x}: M2 sup/RMS = {m2s:.4f}/{m2r:.4f} ({m2s/m2r:.2f}x)   "
          f"M1 sup/RMS = {m1s:.4f}/{m1r:.4f} ({m1s/m1r:.2f}x)")
