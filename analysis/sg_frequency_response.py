#!/usr/bin/env python3
"""Frequency response of the 41-sample SG derivative, orders 2 vs 7.

Normalised to the ideal differentiator (1.0 = exact d/dt at that
frequency):

    f [Hz]     order 2   order 7
       1         0.847     1.000
       2         0.485     1.000
       5         0.073     0.923
      10         0.018     0.163

-3 dB at 1.4 Hz for order 2 and 6.3 Hz for order 7.  The window length
alone does not set the passband: at fixed length the ORDER does, which
is why raising it recovers sinh(C2 tau) -- and why it simultaneously
admits the 4-7 Hz load-excited band documented in
analysis/omega_band_probe.py.

Usage: python analysis/sg_frequency_response.py
"""
import numpy as np
from scipy.signal import savgol_coeffs

dt, fs = 0.0099, 1/0.0099
f = np.linspace(0.1, 30, 600)
print(f"  SG derivative response |H(f)| / (2 pi f)  -- 1.0 = ideal d/dt")
print(f"  {'f [Hz]':>8}{'order 2':>10}{'order 7':>10}{'ratio 7/2':>11}")
resp = {}
for p in (2, 7):
    c = savgol_coeffs(41, p, deriv=1, delta=dt)
    k = np.arange(41) - 20
    H = np.array([np.abs(np.sum(c*np.exp(-2j*np.pi*ff*dt*k))) for ff in f])
    resp[p] = H / (2*np.pi*f)          # normalise by the ideal differentiator
for ff in (1, 2, 5, 8, 10, 12, 15, 20):
    i = np.argmin(np.abs(f-ff))
    print(f"  {ff:8.0f}{resp[2][i]:10.3f}{resp[7][i]:10.3f}"
          f"{resp[7][i]/max(resp[2][i],1e-9):11.1f}")
for p in (2, 7):
    i3 = np.argmax(resp[p] < 0.707)
    print(f"  order {p}: -3 dB of the ideal differentiator at "
          f"{f[i3]:.1f} Hz")
