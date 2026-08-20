#!/usr/bin/env python3
"""Why the SG derivative needs a HIGH ORDER, not a short window.

omega_dot ~ sinh(C2 tau) has essentially no content above 1 Hz, so a
5 Hz smoother should be harmless -- yet the deployed differentiator
(savgol poly=2) loses the growth as its window widens.  The cause is
not the passband: it is that a local parabola cannot represent an
exponential over 2.5 e-foldings.

This script measures the distortion on a NOISE-FREE cosh with the
constants of case_02/Mx/pos_Mx_01, so any error is signal distortion
with no noise involved.  Endpoint error of the derivative:

    (w=9,  p=3)  -0.1%      (w=41, p=3)  -6.4%
    (w=21, p=3)  -1.2%      (w=41, p=5)  -0.2%
    (w=41, p=7)  -0.0%      (w=21, p=5)  -0.0%

so raising the order over the SAME window removes it completely.

Usage: python analysis/sg_derivative_order.py
"""
import numpy as np
from scipy.signal import savgol_filter

C2, dt, N = 6.125, 0.0099, 79          # case_02/Mx/pos_Mx_01
C1 = 0.06                              # amplitude (arbitrary, scales out)
tau = np.arange(N)*dt
om = C1*(np.cosh(C2*tau) - 1.0)
truth = C1*C2*np.sinh(C2*tau)          # analytic omega_dot

print(f"noise-free cosh, C2={C2}, window {N} samples / {tau[-1]:.2f} s")
print(f"{'w':>4}{'ms':>6}{'ord':>5}{'f_c~Hz':>8}"
      f"{'end err %':>11}{'mid err %':>11}{'max|err| %':>12}")
for w, p in [(9,3),(21,3),(41,3),(41,5),(41,7),(21,5),(15,3),(9,2),(61,3)]:
    if w > N: continue
    d = savgol_filter(om, w, p, deriv=1, delta=dt)
    e = 100*(d - truth)/np.maximum(np.abs(truth), 1e-9)
    fc = 2.0/(w*dt)
    mid = N//2
    print(f"{w:4d}{1e3*w*dt:6.0f}{p:5d}{fc:8.1f}"
          f"{e[-1]:11.1f}{e[mid]:11.1f}{np.abs(e[5:]).max():12.1f}")

# where does the error live?
d41 = savgol_filter(om, 41, 3, deriv=1, delta=dt)
e41 = 100*(d41-truth)/np.maximum(np.abs(truth),1e-9)
print("\nerror profile, w=41 p=3  (last 12 samples):")
print("  tau[s] :", " ".join(f"{t:6.2f}" for t in tau[-12:]))
print("  err[%] :", " ".join(f"{v:6.1f}" for v in e41[-12:]))
print(f"\n  interior (tau<0.5 s) max |err| = {np.abs(e41[(tau>0.1)&(tau<0.5)]).max():.1f}%")
print(f"  edge     (last 20 samples)     = {np.abs(e41[-20:]).max():.1f}%")
