#!/usr/bin/env python3
"""What makes the order-7 derivative dip where order 2 does not.

The omega record carries a band at 4-7 Hz, peaking near 5.5 Hz, that
is present ONLY under load: 7.99 deg/s RMS inside the excitation
window against 0.22 deg/s in the pre-onset stretch with the rotors
idling -- a factor of 36, so it is airframe/contact motion excited by
the pivoting, not sensor noise.  (The spectra here use a neutral
degree-6 polynomial detrend, never a Savitzky-Golay one, so the band
is not an artefact of the filter under test.)

The two derivative orders treat that band completely differently.
Normalised to the ideal differentiator, the 41-sample SG derivative
passes 5 Hz at 0.923 for order 7 but only 0.073 for order 2 -- the
-3 dB points are 6.3 Hz and 1.4 Hz.  Differentiation then multiplies
by 2 pi f, so at 5.5 Hz the band arrives in omega_dot with real
weight: the order-7 minus order-2 difference inside the window is a
clean 5.5 Hz oscillation, amplitude 0.30 rad/s^2 and 1.79 rad/s^2
peak to peak, against a signal whose own peak is 1.5-2.3.  The dip
near tau = 0.32 s is one trough of it.

The reading that matters for the ground-effect check: order 7 is more
faithful to the RECORD, order 2 more faithful to the RIGID-BODY MODEL.
The inversion's balance, J_P omega_dot = ..., assumes one rigid body
about a fixed pivot, which a 5.5 Hz structural or rocking mode
violates, so admitting that band injects a term the balance cannot
account for.  This is a second, independent reason to keep the
deployed order 2 -- and it is also why the windowless anchored
polynomial of rate_derivative.omega_dot_poly is the better instrument
than either: fitted over the whole window in the cosh's own basis, it
follows the exponential growth without a local window while averaging
the 5.5 Hz band away.

Usage: PYTHONPATH=<stubs> python analysis/omega_band_probe.py
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np
_R = '/home/user/preflight_identification'
sys.path.insert(0, _R); sys.path.insert(0, _R + '/analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from analysis.rate_derivative import omega_dot
from analysis.ge_dynamics_check import MASS_KG

d = Path(_R + '/DataSet/exp/case_02/Mx'); axis = 'x'
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(d)
    c2, kg = cvp.estimate_rig_constants(bags, axis)
    crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=kg)
name = sorted(c.bag_name for c in crits if c.bag_name.startswith('pos'))[0]
crit = next(c for c in crits if c.bag_name == name)
bag = next(b for b in bags if b.name == name)
sig = cvp.prepare_signals(bag, axis)
roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
n = min(len(roll), len(sig['t']))
_, i1 = cvp.detect_excitation_window(sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
j = crit.onset_idx; i1 = min(i1, n-1)
om = np.asarray(sig['omega'][:n], float)
t = np.asarray(sig['t'][:n], float)
dt = float(np.median(np.diff(t)))

# a long stretch centred on the window, for frequency resolution
a, b = max(0, j-150), min(n, i1+150)
seg = om[a:b]
# NEUTRAL detrend: a degree-6 polynomial in time removes only the slow
# ramp and leaves every band above ~2 Hz untouched (no SG involved)
_x = np.linspace(-1, 1, len(seg))
det = seg - np.polyval(np.polyfit(_x, seg, 6), _x)
f = np.fft.rfftfreq(len(det), d=dt)
A = np.abs(np.fft.rfft(det - det.mean()))*2/len(det)
print(f"\n  segment {len(det)} samples, df = {f[1]:.2f} Hz, "
      f"Nyquist {f[-1]:.1f} Hz")
print(f"  detrended omega spectrum (degree-6 polynomial removed), top bins:")
order = np.argsort(A)[::-1][:8]
for i in sorted(order):
    print(f"    {f[i]:6.2f} Hz   {np.rad2deg(A[i]):7.4f} deg/s")
band = lambda lo, hi: float(np.sqrt(np.sum(A[(f>=lo)&(f<hi)]**2)/2))
for lo, hi in [(1,4),(4,7),(7,12),(12,20),(20,50)]:
    print(f"  RMS {lo:2d}-{hi:2d} Hz : {np.rad2deg(band(lo,hi)):7.4f} deg/s")

# the dip itself: difference between the two orders inside the window
sl = slice(j, i1+1)
d2 = omega_dot(om, dt, 41, 2)[sl]
d7 = omega_dot(om, dt, 41, 7)[sl]
tau = t[sl]-t[j]
diff = d7 - d2
mask = (tau >= 0.05)
z = diff[mask] - diff[mask].mean()
fz = np.fft.rfftfreq(len(z), d=dt); Az = np.abs(np.fft.rfft(z))*2/len(z)
pk = fz[np.argmax(Az)]
print(f"\n  (order7 - order2) inside the window: dominant frequency "
      f"{pk:.1f} Hz, amplitude {Az.max():.3f} rad/s^2")
print(f"  peak-to-peak of the difference: {diff.max()-diff.min():.3f} rad/s^2")


# is the band already there BEFORE the onset (rotors idling, at rest)?
q = om[max(0, j-260):j-10]
if len(q) > 80:
    _xq = np.linspace(-1, 1, len(q))
    dq = q - np.polyval(np.polyfit(_xq, q, 6), _xq)
    fq = np.fft.rfftfreq(len(dq), d=dt)
    Aq = np.abs(np.fft.rfft(dq - dq.mean()))*2/len(dq)
    bq = lambda lo, hi: float(np.sqrt(np.sum(Aq[(fq>=lo)&(fq<hi)]**2)/2))
    print(f"\n  PRE-ONSET stretch, {len(q)} samples, df {fq[1]:.2f} Hz:")
    for lo, hi in [(1,4),(4,7),(7,12),(12,20),(20,50)]:
        print(f"    RMS {lo:2d}-{hi:2d} Hz : {np.rad2deg(bq(lo,hi)):7.4f} deg/s")
    print(f"    peak bin: {fq[np.argmax(Aq[1:])+1]:.2f} Hz")
