#!/usr/bin/env python3
"""What the two derivative routes do to one run, side by side.

The inversion needs omega_dot, and there are two defensible ways to get
it from a 100 Hz rate gyro.  Fit a low-order polynomial in tau over the
window and differentiate it analytically, or take the centred
difference and low-pass the result with a zero-phase Butterworth.  The
choice matters because J_P omega_dot is the only dynamic term in the
balance, and J_P is 0.33 kg.m^2, so 0.5 rad/s^2 of difference is 165
mN.m -- the whole ground-effect moment.

(a) the rate, as reported and as filtered;
(b) omega_dot from the raw centred difference, from the filter, and
    from the polynomial;
(c) the same three as J_P omega_dot, against the modelled ground-effect
    moment so the scale is legible.

The polynomial is anchored at the onset (omega = omega_dot = 0 there by
construction), the filter is not -- that is the substantive difference
between them, and it is visible at tau = 0 in (b).

Usage:
  PYTHONPATH=<stubs> python analysis/omega_dot_compare.py OUT.pdf \
      [case] [Mx|My] [bag] [fc]
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from analysis.rate_derivative import (omega_dot_butter, butter_lowpass,
                                      omega_dot_poly)
from analysis.error_budget import ge_moment, LP

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('omega_dot_compare.pdf')
CASE = sys.argv[2] if len(sys.argv) > 2 else 'case_03'
AXNAME = sys.argv[3] if len(sys.argv) > 3 else 'Mx'
BAG = sys.argv[4] if len(sys.argv) > 4 else 'neg_Mx_045'
FC = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
K, GAIN = 6, 0.890

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
RAW, BW, POLY, MOD = '#b8b7b2', '#2a78d6', '#1baf7a', '#eb6834'
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
J_CAD = dict(x=0.0537, y=0.0537)
Z = 0.261

ax = 'x' if AXNAME == 'Mx' else 'y'
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(ROOT / CASE / AXNAME)
    crits, _ = cvp.extract_piecewise_batch(bags, ax)
crit = next(c for c in crits if c.bag_name == BAG)
bag = {b.name: b for b in bags}[BAG]
s = 1.0 if BAG.startswith('pos') else -1.0

sig = cvp.prepare_signals(bag, ax)
roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
phi_all = roll if ax == 'x' else pitch
n = min(len(phi_all), len(sig['t']))
_, i1 = cvp.detect_excitation_window(sig['moment'],
                                     moment_cap=cvp.MOMENT_CAP.get(ax))
j = crit.onset_idx
i1 = min(i1, n - 1)
sl = slice(j, i1 + 1)
dt = float(np.median(np.diff(sig['t'][:n])))
tau = sig['t'][sl] - sig['t'][j]

om_full = s * sig['omega'][:n] / GAIN
pre = slice(max(0, j - 100), j)
bias = float(np.mean(om_full[pre])) if j >= 20 else om_full[j]
om0 = om_full - bias

om_raw = om0[sl]
om_bw = butter_lowpass(om0, dt, FC)[sl]
omd_raw = np.gradient(om0, dt)[sl]
omd_bw = omega_dot_butter(om0, dt, FC)[sl]
_, om_poly, omd_poly = omega_dot_poly(tau, om_raw, K)

mass = MASS[CASE]
j_p = J_CAD[ax] + mass * (Z ** 2 + LP[ax] ** 2)
q_rest = bag.odom.quaternion[:max(20, j)].mean(axis=0)
q_rest = q_rest / np.linalg.norm(q_rest)
raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest, window=sl)
model = 1e3 * s * raw[sl]

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)
fig.subplots_adjust(left=0.135, right=0.98, bottom=0.075, top=0.94,
                    hspace=0.18)


def dress(a_, ylab, title):
    a_.axhline(0, color=MUTED, lw=0.9, zorder=0)
    a_.set_ylabel(ylab, color=INK2)
    a_.grid(alpha=0.22, lw=0.6, color=MUTED)
    a_.set_axisbelow(True)
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
    a_.set_title(title, fontsize=11, color=INK, loc='left', pad=6)


a = axes[0]
a.plot(tau, om_raw, color=RAW, lw=1.6, label='as reported')
a.plot(tau, om_bw, color=BW, lw=2.2, label=f'Butterworth {FC:g} Hz')
a.plot(tau, om_poly, color=POLY, lw=2.0, ls=(0, (4, 2)),
       label=f'polynomial K={K}')
dress(a, r'$\omega$  [rad/s]', '(a)  the rate')
a.legend(fontsize=10, frameon=False, loc='upper left', labelcolor=INK2)

a = axes[1]
a.plot(tau, omd_raw, color=RAW, lw=1.2, label='centred difference, unfiltered')
a.plot(tau, omd_bw, color=BW, lw=2.4, label=f'  + Butterworth {FC:g} Hz')
a.plot(tau, omd_poly, color=POLY, lw=2.2, ls=(0, (4, 2)),
       label=f'polynomial K={K}, differentiated')
dress(a, r'$\dot\omega$  [rad/s$^2$]', '(b)  the derivative')
a.legend(fontsize=10, frameon=False, loc='upper left', labelcolor=INK2)

a = axes[2]
a.plot(tau, 1e3 * j_p * omd_bw, color=BW, lw=2.4,
       label=f'$J_P\\dot\\omega$, Butterworth {FC:g} Hz')
a.plot(tau, 1e3 * j_p * omd_poly, color=POLY, lw=2.2, ls=(0, (4, 2)),
       label=f'$J_P\\dot\\omega$, polynomial K={K}')
a.plot(tau, model, color=MOD, lw=2.4,
       label=r'$\Delta M_{\mathrm{GE}}$, model')
dress(a, r'moment  [mNm]', '(c)  as a moment, against the model')
a.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a.legend(fontsize=10, frameon=False, loc='upper left', labelcolor=INK2)

fig.suptitle(f'{CASE}/{AXNAME}/{BAG}    '
             f'$J_P$ = {j_p:.3f} kg$\\cdot$m$^2$', fontsize=11.5,
             color=INK, x=0.135, ha='left', y=0.975)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.with_suffix('.png'), bbox_inches='tight', dpi=190)
print(f"-> {OUT}")
print(f"{CASE}/{AXNAME}/{BAG}: window {tau[-1]:.2f} s, J_P {j_p:.3f} kg.m^2\n")
print(f"  {'':22}{'at tau = 0':>12}{'peak':>10}{'as a moment':>14}")
for lab, v in (('centred difference', omd_raw), (f'Butterworth {FC:g} Hz',
                                                 omd_bw),
               (f'polynomial K={K}', omd_poly)):
    print(f"  {lab:22}{v[0]:12.3f}{np.max(np.abs(v)):10.3f}"
          f"{1e3 * j_p * v[0]:+10.1f} mNm")
print(f"\n  the model puts Delta M_GE at {model[0]:.0f} mNm here, so the"
      f"\n  onset value of J_P omega_dot is "
      f"{1e3 * j_p * omd_bw[0] / model[0]:+.2f} of it for the filter and"
      f" {1e3 * j_p * omd_poly[0] / model[0]:+.2f} for the polynomial.")
