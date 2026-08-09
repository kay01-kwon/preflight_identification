#!/usr/bin/env python3
"""Does integrating the rate reproduce the attitude?  One run, plotted.

Integrating a gyro is not normally a way to get an angle -- bias
integrates into a ramp -- but over a 0.7 s excitation window the drift
is small and the comparison is informative, because the dynamic
inversion mixes the two signals: J_P omega_dot comes from the rate and
W z sin(phi) from the attitude.  If the integral does not land on the
attitude, they are not the same motion.

Three curves against the measured excursion:

    int(omega)             the rate as reported
    int(omega / 0.890)     with the measured scale factor divided out
    polynomial fit         the rate fitted by the onset-anchored
                           polynomial and integrated analytically

and the pre-onset second is shown as well, where the vehicle is at rest,
so any slope there is bias rather than motion.

Usage:
  PYTHONPATH=<stubs> python analysis/integrate_omega.py OUT.pdf \
      [case] [Mx|My] [bag]
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
from analysis.rate_derivative import omega_dot_poly

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('integrate_omega.pdf')
CASE = sys.argv[2] if len(sys.argv) > 2 else 'case_03'
AXNAME = sys.argv[3] if len(sys.argv) > 3 else 'Mx'
BAG = sys.argv[4] if len(sys.argv) > 4 else 'neg_Mx_045'

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
GAIN, K = 0.890, 6
PRE = 1.0                      # seconds of pre-onset context to show
COL = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'

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
dt = float(np.median(np.diff(sig['t'][:n])))
k0 = max(0, j - int(PRE / dt))

t = sig['t'][k0:i1 + 1] - sig['t'][j]              # tau, zero at the onset
phi_m = np.rad2deg(s * (phi_all[k0:i1 + 1] - phi_all[j]))
om_raw = s * sig['omega'][:n][k0:i1 + 1]

# Remove the bias using the PRE-ONSET MEAN, not the single sample at the
# onset.  The vehicle is at rest before the onset, so the mean there is
# the bias; one sample is the bias plus that sample's noise, and a
# constant error of a few mrad/s integrates into degrees over the
# window.  (This is the difference between a 0.15 deg drift and a 2 deg
# one on this run.)
bias = float(np.mean(om_raw[t < 0])) if (t < 0).sum() > 10 else om_raw[0]
om0 = om_raw - bias


def integ(y):
    """cumulative trapezoid with the constant chosen so phi(0) = 0"""
    c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])
    return np.rad2deg(c - c[np.searchsorted(t, 0.0)])


phi_i = integ(om0)
phi_g = integ(om0 / GAIN)
post = t >= 0
_, om_fit, _ = omega_dot_poly(t[post], (om0 / GAIN)[post], K)
phi_p = np.full_like(t, np.nan)
cp = np.concatenate([[0.0], np.cumsum(0.5 * (om_fit[1:] + om_fit[:-1])
                                      * np.diff(t[post]))])
phi_p[post] = np.rad2deg(cp)

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.80,
                    wspace=0.26)

series = [(phi_m, INK, 'measured attitude', '-'),
          (phi_i, COL[0], r'$\int\omega\,d\tau$', '-'),
          (phi_g, COL[1], r'$\int(\omega/0.890)\,d\tau$', '-'),
          (phi_p, COL[2], 'polynomial fit, integrated', (0, (4, 2)))]

a0 = axes[0]
a0.axvspan(t[0], 0, color=MUTED, alpha=0.18, lw=0)
a0.annotate('at rest', (t[0] * 0.5, 0), textcoords='offset points',
            xytext=(0, 8), color=INK2, fontsize=8, ha='center')
a0.axhline(0, color=MUTED, lw=0.8, zorder=0)
for y, c, lab, ls in series:
    a0.plot(t, y, color=c, lw=2.0, ls=ls, label=lab, solid_capstyle='round')
a0.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a0.set_ylabel(r'excursion $\delta\varphi$  [deg]', color=INK2)
a0.set_title('(a)  attitude, measured and integrated', color=INK,
             loc='left', pad=6)
a0.legend(fontsize=8, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.30), ncol=2, labelcolor=INK2)
a0.grid(alpha=0.22, lw=0.6, color=MUTED)
a0.set_axisbelow(True)

a1 = axes[1]
a1.axvspan(t[0], 0, color=MUTED, alpha=0.18, lw=0)
a1.axhline(0, color=MUTED, lw=0.8, zorder=0)
SHORT = [r'$\int\omega$', r'$\int\omega/0.890$', 'poly fit']
for q, (y, c, lab, ls) in enumerate(series[1:]):
    a1.plot(t, y - phi_m, color=c, lw=2.0, ls=ls, solid_capstyle='round')
    ok = ~np.isnan(y)
    a1.annotate(SHORT[q], (t[ok][-1], (y - phi_m)[ok][-1]),
                textcoords='offset points', xytext=(6, 0), color=c,
                fontsize=8, va='center')
a1.set_xlim(t[0], t[-1] + 0.45 * (t[-1] - max(t[0], 0)))
a1.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a1.set_ylabel('integrated $-$ measured  [deg]', color=INK2)
a1.set_title('(b)  the difference', color=INK, loc='left', pad=6)
a1.grid(alpha=0.22, lw=0.6, color=MUTED)
a1.set_axisbelow(True)
for a_ in axes:
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)
fig.suptitle(f'{CASE}/{AXNAME}/{BAG}', fontsize=9.5, color=INK,
             x=0.075, ha='left', y=1.0)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.with_suffix('.png'), bbox_inches='tight', dpi=200)

pre = t < 0
print(f"{CASE}/{AXNAME}/{BAG}:  window {t[post][-1]:.2f} s, "
      f"excursion {phi_m[-1]:.2f} deg")
print(f"\npre-onset second (vehicle at rest, so this is bias):")
print(f"  mean rate (the bias) {bias*1e3:+7.2f} mrad/s")
print(f"  drift it integrates  {np.rad2deg(bias*PRE):+7.3f} deg over {PRE:.1f} s")
print(f"  residual scatter     {np.std(om_raw[pre])*1e3:7.2f} mrad/s")
print(f"\nat the window end [deg]:")
print(f"  {'measured attitude':30}{phi_m[-1]:8.3f}")
for y, _, lab, _ in series[1:]:
    lab = lab.replace('$', '').replace('\\int', 'int ').replace('\\,d\\tau', '')
    print(f"  {lab:30}{y[-1]:8.3f}   ratio {y[-1]/phi_m[-1]:.3f}")
print(f"\n-> {OUT}")
