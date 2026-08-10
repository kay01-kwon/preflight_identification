#!/usr/bin/env python3
"""Is the Savitzky-Golay derivative attenuating omega_dot?

Plots one run's angular acceleration from several filter widths against
the analytic derivative of the fitted cosh, and reports the ground-effect
slope each width produces.  If the filter is responsible for the residual
slope, that slope must move with the width; if it does not, the filter is
exonerated.

Usage: PYTHONPATH=<stubs> python analysis/omega_dot_probe.py [outdir] [case/axis] [bag]
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from analysis.rate_derivative import omega_dot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from analysis.pnls_constants import PNLS_CONSTANTS
from error_budget import ge_moment
from utils import math_tools

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('docs')
SEL = sys.argv[2] if len(sys.argv) > 2 else 'case_02/Mx'
BAG = sys.argv[3] if len(sys.argv) > 3 else None
ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G, Z, WIDTHS = 9.81, 0.261, (5, 9, 15, 21, 31)
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
        'case_04': 3.220, 'case_05': 3.220}
OFF_MM = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
          ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
          ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
          ('case_05','My'):-10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

d = ROOT / SEL
case, axname = d.parent.name, d.name
axis = 'x' if axname == 'Mx' else 'y'
mass = MASS[case]; W = mass * G
j_p = J_CAD[axis] + mass * (Z ** 2 + LP[axis] ** 2)
c2f, kf = PNLS_CONSTANTS[(case, axname)]
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(d)
    crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2f, ramp_gain=kf)
crit = next((c for c in crits if c.bag_name == BAG), None) if BAG else \
       next(c for c in crits if c.bag_name.startswith('pos'))
bag = {b.name: b for b in bags}[crit.bag_name]
sig = cvp.prepare_signals(bag, axis)
roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
phi_all = roll if axis == 'x' else pitch
n = min(len(phi_all), len(sig['t']))
s = 1.0 if crit.bag_name.startswith('pos') else -1.0
_, i1 = cvp.detect_excitation_window(sig['moment'],
                                     moment_cap=cvp.MOMENT_CAP.get(axis))
j = crit.onset_idx; i1 = min(i1, n - 1); sl = slice(j, i1 + 1)
tau = sig['t'][sl] - sig['t'][j]
phi = s * (phi_all[sl] - phi_all[j]); deg = np.rad2deg(phi)
om_full = s * sig['omega'][:n]
om = om_full[sl]; m = s * sig['moment'][sl]; f = sig['f_col'][sl]
dt = float(np.median(np.diff(tau)))
piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[axis]
a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
raw = ge_moment(bag, sig, axis, n, s > 0, window=sl)
if raw is None:
    sys.exit(f"ge_moment declined {crit.bag_name}: a rotor height fell below "
             f"the near-ground guard, so the image model is out of range. "
             f"Pick another bag.")
ge_mod = s * raw[sl]

mdot = float(np.polyfit(tau, m, 1)[0])
om_fit = kf * mdot * (np.cosh(np.clip(c2f * tau, 0, 30)) - 1.0) + om[0]
omd_fit = kf * mdot * c2f * np.sinh(np.clip(c2f * tau, 0, 30))

print(f"{crit.bag_name}  ({case}/{axname})  Mdot={mdot:.3f} N.m/s  "
      f"window {len(tau)} samples = {tau[-1]:.2f} s  excursion {deg[-1]:.2f} deg")
print(f"cosh constants C2={c2f:.3f}, K={kf:.4f};  J_P={j_p:.4f}, l_p={1e3*lp:.1f} mm\n")
print(f"{'SG width':>9}{'[ms]':>7}{'peak om_dot':>13}{'vs fit':>9}"
      f"{'GE slope':>11}{'GE level':>11}")
res = {}
for w in WIDTHS:
    ww = min(w if w % 2 else w + 1, len(tau) - (1 - len(tau) % 2))
    # differentiate the full trace, then slice, so that widening
    # the window does not simply widen the extrapolated edge
    omd = omega_dot(om_full, dt, ww)[sl]
    ge = (j_p * omd - m - f * lp + W * a * np.cos(phi) - W * Z * np.sin(phi))
    sd, id_ = np.polyfit(deg, 1e3 * ge, 1)
    res[w] = (omd, sd, id_)
    print(f"{ww:9d}{1e3*ww*dt:7.0f}{omd.max():13.3f}"
          f"{100*(omd.max()/omd_fit.max()-1):+8.1f}%{sd:11.2f}{id_:11.1f}")
sm = float(np.polyfit(deg, 1e3 * ge_mod, 1)[0])
ge_fit = (j_p * omd_fit - m - f * lp + W * a * np.cos(phi) - W * Z * np.sin(phi))
sdf, idf = np.polyfit(deg, 1e3 * ge_fit, 1)
print(f"{'cosh fit':>9}{'--':>7}{omd_fit.max():13.3f}{'':9}{sdf:11.2f}{idf:11.1f}")
print(f"{'model':>9}{'':7}{'':13}{'':9}{sm:11.2f}"
      f"{float(np.polyfit(deg, 1e3*ge_mod, 1)[1]):11.1f}")

fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                       gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(WIDTHS)))
ax[0].plot(deg, omd_fit, 'k--', lw=2.0, zorder=6,
           label=r'analytic $\dot\omega$ of the fitted cosh')
for (w, (omd, _, _)), c in zip(res.items(), cols):
    ax[0].plot(deg, omd, color=c, lw=1.3,
               label=f'Savitzky-Golay, {w} samples ({1e3*w*dt:.0f} ms)')
ax[0].set_ylabel(r'$\dot\omega$  [rad/s$^2$]')
ax[0].legend(fontsize=7.5, loc='upper left', frameon=False)
ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_axisbelow(True)
ax[1].plot(deg, 1e3 * ge_mod, color='#D55E00', lw=2.4, zorder=6,
           label='image-superposition model')
for (w, (_, _, _)), c in zip(res.items(), cols):
    omd = res[w][0]
    ge = (j_p * omd - m - f * lp + W * a * np.cos(phi) - W * Z * np.sin(phi))
    ax[1].plot(deg, 1e3 * ge, color=c, lw=1.1)
ax[1].plot(deg, 1e3 * ge_fit, 'k--', lw=2.0, zorder=5,
           label=r'inversion using the analytic $\dot\omega$')
ax[1].set_xlabel(r'tilt excursion from the onset  [deg]')
ax[1].set_ylabel(r'$\Delta M_{\mathrm{GE}}$  [mN$\cdot$m]')
ax[1].legend(fontsize=7.5, loc='lower left', frameon=False)
ax[1].grid(alpha=0.25, lw=0.6); ax[1].set_axisbelow(True)
for s_ in ('top', 'right'):
    ax[0].spines[s_].set_visible(False); ax[1].spines[s_].set_visible(False)
ax[0].set_title(f'{crit.bag_name}  —  {case}/{axname}', fontsize=10)
fig.savefig(OUT / 'fig_omega_dot_probe.pdf', bbox_inches='tight')
print(f"\n-> {OUT / 'fig_omega_dot_probe.pdf'}")
