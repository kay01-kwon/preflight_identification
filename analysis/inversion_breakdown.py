#!/usr/bin/env python3
"""Term-by-term anatomy of the dynamic inversion, for one run.

The inversion reads

    dM_GE = J_P omega_dot - m - f l_p + W a cos(phi) - W z_CoM sin(phi)

and its residual against the ground-effect model drifts with attitude.
This plots each term separately so the drift can be attributed to one
of them by eye.

Everything is shown as a CHANGE FROM THE ONSET, term(tau) - term(0).
Two reasons.  The absolute terms are dominated by W a cos(phi) and
f l_p, each of order 4 N.m, which cancel to leave a balance of order
0.1 N.m -- plotted raw, nothing else is visible.  And the onset is the
anchor of the whole method: the balance holds exactly there, so the
question is only which term moves afterwards.

The x axis is TIME, not tilt.  Plotted against tilt the curves double
back on themselves, because the tilt is not monotonic in the first
half-degree -- the airframe rocks before it commits -- and a
non-monotonic abscissa hides exactly the transient one is looking for.
It also makes the residual "slope in mN.m/deg" a line fitted through
what is largely a step.

Panel (a) is the five terms, (b) their sum against the ground-effect
model, and (c) the tilt itself, which is what shows the rocking.

Usage:
  PYTHONPATH=<stubs> python analysis/inversion_breakdown.py OUT.pdf \
      [case] [Mx|My] [bag]

With no case given it picks the run whose attitude slope is closest to
its group median, so the picture is representative rather than chosen.
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment
from analysis.pnls_constants import PNLS_CONSTANTS
from analysis.rate_derivative import omega_dot, edge_margin

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('inversion.pdf')
CASE = sys.argv[2] if len(sys.argv) > 2 else 'case_03'
AXNAME = sys.argv[3] if len(sys.argv) > 3 else 'Mx'
BAG = sys.argv[4] if len(sys.argv) > 4 else None

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G, Z = 9.81, 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

# categorical slots 1-5, light mode (adjacent pairlist -- line chart)
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#b8b7b2', '#fcfcfb'

ax = 'x' if AXNAME == 'Mx' else 'y'
mass = MASS[CASE]
W = mass * G
j_p = J_CAD[ax] + mass * (Z ** 2 + LP[ax] ** 2)
c2f, kf = PNLS_CONSTANTS[(CASE, AXNAME)]
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(ROOT / CASE / AXNAME)
    crits, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2f,
                                           ramp_gain=kf)
by = {b.name: b for b in bags}


def decompose(crit):
    bag = by[crit.bag_name]
    s = 1.0 if crit.bag_name.startswith('pos') else -1.0
    roll, pitch = math_tools.quaternion_to_euler_vectorized(
        bag.odom.quaternion)
    phi_all = roll if ax == 'x' else pitch
    sig = cvp.prepare_signals(bag, ax)
    n = min(len(phi_all), len(sig['t']))
    i0w, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
    j = crit.onset_idx
    i1 = min(i1, n - 1)
    if i1 - j < 15:
        return None
    sl = slice(j, i1 + 1)
    tau = sig['t'][sl] - sig['t'][j]
    w = min(9, len(tau) - (1 - len(tau) % 2))
    if w < 5:
        return None
    phi_abs = s * phi_all[sl]
    phi_rel = s * (phi_all[sl] - phi_all[j])
    m = s * sig['moment'][sl]
    f = sig['f_col'][sl]
    # differentiate the full trace, then slice -- see
    # analysis/rate_derivative.py for why slicing first breaks the onset
    dt = float(np.median(np.diff(sig['t'][:n])))
    om_full = s * sig['omega'][:n]
    omd_full = omega_dot(om_full, dt, w)
    om, omd = om_full[sl], omd_full[sl]
    if not edge_margin(n, j, i1, w)['ok']:
        return None
    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, ax)
    lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) \
        else LP[ax]
    a = lp + s * OFF_SIGN[AXNAME] * OFF_MM[(CASE, AXNAME)] * 1e-3
    q_rest = bag.odom.quaternion[:max(20, i0w)].mean(axis=0)
    q_rest = q_rest / np.linalg.norm(q_rest)
    raw = ge_moment(bag, sig, ax, n, s > 0, q_rest=q_rest)
    if raw is None:
        return None
    terms = {
        r'$J_P\dot\omega$': j_p * omd,
        r'$-\,m$': -m,
        r'$-\,f\,l_p$': -f * lp,
        r'$+\,Wa\cos\varphi$': W * a * np.cos(phi_abs),
        r'$-\,Wz\sin\varphi$': -W * Z * np.sin(phi_abs),
    }
    inv = sum(terms.values())
    model = s * raw[sl]
    mdot = abs(float(np.polyfit(tau, m, 1)[0]))
    slope = float(np.polyfit(np.rad2deg(phi_rel), inv - model, 1)[0]) * 1e3
    return dict(bag=crit.bag_name, tau=tau, phi=np.rad2deg(phi_rel),
                terms=terms, inv=inv, model=model, mdot=mdot, slope=slope,
                lp=lp, a=a)


cand = [c for c in (decompose(cr) for cr in crits) if c]
if not cand:
    raise SystemExit(f"no usable run in {CASE}/{AXNAME}")
if BAG:
    pick = next(c for c in cand if c['bag'] == BAG)
else:
    med = np.median([c['slope'] for c in cand])
    pick = min(cand, key=lambda c: abs(c['slope'] - med))
    print(f"group median slope {med:+.1f} mN.m/deg over {len(cand)} runs")
print(f"picked {CASE}/{AXNAME}/{pick['bag']}  "
      f"Mdot={pick['mdot']:.2f} N.m/s  slope={pick['slope']:+.1f} mN.m/deg")
print(f"  l_p={pick['lp']*1e3:.1f} mm, a={pick['a']*1e3:.1f} mm, "
      f"J_P={j_p:.3f} kg.m^2, W={W:.2f} N")

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'pdf.fonttype': 42,
})
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7),
                         gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.8]))
fig.subplots_adjust(left=0.062, right=0.99, bottom=0.16, top=0.80,
                    wspace=0.30)
x = pick['tau']
XL = (-0.02, x.max() * 1.10)

a0 = axes[0]
a0.axhline(0, color=MUTED, lw=0.8, zorder=0)
for k, (lab, v) in enumerate(pick['terms'].items()):
    y = 1e3 * (v - v[0])
    a0.plot(x, y, color=COL[k], lw=1.9, label=lab, solid_capstyle='round',
            zorder=4)
a0.set_xlim(*XL)
a0.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a0.set_ylabel(r'change from the onset  [mN$\cdot$m]', color=INK2)
a0.set_title('(a)  the five terms of the inversion', color=INK, loc='left',
             pad=6)
a0.legend(fontsize=8, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.26), ncol=5, labelcolor=INK2,
          columnspacing=0.9, handlelength=1.3)
a0.grid(alpha=0.22, lw=0.6, color=MUTED)
a0.set_axisbelow(True)

a1 = axes[1]
a1.axhline(0, color=MUTED, lw=0.8, zorder=0)
series = [(1e3 * (pick['inv'] - pick['inv'][0]), COL[0], 'inversion'),
          (1e3 * (pick['model'] - pick['model'][0]), COL[1], 'GE model'),
          (1e3 * ((pick['inv'] - pick['model'])
                  - (pick['inv'][0] - pick['model'][0])), INK, 'residual')]
for y, c, lab in series:
    a1.plot(x, y, color=c, lw=2.4 if lab == 'residual' else 2.0,
            ls=(0, (4, 2)) if lab == 'residual' else '-',
            label=lab, solid_capstyle='round', zorder=4)
    a1.annotate(lab, (x[-1], y[-1]), textcoords='offset points',
                xytext=(7, 0), color=c, fontsize=8.5, fontweight='bold',
                va='center')
a1.set_xlim(XL[0], XL[1] * 1.16)
a1.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a1.set_ylabel(r'change from the onset  [mN$\cdot$m]', color=INK2)
a1.set_title('(b)  their sum, against the model', color=INK, loc='left',
             pad=6)
a1.grid(alpha=0.22, lw=0.6, color=MUTED)
a1.set_axisbelow(True)

a2 = axes[2]
a2.axhline(0, color=MUTED, lw=0.8, zorder=0)
a2.plot(x, pick['phi'], color=COL[4], lw=2.2, solid_capstyle='round',
        zorder=4)
back = float(np.min(pick['phi']))
if back < -0.02:
    a2.axhspan(back, 0, color=COL[4], alpha=0.12, lw=0, zorder=1)
    a2.annotate(f'rocks back {abs(back):.2f}°', (x.max() * 0.5, back),
                textcoords='offset points', xytext=(0, -13), color=COL[4],
                fontsize=8, ha='center')
a2.set_xlim(*XL)
a2.set_xlabel(r'time from the onset, $\tau$  [s]', color=INK2)
a2.set_ylabel(r'tilt $\delta\varphi$  [deg]', color=INK2)
a2.set_title('(c)  the tilt itself', color=INK, loc='left', pad=6)
a2.grid(alpha=0.22, lw=0.6, color=MUTED)
a2.set_axisbelow(True)

for a_ in axes:
    for sp in ('top', 'right'):
        a_.spines[sp].set_visible(False)

fig.suptitle(f"{CASE}/{AXNAME}/{pick['bag']}   "
             rf"$\dot M$ = {pick['mdot']:.2f} N$\cdot$m/s", fontsize=9.5,
             color=INK, x=0.062, ha='left', y=1.06)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.with_suffix('.png'), bbox_inches='tight', dpi=200)
print(f"-> {OUT}")

# Split the excursion at 1 deg: how much of the drift is an onset
# transient, and how much is a genuine attitude trend?
early = pick['phi'] <= 1.0
print(f"\n{'term':24}{'by dphi=1deg':>15}{'over the rest':>15}"
      f"{'total':>10}   [mN.m]")
for lab, v in pick['terms'].items():
    y = 1e3 * (v - v[0])
    e = y[early][-1] if early.any() else 0.0
    print(f"{lab:24}{e:15.1f}{y[-1]-e:15.1f}{y[-1]:10.1f}")
for y, _, lab in series:
    e = y[early][-1] if early.any() else 0.0
    print(f"{lab:24}{e:15.1f}{y[-1]-e:15.1f}{y[-1]:10.1f}")
# --- what frequency is the residual ringing at? -------------------
res = 1e3 * ((pick['inv'] - pick['model'])
             - (pick['inv'][0] - pick['model'][0]))
dt = float(np.median(np.diff(x)))
ac = res - np.polyval(np.polyfit(x, res, 1), x)      # detrend
F = np.fft.rfftfreq(len(ac), dt)
P = np.abs(np.fft.rfft(ac * np.hanning(len(ac)))) ** 2
band = (F > 2) & (F < 40)
f_pk = F[band][np.argmax(P[band])]
print(f"\nresidual ringing: peak at {f_pk:.1f} Hz "
      f"(period {1e3/f_pk:.0f} ms), AC amplitude "
      f"{np.std(ac):.0f} mN.m RMS vs a {abs(res[-1]):.0f} mN.m total offset")
print(f"  an equivalent torsional stiffness would be "
      f"J_P (2 pi f)^2 = {j_p*(2*np.pi*f_pk)**2:.0f} N.m/rad")

print(f"\ntilt is non-monotonic: it rocks back to "
      f"{np.min(pick['phi']):+.2f} deg before committing; "
      f"{100*np.mean(np.diff(pick['phi'])<0):.0f}% of samples move backwards")
