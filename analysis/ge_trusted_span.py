#!/usr/bin/env python3
"""Does the dynamic inversion agree with the interference model where
the differentiator can be trusted?

The regime-mixing argument fixes the trim, not taste: within one
half-width of the excitation window's end the Savitzky-Golay support
reaches past the moment cap, where the command stops ramping, so those
samples mix two regimes and are excluded.  At the 5 Hz-rule window
(41 samples, order 2) that is the last 20 samples (~0.20 s).

Over the span that remains, the comparison against the parameter-free
image-superposition (rotor-interference) model is made on the LEVEL:
the per-run mean difference

    d = mean( dM_GE^dyn - dM_GE^model )        [trimmed span]

The SLOPE (attitude dependence) is deliberately not claimed: trimming
spends the excursion (median tilt left ~1.6 deg), so the attitude
range needed to resolve the model's -2.5..-0.1 mN.m/deg is no longer
in the data -- the level is what the trusted span can test.

Constants: J_CoM = 0.051 kg m^2 (CAD Table 5), J_P by the parallel
axis, z_CoM = 0.261 m.

Usage: PYTHONPATH=<stubs> python analysis/ge_trusted_span.py [out.png]
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import critical_value_getter_piecewise as cvp             # noqa: E402
from utils.extractor import load_excitation_dataset       # noqa: E402
from utils import math_tools                              # noqa: E402
from analysis.ge_dynamics_check import (                  # noqa: E402
    MASS_KG, G, OFF_SIGN, OFF_MM, j_parallel, analyse)

Z = 0.261
W_SG = 41              # the 5 Hz-rule window, order 2 (deployed order)
K_TRIM = 20            # half-width: the regime-mixing exclusion


def collect():
    root = Path(_HERE).resolve().parents[0] / 'DataSet' / 'exp'
    analyse.keep_traces = True
    rows = []
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        case, axname = d.parent.name, d.name
        mass = MASS_KG[case]
        analyse.W = mass * G
        analyse.off_truth = OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
        j_p = j_parallel(axis, Z, mass)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bags = load_excitation_dataset(d)
                c2 = float(np.sqrt(analyse.W * Z / j_p))
                crits, _ = cvp.extract_piecewise_batch(
                    bags, axis, cosh_c2=c2, ramp_gain=1.0 / (analyse.W * Z))
        except Exception as exc:                           # noqa: BLE001
            print(f"  {case}/{axname}: skipped ({type(exc).__name__})")
            continue
        by = {b.name: b for b in bags}
        cache = {}
        for crit in crits:
            bag = by.get(crit.bag_name)
            if bag is None:
                continue
            if crit.bag_name not in cache:
                with contextlib.redirect_stdout(io.StringIO()):
                    sig = cvp.prepare_signals(bag, axis)
                roll, pitch = math_tools.quaternion_to_euler_vectorized(
                    bag.odom.quaternion)
                cache[crit.bag_name] = (sig, roll if axis == 'x' else pitch)
            sig, phi_all = cache[crit.bag_name]
            nn = min(len(phi_all), len(sig['t']))
            r = analyse(bag, crit, axis, sig, phi_all, nn, Z, j_p, W_SG)
            if r and 'trace' in r:
                r.update(case=case, axisname=axname)
                rows.append(r)
        print(f"  {case}/{axname}: done")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_trusted_span.png'
    rows = collect()
    print(f"\n  {len(rows)} runs;  w = {W_SG}, order 2, trim k = {K_TRIM}\n")

    d_trim, d_full, phi_left, rates = [], [], [], []
    for r in rows:
        ph, gd, gm = r['trace']
        e = len(ph) - K_TRIM
        if e < 10:
            continue
        d_trim.append(float(np.mean(gd[:e] - gm[:e])))
        d_full.append(float(np.mean(gd - gm)))
        phi_left.append(ph[e - 1])
        rates.append(r['bag'])
    d_trim, d_full = np.array(d_trim), np.array(d_full)
    lvl = np.array([float(np.mean(r['trace'][2])) for r in rows])

    def stats(v, lab):
        print(f"  {lab:<26} median {np.median(v):+7.1f}  IQR "
              f"[{np.percentile(v,25):+7.1f},{np.percentile(v,75):+7.1f}]"
              f"  RMS {np.sqrt(np.mean(v**2)):6.1f}  "
              f"|d|<100: {int(np.sum(np.abs(v)<100))}/{len(v)}")

    stats(d_full, 'mean diff, whole window')
    stats(d_trim, 'mean diff, trimmed span')
    print(f"  model level for scale: median {np.median(lvl):.1f} mN.m")
    print(f"  median tilt left in the trimmed span: "
          f"{np.median(phi_left):.2f} deg")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.115,
                        wspace=0.24)

    for r in rows:
        ph, gd, gm = r['trace']
        e = len(ph) - K_TRIM
        if e < 10:
            continue
        a1.plot(ph[:e], gd[:e], '-', color='#c0392b', lw=0.5, alpha=0.30)
        a1.plot(ph[:e], gm[:e], '-', color='#e08214', lw=0.5, alpha=0.30)
    a1.plot([], [], '-', color='#c0392b', lw=1.6,
            label=f'dynamic inversion, trimmed ({len(d_trim)} runs)')
    a1.plot([], [], '-', color='#e08214', lw=1.6,
            label='rotor-interference model')
    a1.axhline(0, color='0.5', lw=0.8)
    a1.set_xlabel(r'excursion $\delta\varphi$ [deg]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) the trusted span only (last half-width excluded):\n'
                 'no zero crossings; the point scatter is $J_P\\dot\\omega$ '
                 'noise -- the per-run mean is the test', fontsize=11)
    a1.legend(fontsize=8.5, loc='upper right')
    a1.grid(alpha=0.22, lw=0.4)

    bins = np.linspace(-400, 400, 33)
    a2.hist(np.clip(d_full, bins[0], bins[-1]), bins=bins, color='0.75',
            alpha=0.9, label=f'whole window  (RMS {np.sqrt(np.mean(d_full**2)):.0f})')
    a2.hist(np.clip(d_trim, bins[0], bins[-1]), bins=bins, color='#1a5276',
            alpha=0.75, label=f'trimmed span  (RMS {np.sqrt(np.mean(d_trim**2)):.0f})')
    a2.axvline(0, color='#e08214', lw=2.0, label='model')
    a2.axvline(float(np.median(d_trim)), color='#1a5276', lw=1.2, ls='--',
               label=f'trimmed median {np.median(d_trim):+.0f}')
    a2.set_xlabel(r'per-run mean of $\Delta M_{GE}^{dyn} - \Delta M_{GE}'
                  r'^{model}$ [mN$\cdot$m]', fontsize=10)
    a2.set_ylabel('runs', fontsize=10)
    a2.set_title('(b) level agreement per run: trimmed median $+77$ matches the\n'
                 'static check\'s $+70$ mN$\\cdot$m offset '
                 f'(model level itself: median {np.median(lvl):.0f} '
                 'mN$\\cdot$m)', fontsize=11)
    a2.legend(fontsize=8.5, loc='upper left')
    a2.grid(alpha=0.22, lw=0.4, axis='y')

    fig.suptitle('Dynamic GE moment against the rotor-interference model, '
                 f'trusted span ($w={W_SG}$, order 2, trim {K_TRIM}; '
                 r'$J_{CoM}=0.051$ kg m$^2$, $z_{CoM}=0.261$ m)',
                 fontsize=12, y=0.965)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
