#!/usr/bin/env python3
"""Does the dynamic inversion agree with the interference model where
the differentiator can be trusted?

The regime-mixing argument fixes the trim, not taste: within one
half-width of the excitation window's end the Savitzky-Golay support
reaches past the moment cap, where the command stops ramping, so those
samples mix two regimes and are excluded.  At the deployed
differentiator (9 samples, order 2, the default here) that is the
last 4 samples; GE_W=41 selects the 5 Hz-rule window, whose exclusion
is 20 samples.

Over the span that remains, the comparison against the parameter-free
image-superposition (rotor-interference) model is made on the LEVEL:
the per-run mean difference

    d = mean( dM_GE^dyn - dM_GE^model )        [trimmed span]

The GE model is evaluated on attitude RELATIVE to the resting pose:
q_rest, the median pre-ramp quaternion, is passed to ge_moment so the
rotor heights are taken along the pad normal rather than world
vertical (the pad is not level; see analysis/attitude_reference.py).

Campaign result at the deployed differentiator: median d = +26 mN.m
(IQR -43..+85, RMS 106) of a 157 mN.m model level, |d| < 100 on
99/140 runs, 4.75 deg median tilt left.  The wider 41-sample window
does worse (median +77, RMS 128, 1.6 deg left): its parabola clips
more of the sinh growth and its exclusion spends more of the
excursion.  The SLOPE (attitude dependence) is deliberately not
claimed at either setting -- the range needed to resolve the model's
-2.5..-0.1 mN.m/deg is not available in the trusted span.

Panel (a) aggregates the campaign by tilt -- median, IQR and 10-90%
bands over all trusted-span samples in 0.25 deg bins (>= 150 samples
per bin) -- rather than drawing 140 individual traces.

Constants: J_CoM = 0.051 kg m^2 (CAD Table 5), z_CoM = 0.261 m, and
J_P by the parallel axis WITH the CoM offset in the horizontal leg:
J_P = J_CoM + m (z^2 + (l_p + s lambda)^2), the offset component
perpendicular to the pivot line, so J_P is direction-dependent (a
<= 4% effect; Mx pos moves from +11 to +1 mN.m, everything else
within a few mN.m).

Usage: PYTHONPATH=<stubs> [GE_W=9] [GE_K=4]
       python analysis/ge_trusted_span.py [out.png]
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
from error_budget import LP                               # noqa: E402
from analysis.ge_dynamics_check import (                  # noqa: E402
    MASS_KG, G, OFF_SIGN, OFF_MM, J_COM, j_parallel, analyse)

Z = 0.261
W_SG = int(os.environ.get('GE_W', '9'))     # SG window (deployed 9)
SG_P = int(os.environ.get('GE_P', '2'))     # SG order (noise doc uses 3)
K_TRIM = int(os.environ.get('GE_K', str(max((W_SG - 1) // 2, 1))))
# trim = the window's own half-width: the regime-mixing exclusion


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
        j_p = j_parallel(axis, Z, mass)      # dataset-level, for extraction
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
                # resting attitude before the ramp: the GE model is
                # evaluated on attitude RELATIVE to it (heights along
                # the pad normal), not on absolute world attitude
                i0w, _ = cvp.detect_excitation_window(
                    sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
                qr = np.median(bag.odom.quaternion[:max(20, i0w)], axis=0)
                qr = qr / np.linalg.norm(qr)
                cache[crit.bag_name] = (
                    sig, roll if axis == 'x' else pitch, qr)
            sig, phi_all, qr = cache[crit.bag_name]
            nn = min(len(phi_all), len(sig['t']))
            # parallel-axis J_P with the CoM offset in the horizontal
            # leg: the offset component perpendicular to the pivot line
            # (the tipping-sense lambda) adds to l_p, so J_P is
            # direction-dependent
            s_dir = 1.0 if crit.bag_name.startswith('pos') else -1.0
            arm_jp = LP[axis] + s_dir * analyse.off_truth
            j_p_run = J_COM[axis] + mass * (Z ** 2 + arm_jp ** 2)
            r = analyse(bag, crit, axis, sig, phi_all, nn, Z, j_p_run,
                        W_SG, q_rest=qr, sg_poly=SG_P)
            if r and 'trace' in r:
                r.update(case=case, axisname=axname)
                rows.append(r)
        print(f"  {case}/{axname}: done")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_trusted_span.png'
    rows = collect()
    print(f"\n  {len(rows)} runs;  w = {W_SG}, order {SG_P}, "
          f"trim k = {K_TRIM}\n")

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

    PH, GD, GM = [], [], []
    for r in rows:
        ph, gd, gm = r['trace']
        e = len(ph) - K_TRIM
        if e < 10:
            continue
        PH.append(ph[:e]); GD.append(gd[:e]); GM.append(gm[:e])
    PH, GD, GM = map(np.concatenate, (PH, GD, GM))
    bins = np.arange(0.0, PH.max() + 0.25, 0.25)
    cx, md_d, q1_d, q3_d, p10, p90, md_m, lo_m, hi_m = ([] for _ in range(9))
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (PH >= b0) & (PH < b1)
        if m.sum() < 150:
            continue
        cx.append(0.5 * (b0 + b1))
        md_d.append(np.median(GD[m])); q1_d.append(np.percentile(GD[m], 25))
        q3_d.append(np.percentile(GD[m], 75))
        p10.append(np.percentile(GD[m], 10)); p90.append(np.percentile(GD[m], 90))
        md_m.append(np.median(GM[m])); lo_m.append(np.percentile(GM[m], 10))
        hi_m.append(np.percentile(GM[m], 90))
    a1.fill_between(cx, p10, p90, color='#c0392b', alpha=0.12, lw=0)
    a1.fill_between(cx, q1_d, q3_d, color='#c0392b', alpha=0.25, lw=0)
    a1.plot(cx, md_d, '-', color='#c0392b', lw=2.0)
    a1.fill_between(cx, lo_m, hi_m, color='#e08214', alpha=0.35, lw=0)
    a1.plot(cx, md_m, '-', color='#e08214', lw=2.0)
    a1.plot([], [], '-', color='#c0392b', lw=1.6,
            label=f'dynamic inversion: median, IQR, 10-90% '
                  f'({len(d_trim)} runs)')
    a1.plot([], [], '-', color='#e08214', lw=1.6,
            label='rotor-interference model: median, 10-90%')
    a1.axhline(0, color='0.5', lw=0.8)
    a1.set_xlabel(r'excursion $\delta\varphi$ [deg]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) campaign aggregate by tilt (last half-width '
                 'excluded;\nbins with $\\geq$150 samples): the band is '
                 'run-to-run and noise spread', fontsize=11)
    a1.legend(fontsize=8.5, loc='upper left')
    a1.set_ylim(-260, 640)
    a1.set_xlim(left=0.0)
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
    a2.set_title('(b) level agreement per run, before and after the trim\n'
                 f'trimmed median {np.median(d_trim):+.0f} of a '
                 f'{np.median(lvl):.0f} mN$\\cdot$m model level '
                 f'($|d|{{<}}100$: {int(np.sum(np.abs(d_trim)<100))}'
                 f'/{len(d_trim)})', fontsize=11)
    a2.legend(fontsize=8.5, loc='upper left')
    a2.grid(alpha=0.22, lw=0.4, axis='y')

    fig.suptitle('Dynamic GE moment against the rotor-interference model, '
                 f'trusted span ($w={W_SG}$, order {SG_P}, trim {K_TRIM}; '
                 r'$J_{CoM}=0.051$ kg m$^2$, $z_{CoM}=0.261$ m)',
                 fontsize=12, y=0.965)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
