#!/usr/bin/env python3
"""The dynamic ground-effect check over the whole campaign, in one figure.

Every run of every dataset is inverted for dM_GE with the DEPLOYED
Savitzky-Golay order (2) at two windows -- the deployed 9 samples and
the 41 samples the noise model's 5 Hz rule would ask for -- and the
140 traces are drawn together with the two diagnostics that decide how
to read them.

(a) All 140 inversions against tilt, with the parameter-free image
    superposition model.  The model is a flat band near 150 mN.m; the
    inversions fan downward, crossing zero, which no ground-effect
    moment can do at this geometry.

(b) Fitted slope against the run's excursion range.  A physical
    attitude dependence would repeat regardless of how far the run
    tipped.  Instead the slope tracks the range at r = +0.89 (w = 9)
    and +0.95 (w = 41): short runs read steep, long runs shallow --
    the signature of a fitting artefact.

(c) Slope against the number of trailing samples excluded from the
    fit, median and IQR over the campaign.  Trimming the tail does
    pull the slope toward the model, but phi grows exponentially, so
    the trimmed samples carry nearly all the attitude range: the
    second axis reports what excursion is left.  Note the batch
    numbers quoted elsewhere use the WHOLE window (k = 0); nothing is
    trimmed by default.

Usage: PYTHONPATH=<stubs> python analysis/ge_campaign_figure.py [out.png]
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
WINDOWS = [(9, '#c0392b', 'SG 9 samples (90 ms), order 2 -- deployed'),
           (41, '#1a5276', 'SG 41 samples (410 ms), order 2 -- the 5 Hz rule')]
DROPS = [0, 3, 5, 8, 10, 15, 20, 30]


def collect():
    root = Path(_HERE).resolve().parents[0] / 'DataSet' / 'exp'
    analyse.keep_traces = True
    rows = {w: [] for w, _, _ in WINDOWS}
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
                    bags, axis, cosh_c2=c2,
                    ramp_gain=1.0 / (analyse.W * Z))
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
            for w, _, _ in WINDOWS:
                r = analyse(bag, crit, axis, sig, phi_all, nn, Z, j_p, w)
                if r and 'trace' in r:
                    r.update(case=case, axis=axname)
                    rows[w].append(r)
        print(f"  {case}/{axname}: {len(cache)} bags")
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'ge_campaign.png'
    rows = collect()
    n = len(rows[WINDOWS[0][0]])
    print(f"\n  {n} runs per window\n")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.6, 4.9))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.115,
                        wspace=0.29)

    # ---- (a) every inversion, plus the model ----
    w0, c0, _ = WINDOWS[0]
    for r in rows[w0]:
        ph, gd, gm = r['trace']
        a1.plot(ph, gd, '-', color=c0, lw=0.5, alpha=0.30)
        a1.plot(ph, gm, '-', color='#e08214', lw=0.5, alpha=0.30)
    a1.plot([], [], '-', color=c0, lw=1.6, label=f'dynamic inversion ({n} runs)')
    a1.plot([], [], '-', color='#e08214', lw=1.6,
            label='image-superposition model')
    a1.axhline(0, color='0.5', lw=0.8)
    a1.set_xlabel(r'excursion $\delta\varphi$ [deg]', fontsize=10)
    a1.set_ylabel(r'$\Delta M_{GE}$ [mN$\cdot$m]', fontsize=10)
    a1.set_title('(a) every run of the campaign, deployed differentiator\n'
                 'the model is flat near 150; the inversions fan through zero',
                 fontsize=11)
    a1.legend(fontsize=8.5, loc='lower left')
    a1.grid(alpha=0.22, lw=0.4)

    # ---- (b) the fitting-artefact signature ----
    for w, c, lab in WINDOWS:
        dphi = np.array([r['dphi_deg'] for r in rows[w]])
        sd = np.array([r['slope_dyn'] for r in rows[w]])
        rr = float(np.corrcoef(dphi, sd)[0, 1])
        print(f"  w={w:2d}: per-run corr(slope, excursion) r = {rr:+.3f} "
              f"over {len(sd)} runs")
        a2.scatter(dphi, sd, s=14, c=c, alpha=0.55, lw=0,
                   label=f'{lab.split(" --")[0]}   $r={rr:+.2f}$')
    sm = np.array([r['slope_mod'] for r in rows[w0]])
    a2.axhline(float(np.median(sm)), color='#e08214', lw=2.0,
               label=f'model slope (median {np.median(sm):+.1f})')
    a2.set_ylim(top=30)
    a2.set_xlabel(r'excursion range of the run [deg]', fontsize=10)
    a2.set_ylabel(r'fitted slope [mN$\cdot$m/deg]', fontsize=10)
    a2.set_title('(b) the slope correlates with how far the run tipped\n'
                 'a physical attitude dependence would not', fontsize=11)
    a2.legend(fontsize=8, loc='lower right')
    a2.grid(alpha=0.22, lw=0.4)

    # ---- (c) trimming the tail, campaign-wide ----
    for w, c, lab in WINDOWS:
        med, q1, q3, left = [], [], [], []
        for k in DROPS:
            v, lf = [], []
            for r in rows[w]:
                ph, gd, _ = r['trace']
                e = len(ph) - k
                if e < 15:
                    continue
                v.append(np.polyfit(ph[:e], gd[:e], 1)[0])
                lf.append(ph[e - 1])
            if not v:
                continue
            med.append(np.median(v)); q1.append(np.percentile(v, 25))
            q3.append(np.percentile(v, 75)); left.append(np.median(lf))
            print(f"    w={w:2d} k={k:2d}: median slope {np.median(v):8.1f}"
                  f"   median phi left {np.median(lf):5.2f} deg  (n={len(v)})")
        kk = DROPS[:len(med)]
        a3.fill_between(kk, q1, q3, color=c, alpha=0.16, lw=0)
        a3.plot(kk, med, '-o', color=c, lw=1.5, ms=4,
                label=f'{lab.split(" --")[0]} (median, IQR)')
        if w == w0:
            for k, lv in zip(kk, left):
                if k in (0, 10, 20, 30):
                    a3.annotate(f'{lv:.1f}°', xy=(k, q3[kk.index(k)]),
                                xytext=(0, 6), textcoords='offset points',
                                fontsize=7.5, color='0.35', ha='center')
    a3.axhline(float(np.median(sm)), color='#e08214', lw=2.0,
               label='model slope')
    a3.axvspan(0, 20, color='0.85', alpha=0.5, lw=0, zorder=0)
    a3.set_xlabel('trailing samples excluded from the fit', fontsize=10)
    a3.set_ylabel(r'fitted slope [mN$\cdot$m/deg]', fontsize=10)
    a3.set_title('(c) trimming the tail moves the slope, but spends the\n'
                 r'excursion: labels give the median $\varphi$ left',
                 fontsize=11)
    a3.legend(fontsize=8, loc='lower right')
    a3.grid(alpha=0.22, lw=0.4)

    fig.suptitle('The dynamic ground-effect check over all '
                 f'{n} runs (order 2, $z_{{CoM}}$ = {Z:.3f} m, '
                 'CAD parallel-axis $J_P$)', fontsize=12.5, y=0.965)
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")

    for w, _, lab in WINDOWS:
        sd = np.array([r['slope_dyn'] for r in rows[w]])
        print(f"  {lab:52s} slope median {np.median(sd):7.1f}, "
              f"range {sd.min():.1f}..{sd.max():.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
