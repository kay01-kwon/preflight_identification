#!/usr/bin/env python3
"""Free-exponent fit of the excitation windows, over the whole campaign.

The three-parameter family {cosh, sinh, 1} is partly degenerate over a
0.8 s window -- a smaller exponent with a larger sinh coefficient
mimics a larger one -- so the free-C2 result of
analysis/cosh_differentiator.py could not be read as a measurement.
This script removes that degeneracy by dropping the sinh, fitting the
CONSTRAINED form the manuscript uses,

    omega(tau) = C1 (cosh(C2 tau) - 1)          [+ c]

with omega(0) = 0 imposed.  C1 is linear once C2 is fixed, so each run
is a one-dimensional search with a closed-form inner solve.  Two
variants are reported: two coefficients (C1, C2) and three (C1, C2 and
a baseline c, since the gyro carries a bias b in model (1)).

For every run it reports the fitted exponent against two references:
the pipeline's calibrated C2 (estimate_rig_constants, per dataset) and
the CAD parallel-axis prediction sqrt(W z / J_P) with
J_P = J_COM + m (z^2 + l_p^2).

Usage: PYTHONPATH=<stubs> python analysis/c2_campaign_fit.py [z_com]
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np

_R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R)); sys.path.insert(0, str(_R / 'analysis'))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from analysis.ge_dynamics_check import MASS_KG, G, j_parallel

Z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.261
GRID = np.linspace(1.5, 22.0, 1400)


def fit(tau, om, c2v, baseline):
    u = np.cosh(c2v * tau) - 1.0
    A = np.column_stack([u, np.ones_like(u)]) if baseline else u[:, None]
    co, *_ = np.linalg.lstsq(A, om, rcond=None)
    return float(np.mean((om - A @ co) ** 2)), co


def best(tau, om, baseline):
    r = [fit(tau, om, c, baseline)[0] for c in GRID]
    i = int(np.argmin(r))
    return GRID[i], r[i]


def main():
    rows = []
    print(f"  z_CoM = {Z:.3f} m;  constrained fit omega = C1(cosh C2 tau - 1)\n")
    print(f"  {'dataset':<14}{'bag':<12}{'C2 cal':>8}{'C2 CAD':>8}"
          f"{'C2 fit2':>9}{'C2 fit3':>9}{'RMS fit2':>10}{'RMS cal':>9}")
    for case in sorted(MASS_KG):
        for axn in ('Mx', 'My'):
            d = _R / 'DataSet' / 'exp' / case / axn
            if not d.is_dir():
                continue
            axis = 'x' if axn == 'Mx' else 'y'
            W = MASS_KG[case] * G
            c2_cad = float(np.sqrt(W * Z / j_parallel(axis, Z, MASS_KG[case])))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    bags = load_excitation_dataset(d)
                    c2_cal, kg = cvp.estimate_rig_constants(bags, axis)
                    crits, _ = cvp.extract_piecewise_batch(
                        bags, axis, cosh_c2=c2_cal, ramp_gain=kg)
            except Exception as exc:                       # noqa: BLE001
                print(f"  {case}/{axn}: skipped ({type(exc).__name__})")
                continue
            for cr in sorted(crits, key=lambda c: c.bag_name):
                bg = next((b for b in bags if b.name == cr.bag_name), None)
                if bg is None:
                    continue
                s = 1.0 if cr.bag_name.startswith('pos') else -1.0
                sg = cvp.prepare_signals(bg, axis)
                rr, pp = math_tools.quaternion_to_euler_vectorized(
                    bg.odom.quaternion)
                pa = rr if axis == 'x' else pp
                nn = min(len(pa), len(sg['t']))
                _, ii = cvp.detect_excitation_window(
                    sg['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
                jj, ii = cr.onset_idx, min(ii, nn - 1)
                if ii - jj < 20:
                    continue
                sl = slice(jj, ii + 1)
                tau = sg['t'][sl] - sg['t'][jj]
                om = s * np.asarray(sg['omega'][:nn], float)[sl]
                c2a, ra = best(tau, om, False)
                c2b, _ = best(tau, om, True)
                rcal, _ = fit(tau, om, c2_cal, False)
                print(f"  {case+'/'+axn:<14}{cr.bag_name:<12}{c2_cal:8.2f}"
                      f"{c2_cad:8.2f}{c2a:9.2f}{c2b:9.2f}"
                      f"{np.rad2deg(np.sqrt(ra)):10.3f}"
                      f"{np.rad2deg(np.sqrt(rcal)):9.3f}")
                rows.append((case, axn, c2_cal, c2_cad, c2a, c2b,
                             np.sqrt(ra), np.sqrt(rcal)))
    if not rows:
        return 1
    a = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows])
    print(f"\n  {len(rows)} runs")
    print(f"  calibrated C2 : median {np.median(a[:,0]):.2f}  "
          f"range {a[:,0].min():.2f}-{a[:,0].max():.2f}")
    print(f"  CAD  C2       : median {np.median(a[:,1]):.2f}  "
          f"range {a[:,1].min():.2f}-{a[:,1].max():.2f}")
    print(f"  fitted C2 (2) : median {np.median(a[:,2]):.2f}  "
          f"range {a[:,2].min():.2f}-{a[:,2].max():.2f}")
    print(f"  fitted C2 (3) : median {np.median(a[:,3]):.2f}  "
          f"range {a[:,3].min():.2f}-{a[:,3].max():.2f}")
    print(f"\n  fitted/CAD        : median {np.median(a[:,2]/a[:,1]):.3f}")
    print(f"  fitted/calibrated : median {np.median(a[:,2]/a[:,0]):.3f}")
    better = int(np.sum(a[:, 4] < a[:, 5]))
    print(f"  free fit beats the calibrated exponent on {better}/{len(rows)}"
          f" runs (median RMS {np.rad2deg(np.median(a[:,4])):.3f} vs "
          f"{np.rad2deg(np.median(a[:,5])):.3f} deg/s)")
    jp = np.array([MASS_KG[r[0]] * G * Z / r[4] ** 2 for r in rows])
    fl = np.array([j_parallel('x' if r[1] == 'Mx' else 'y', Z, MASS_KG[r[0]])
                   for r in rows])
    print(f"  implied J_P = Wz/C2_fit^2 : median {np.median(jp):.3f} "
          f"vs parallel-axis {np.median(fl):.3f} kg m^2 "
          f"(ratio {np.median(jp/fl):.2f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
