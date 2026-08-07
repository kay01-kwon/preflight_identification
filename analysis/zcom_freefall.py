#!/usr/bin/env python3
"""
z_CoM from the free tip-over, with no load cell and no truth of any kind
========================================================================
At the moment peak the throttle is cut: the collective drops from ~21 N
to zero in ~0.25 s and the commanded moment goes with it.  Whatever the
vehicle does afterwards it does under GRAVITY ALONE, rotating about the
same landing-gear contact line — and on the runs that tip past the
balance point that is a large-amplitude pendulum, 25 deg to 60 deg of
travel.  Nothing in it depends on the load cell, on the CoM-offset
truth, on the ramp model, on the identified onset, or on the ground
effect (the rotors are stopped).

Let rho be the distance from the contact line to the CoM and psi the
angle that line makes with the upward vertical at rest, so the CoM sits
z_CoM = rho cos(psi) above the contact plane and h_0 = rho sin(psi)
horizontally INBOARD of it.  With dphi the rotation away from the
resting attitude, the CoM is at Theta = dphi - psi from the vertical and

    J_P Theta'' = W rho sin(Theta) ,

which passes through zero at dphi = psi — the balance angle, plainly
visible in the trace as the angular rate flattening before it runs away.
The energy integral needs no differentiation of the gyro:

    omega^2 = C - A cos(dphi) - B sin(dphi) ,
        A = 2 W z_CoM / J_P = 2 C_2^2 ,     B = 2 W h_0 / J_P

— a LINEAR least-squares fit in (C, A, B) on the measured angular rate
and attitude.  dphi MUST be referenced to the resting attitude: the
throttle is cut some 25 deg into the tip-over, already past the balance
point, so a free-phase-relative angle puts psi 25 deg too low.  Then

    z_CoM / h_0 = A / B          (mass-free, inertia-free)
    z_CoM       = h_0 A / B      with h_0 from the mocap circle fit
    J_P         = 2 W h_0 / B    if the weight is known
    C_2         = sqrt(A / 2)    directly, with no onset model

The balance angle psi = atan(B/A) is visible by eye in the trace: the
angular acceleration passes through zero as the CoM crosses over the
contact line, and the rotation only runs away after that.

Usage
-----
python analysis/zcom_freefall.py DataSet/exp
python analysis/zcom_freefall.py DataSet/exp --min-travel 20 --save-csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    prepare_signals, detect_excitation_window, estimate_pivot_from_mocap,
    commanded_ramp_rate, MOMENT_CAP)

G = 9.81


def free_phase(t, om, f, phi, i1, thrust_frac, min_samples):
    """Index slice of the coasting phase after the throttle cut."""
    nom = float(np.median(f[max(0, i1 - 40):i1 + 1]))
    if not np.isfinite(nom) or nom <= 0:
        return None
    off = np.where(f[i1:] < thrust_frac * nom)[0]
    if len(off) == 0:
        return None
    j0 = i1 + int(off[0])
    # the pendulum ends when the vehicle stops turning (it has landed on
    # its side, or fallen back): first sign change of omega, or the last
    # sample before |omega| collapses from its peak
    w = om[j0:]
    if len(w) < min_samples:
        return None
    s = np.sign(w[0])
    turn = np.where(np.sign(w) != s)[0]
    j1 = j0 + (int(turn[0]) if len(turn) else len(w)) - 1
    # trim the impact: cut at the peak |omega| if it collapses afterwards
    seg = np.abs(om[j0:j1 + 1])
    if len(seg) > 4:
        k = int(np.argmax(seg))
        if k < len(seg) - 2 and seg[-1] < 0.4 * seg[k]:
            j1 = j0 + k
    return (j0, j1) if j1 - j0 + 1 >= min_samples else None


def fit_run(t, om, phi, sl, sgn, phi_rest):
    """omega^2 = C - A cos(dphi) - B sin(dphi),  dphi from the REST attitude.

    A = 2 W z_CoM / J_P = 2 C_2^2 ,   B = 2 W h_0 / J_P ,  tan(psi) = B/A.
    Referencing dphi to the resting attitude is essential: the throttle is
    cut ~25 deg into the tip-over, already past the balance point, so a
    free-phase-relative angle would put psi 25 deg too low.
    """
    j0, j1 = sl
    dphi = sgn * (phi[j0:j1 + 1] - phi_rest)
    y = om[j0:j1 + 1] ** 2
    X = np.column_stack([np.ones_like(dphi), -np.cos(dphi), -np.sin(dphi)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    dof = max(len(y) - 3, 1)
    se = np.sqrt(r @ r / dof * np.diag(np.linalg.pinv(X.T @ X)))
    return (c, se, float(np.sqrt(r @ r / len(y))),
            float(np.degrees(dphi[-1] - dphi[0])),
            float(np.degrees(dphi[0])))


def main():
    p = argparse.ArgumentParser(
        description="z_CoM from the free (unpowered) tip-over.")
    p.add_argument('root', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--thrust-frac', type=float, default=0.05,
                   help="collective below this fraction of nominal counts "
                        "as coasting")
    p.add_argument('--min-travel', type=float, default=15.0,
                   help="minimum free-phase travel [deg] for a run to be "
                        "used — the fit needs the large-angle curvature")
    p.add_argument('--min-samples', type=int, default=8)
    p.add_argument('--mass', type=float, default=3.22,
                   help="vehicle mass [kg], only for J_P and the W z_CoM "
                        "readout; z_CoM itself does not use it")
    p.add_argument('--save-csv', action='store_true')
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root
    rows = []
    for d in sorted(root.glob('case_*/M[xy]')):
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
        for b in bags:
            sig = prepare_signals(b, axis)
            t, om, M, f = sig['t'], sig['omega'], sig['moment'], sig['f_col']
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                b.odom.quaternion)
            phi = (roll if axis == 'x' else pitch)
            n = min(len(t), len(phi))
            t, om, M, f, phi = t[:n], om[:n], M[:n], f[:n], phi[:n]
            i0, i1 = detect_excitation_window(M, moment_cap=MOMENT_CAP.get(axis))
            sgn = +1.0 if b.name.startswith('pos') else -1.0
            sl = free_phase(t, om, f, phi, i1, args.thrust_frac,
                            args.min_samples)
            if sl is None:
                continue
            phi_rest = float(np.median(phi[:max(1, i0)]))
            c, se, rms, travel, entry = fit_run(t, om, phi, sl, sgn, phi_rest)
            if travel < args.min_travel or c[1] <= 0 or c[2] <= 0:
                rows.append(dict(case=d.parent.name, axis=d.name, bag=b.name,
                                 travel=travel, used=0))
                continue
            # arm from the same circle fit the pipeline already uses
            piv = estimate_pivot_from_mocap(b, t[i0], axis)
            h0 = piv['pivot_abs'] * 1e-3
            A, B = c[1], c[2]
            psi = np.degrees(np.arctan2(B, A))
            z = h0 * A / B
            W = args.mass * G
            rows.append(dict(
                case=d.parent.name, axis=d.name, bag=b.name,
                rate=commanded_ramp_rate(b.name) or np.nan,
                n=sl[1] - sl[0] + 1, travel=travel, entry=entry, used=1,
                A=A, B=B, seA=se[1], seB=se[2], rms=rms,
                psi=psi, h0=h0, z=z, c2=np.sqrt(A / 2.0),
                JP=2 * W * h0 / B if h0 == h0 else np.nan))
        print(f"  assessed {d}", flush=True)

    use = [r for r in rows if r.get('used') and np.isfinite(r.get('z', np.nan))]
    print(f"\n{len(use)} usable free tip-overs out of {len(rows)} runs "
          f"(travel >= {args.min_travel:.0f} deg)\n")
    print(f"  {'case':9}{'ax':4}{'bag':14}{'entry°':>8}{'travel°':>8}{'n':>4}"
          f"{'A=2C2²':>9}{'B':>9}{'psi°':>7}{'h0 [mm]':>9}"
          f"{'z_CoM [mm]':>12}{'C2':>7}{'J_P':>8}")
    for r in sorted(use, key=lambda r: (r['case'], r['axis'], r['bag'])):
        print(f"  {r['case']:9}{r['axis']:4}{r['bag']:14}{r['entry']:8.1f}"
              f"{r['travel']:8.1f}{r['n']:4d}{r['A']:9.1f}{r['B']:9.1f}{r['psi']:7.1f}"
              f"{1e3 * r['h0']:9.1f}{1e3 * r['z']:12.0f}{r['c2']:7.2f}"
              f"{r['JP']:8.3f}")

    def stat(k, s=1.0):
        v = s * np.array([r[k] for r in use if np.isfinite(r[k])])
        return v.mean(), v.std(ddof=1), np.median(v), len(v)

    print()
    for ax in (None, 'Mx', 'My'):
        sub = [r for r in use if ax is None or r['axis'] == ax]
        if len(sub) < 2:
            continue
        z = 1e3 * np.array([r['z'] for r in sub])
        ps = np.array([r['psi'] for r in sub])
        c2 = np.array([r['c2'] for r in sub])
        jp = np.array([r['JP'] for r in sub])
        lbl = ax or 'all'
        print(f"  {lbl:4} n={len(sub):2d}  z_CoM {z.mean():6.0f} ± "
              f"{z.std(ddof=1) / np.sqrt(len(z)):.0f} mm (sd {z.std(ddof=1):.0f})"
              f"   psi {ps.mean():5.1f}°   C2 {c2.mean():4.2f} ± "
              f"{c2.std(ddof=1):.2f} rad/s   J_P {jp.mean():.3f} ± "
              f"{jp.std(ddof=1):.3f} kg m²")
    # pos/neg pair: two equations, two unknowns (z_CoM and the CoM offset)
    byg = defaultdict(dict)
    for r in use:
        byg[(r['case'], r['axis'])].setdefault(
            'pos' if r['bag'].startswith('pos') else 'neg', []).append(r)
    print(f"\n  pos/neg pair solve — h_0 = a -+ lambda gives two equations "
          f"in (z_CoM, lambda):")
    print(f"  {'case':9}{'ax':4}{'r_pos':>8}{'r_neg':>8}{'a_pos':>8}"
          f"{'a_neg':>8}{'lambda [mm]':>13}{'z_CoM [mm]':>12}")
    zz = []
    for k in sorted(byg):
        v = byg[k]
        if 'pos' not in v or 'neg' not in v:
            continue
        rp = float(np.mean([x['A'] / x['B'] for x in v['pos']]))
        rn = float(np.mean([x['A'] / x['B'] for x in v['neg']]))
        apos = float(np.mean([x['h0'] for x in v['pos']]))
        aneg = float(np.mean([x['h0'] for x in v['neg']]))
        lam = (apos * rp - aneg * rn) / (rp + rn)
        z = (apos - lam) * rp
        zz.append(z)
        print(f"  {k[0]:9}{k[1]:4}{rp:8.2f}{rn:8.2f}{1e3 * apos:8.1f}"
              f"{1e3 * aneg:8.1f}{1e3 * lam:13.1f}{1e3 * z:12.0f}")
    if zz:
        zz = 1e3 * np.array(zz)
        print(f"  pair-solved z_CoM: {zz.mean():.0f} ± "
              f"{zz.std(ddof=1) / np.sqrt(len(zz)):.0f} mm "
              f"(sd {zz.std(ddof=1):.0f}, n = {len(zz)})")

    # parallel-axis closure: J_cm = J_P - m (h_0^2 + z^2) must agree
    # between the two axes for a six-fold symmetric airframe
    print("\n  parallel-axis closure  J_cm = J_P - m (h_0^2 + z_CoM^2):")
    for ax in ('Mx', 'My'):
        sub = [r for r in use if r['axis'] == ax]
        if not sub:
            continue
        jcm = np.array([r['JP'] - args.mass * (r['h0'] ** 2 + r['z'] ** 2)
                        for r in sub])
        print(f"    {ax}: J_cm = {jcm.mean():.3f} ± {jcm.std(ddof=1):.3f} "
              f"kg m^2   (n = {len(sub)})")
    print("\n  z_CoM = h_0 A/B uses no mass, no inertia, no load cell, no "
          "onset model\n  and no ground-effect model (the rotors are "
          "stopped).  C_2 = sqrt(A/2) is\n  the same rig constant the ramp "
          "calibration estimates, here measured\n  directly off the free "
          "pendulum — and h_0 carries the unknown CoM offset\n  (|lambda| <= "
          "20 mm on h_0 ~ 113 mm, i.e. <= 18% on z_CoM) as its only "
          "systematic.")

    if args.save_csv and rows:
        out.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(out / 'zcom_freefall_runs.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nTable -> {out / 'zcom_freefall_runs.csv'}")


if __name__ == '__main__':
    main()
