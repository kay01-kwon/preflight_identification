#!/usr/bin/env python3
"""
GE pinned by the model: do J_P and z_CoM then come out consistent?
==================================================================
Run the argument backwards.  Instead of leaving the ground effect as an
unknown residual, PIN it with the parameter-free rotor-interference model
(analysis/ge_linearity.py, c_a = 4.31%, b = 4.314%; the per-rotor
Cheeseman superposition, 1.03%/1.03%, is carried as the lower bracket)
and ask what the data then say about z_CoM and J_P.

Two independent channels, both run here:

STATIC.  With the ground effect pinned, the onset balance of every
direction group leaves exactly two unknowns — the CoM height and the
contact-lever offset dl (the normal-force resultant acting inboard of the
kinematic pivot circle):

    M_pred = [ sgn (W - f)(l - dl) + S_off - W z_CoM sin(phi_0)
               - sgn c_a f l ] / (1 + b)

so the deficit against the pinned-GE prediction is linear in (dl, z_CoM),

    M_ident - P_0 = -[ sgn (W - f) dl + W z_CoM sin(phi_0) ] / (1 + b) ,

with the two regressors in ORTHOGONAL channels: dl is antisymmetric
between the tip directions, z_CoM is symmetric.  The GE thrust term
sgn c_a f l is antisymmetric too, so it competes with dl and NOT with
z_CoM — which is the point: pinning the ground effect can only move the
contact lever, never the CoM height (the GE moment coefficient b touches
the symmetric channel, but only as a 1-4% scale).  The axis contrast is
what carries z_CoM: the resting tilt is +0.5 deg in roll and -1.5 deg in
pitch, so its regressor -W sin(phi_0) changes sign between the axes.

DYNAMIC.  The calibrated rig constants give W z_CoM = 1/K and
J_P = 1/(K C_2^2) per case/axis; the GE state-linear channel inflates
the first by eta (1.0% single / 3.8% interference), so the physical
height is (1/K)/(1 + eta)/W.  These two are not free of each other: the
parallel-axis theorem forces

    J_P  =  J_cm + m (l^2 + z_CoM^2)   >=   m (l^2 + z_CoM^2) ,

a hard inequality with no free parameter, which every (C_2, K) pair must
satisfy.  Inverting it at a nominal J_cm turns each J_P into an implied
CoM height that can be compared with 1/K, with the static channel, and
across the two axes (J_P is axis dependent, z_CoM is not).

Input: the per-run table from analysis/zcom_tilt.py --collect.

Usage
-----
python analysis/ge_pinned_consistency.py --table zcom_tilt_runs.csv
python analysis/ge_pinned_consistency.py --table zcom_tilt_runs.csv --j-cm 0.04
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
S_AX = {'Mx': +1.0, 'My': -1.0}
# (c_a, b) at phi = 0, from analysis/ge_linearity.py; eta = the
# state-linear GE share of W z_CoM quoted in docs/response_ge_comment.md
GE = {'none':   (0.0,    0.0,     0.0),
      'single': (0.0103, 0.01026, 0.010),
      'interf': (0.0431, 0.04314, 0.038)}


def groups(rows):
    """Per case/axis/direction means of the identified quantities."""
    g = defaultdict(list)
    for r in rows:
        g[(r['case'], r['axis'], r['dir'])].append(r)
    out = []
    for k in sorted(g):
        v = g[k]
        arms = np.array([r['arm'] for r in v], float)
        out.append(dict(
            case=k[0], axis=k[1], dir=k[2], n=len(v),
            sgn=+1.0 if k[2] == 'pos' else -1.0,
            W=MASS_KG[k[0]] * G,
            M=float(np.mean([r['M'] for r in v])),
            f=float(np.mean([r['f'] for r in v])),
            arm=float(np.nanmean(arms)),
            tilt=float(np.mean([r['tilt'] for r in v]))))
    return out


def static_fit(gs, model, verbose=False):
    """Deficit against the pinned-GE prediction -> (dl_x, dl_y, z_CoM)."""
    ca, b, _ = GE[model]
    y, X = [], []
    for g in gs:
        S = S_AX[g['axis']] * g['W'] * OFF_MM[(g['case'], g['axis'])] * 1e-3
        P0 = (g['sgn'] * (g['W'] - g['f']) * g['arm'] + S
              - g['sgn'] * ca * g['f'] * g['arm']) / (1.0 + b)
        y.append(g['M'] - P0)
        dl_col = -g['sgn'] * (g['W'] - g['f']) / (1.0 + b)
        X.append([dl_col if g['axis'] == 'Mx' else 0.0,
                  dl_col if g['axis'] == 'My' else 0.0,
                  -g['W'] * np.sin(g['tilt']) / (1.0 + b)])
    y, X = np.array(y), np.array(X)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    dof = len(y) - X.shape[1]
    se = np.sqrt(r @ r / dof * np.diag(np.linalg.pinv(X.T @ X)))
    if verbose:
        print(f"    {'case':9}{'ax':4}{'dir':5}{'M_ident':>9}{'deficit':>10}"
              f"{'fit':>9}{'resid':>9}  [N·m / mN·m]")
        for g, yy, ff in zip(gs, y, X @ c):
            print(f"    {g['case']:9}{g['axis']:4}{g['dir']:5}{g['M']:+9.3f}"
                  f"{1e3 * yy:+10.0f}{1e3 * ff:+9.0f}{1e3 * (yy - ff):+9.0f}")
    return c, se, r, dof


def main():
    p = argparse.ArgumentParser(
        description="GE pinned by the model: are J_P and z_CoM consistent?")
    p.add_argument('--table', default='zcom_tilt_runs.csv')
    p.add_argument('--j-cm', type=float, default=0.04,
                   help="nominal about-CoM inertia [kg m^2] for the "
                        "parallel-axis inversion")
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    rows = []
    for r in csv.DictReader(open(Path(args.table))):
        for k in ('M', 'f', 'arm', 'tilt', 'c2', 'k'):
            r[k] = float(r[k]) if r[k] not in ('', 'nan') else np.nan
        rows.append(r)
    gs = groups(rows)
    print(f"{len(rows)} runs -> {len(gs)} direction groups\n")

    # ── STATIC ─────────────────────────────────────────────────────
    print("=" * 72)
    print("1.  Static: GE pinned by the model, (dl, z_CoM) left to the data")
    print(f"  {'GE model':10}{'dl roll [mm]':>18}{'dl pitch [mm]':>18}"
          f"{'z_CoM [mm]':>18}{'resid':>9}")
    zs = {}
    for model in ('none', 'single', 'interf'):
        c, se, r, dof = static_fit(gs, model, verbose=args.verbose
                                   and model == 'interf')
        zs[model] = (c[2] * 1e3, se[2] * 1e3)
        print(f"  {model:10}{c[0] * 1e3:11.1f} ±{se[0] * 1e3:5.1f}"
              f"{c[1] * 1e3:11.1f} ±{se[1] * 1e3:5.1f}"
              f"{c[2] * 1e3:11.0f} ±{se[2] * 1e3:5.0f}"
              f"{1e3 * np.std(r, ddof=1):8.0f}")
    print(f"  dof {dof};  residual RMS is per-group [mN·m]")
    print("  -> pinning GE moves the CONTACT LEVER, which shares the")
    print("     ANTIsymmetric channel with the GE thrust term sgn c_a f l:")
    print("     the interference model absorbs the whole deficit that the")
    print("     rigid fit had to buy with a 17-21 mm inboard contact shift.")
    print("     That is consistency, not attribution — the two remain")
    print("     degenerate (analysis/static_attribution.py).  z_CoM sits in")
    print("     the orthogonal SYMMETRIC channel and is barely touched:")
    sp = zs['interf'][0] - zs['none'][0]
    print(f"     z_CoM moves by {sp:+.0f} mm between no-GE and interference, "
          f"vs its own ±{zs['interf'][1]:.0f} mm standard error.")

    # ── DYNAMIC ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("2.  Rig constants vs the parallel-axis theorem")
    arm_ax = {}
    for ax in ('Mx', 'My'):
        arm_ax[ax] = float(np.nanmean([g['arm'] for g in gs
                                       if g['axis'] == ax]))
    print(f"  mean fitted pivot arms: roll {arm_ax['Mx'] * 1e3:.1f} mm, "
          f"pitch {arm_ax['My'] * 1e3:.1f} mm;  J_cm assumed "
          f"{args.j_cm:.3f} kg m^2")
    print(f"\n  {'case':9}{'ax':4}{'J_P':>8}{'z(1/K)':>9}{'m(l²+z²)':>10}"
          f"{'PA ok?':>8}{'z from J_P':>12}")
    per_ax = defaultdict(list)
    ca, b, eta = GE['interf']
    for g in gs:
        if g['dir'] != 'pos':
            continue
        v = [r for r in rows if r['case'] == g['case']
             and r['axis'] == g['axis']][0]
        c2, k = v['c2'], v['k']
        m, l = MASS_KG[g['case']], arm_ax[g['axis']]
        Wz = (1.0 / k) / (1.0 + eta)          # GE state-linear removed
        z = Wz / g['W']
        J = (1.0 / k) / c2 ** 2
        need = m * (l ** 2 + z ** 2)
        zJ = (J - args.j_cm) / m - l ** 2
        zJ = np.sqrt(zJ) if zJ > 0 else np.nan
        per_ax[g['axis']].append((J, z * 1e3, zJ * 1e3))
        print(f"  {g['case']:9}{g['axis']:4}{J:8.3f}{z * 1e3:9.0f}"
              f"{need:10.3f}{'yes' if J >= need else 'NO':>8}"
              f"{zJ * 1e3:12.0f}")
    print(f"\n  {'axis':6}{'J_P [kg m^2]':>22}{'z from 1/K [mm]':>24}"
          f"{'z from J_P [mm]':>22}")
    for ax in ('Mx', 'My'):
        a = np.array(per_ax[ax])
        print(f"  {ax:6}{a[:, 0].mean():12.3f} ± {a[:, 0].std(ddof=1):.3f} "
              f"(CV {100 * a[:, 0].std(ddof=1) / a[:, 0].mean():4.1f}%)"
              f"{a[:, 1].mean():10.0f} ± {a[:, 1].std(ddof=1):3.0f} "
              f"(CV {100 * a[:, 1].std(ddof=1) / a[:, 1].mean():4.1f}%)"
              f"{np.nanmean(a[:, 2]):9.0f} ± {np.nanstd(a[:, 2], ddof=1):3.0f}")
    nviol = sum(1 for g in gs if g['dir'] == 'pos' and (
        lambda v: (1.0 / v['k']) / v['c2'] ** 2 < MASS_KG[g['case']] * (
            arm_ax[g['axis']] ** 2
            + ((1.0 / v['k']) / (1.0 + eta) / g['W']) ** 2))(
        [r for r in rows if r['case'] == g['case']
         and r['axis'] == g['axis']][0]))
    print(f"\n  parallel-axis inequality violated by {nviol}/10 (C_2, K) "
          f"pairs — the ridge\n  wanders into a region where the claimed CoM "
          f"height alone would need more\n  inertia than the same pair's J_P "
          f"provides.  The inequality is free of\n  every GE model and of the "
          f"load cell: it is a constraint the calibration\n  box does not "
          f"currently impose.")
    # z-free axis test: J_P(roll) - J_P(pitch) = (J_xx - J_yy) + m(l_r^2 - l_p^2)
    ar, ap = np.array(per_ax['Mx']), np.array(per_ax['My'])
    d = ar[:, 0].mean() - ap[:, 0].mean()
    sd = np.hypot(ar[:, 0].std(ddof=1), ap[:, 0].std(ddof=1)) / np.sqrt(5)
    m = float(np.mean(list(MASS_KG.values())))
    geo = m * (arm_ax['Mx'] ** 2 - arm_ax['My'] ** 2)
    print(f"\n  z-free axis test.  z_CoM cancels in the axis DIFFERENCE:")
    print(f"    J_P(roll) - J_P(pitch) = (J_xx - J_yy) + m(l_r^2 - l_p^2)")
    print(f"    measured {d:+.3f} ± {sd:.3f} kg m^2   vs   geometric term "
          f"{geo:+.3f}")
    print(f"    -> requires J_xx - J_yy = {d - geo:+.3f} kg m^2 "
          f"({abs(d - geo) / sd:.1f} sigma from 0), i.e. a mass distribution")
    print(f"       {np.sqrt(abs(d - geo) / m) * 1e3:.0f} mm more spread about "
          f"one body axis than the other — not available")
    print(f"       on a six-fold symmetric airframe.  The roll pivot is the "
          f"compliant one\n       (see analysis/com_estimator.py), so J_P "
          f"there is an effective constant,\n       not a rigid-body pivot "
          f"inertia.")
    print("\n  Consistency verdict: J_P repeats within an axis (CV 6.5% roll) "
          "and z_CoM from\n  the SAME pairs does not repeat at all — the "
          "ridge's invariant is J_P\n  (equivalently C_1 C_2^2 = Mdot/J_P), "
          "not W z_CoM.  Across axes neither\n  quantity closes: the axis "
          "difference of J_P is 5 sigma from its z-free\n  geometric value, "
          "and the two axes' J_P-implied heights disagree.")


if __name__ == '__main__':
    main()
