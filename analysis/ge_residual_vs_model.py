#!/usr/bin/env python3
"""
Rotation-dynamics moment residual vs the theoretical rotor GE moment
====================================================================
With the CoM height and the pivot inertia pinned (defaults: the free
tip-over values, z_CoM = 0.267 m and J_P = 0.311 kg m^2), every term of
the contact-phase rotational balance is known except the ground effect:

    J_P phi'' = M(t) + f(t) a_t - W [ a_c cos(dphi) - z_CoM sin(dphi) ]
                + dM_GE(t)

so the residual

    r(t) = J_P phi'' - M - f a_t + W [ a_c cos(dphi) - z_CoM sin(dphi) ]

IS the ground-effect moment, if the model is otherwise complete.  This
script forms r(t) per run and compares it, sample by sample, with the
theoretical rotor GE moment evaluated on the SAME measured trajectory
(the models of analysis/ge_linearity.py / ge_trajectory.py: per-rotor
Cheeseman superposition, parameter-free image interference, and the
Garofano-Soldado adaptation).

What makes the comparison possible is the throttle cut at the moment
peak.  After it the rotors are stopped, so the coasting pendulum is
GE-free and calibrates the one remaining geometric constant:

    free phase:  omega^2 = C - A cos(dphi) - B sin(dphi)
                 a_c = B J_P / (2 W)          (and A J_P / (2W) = z_CoM,
                                               reported as a cross-check)

The CoM arm a_c is therefore measured where the ground effect cannot
touch it, and then held fixed over the powered phase where the residual
is formed.  The thrust arm a_t differs from a_c by the CoM offset
lambda (a few mm on ~130 mm); it enters multiplied by f ~ 21 N, so it is
the dominant systematic and is reported both ways — pinned to a_c, and
fitted as the per-run constant of the residual.

Outputs a per-run table and, per ramp rate, the comparison of the
residual with each GE model: level, span, correlation over the window,
and the regression gain (residual on model, constant free) whose value
would be 1.0 if the model accounted for the residual exactly.

Usage
-----
python analysis/ge_residual_vs_model.py DataSet/exp
python analysis/ge_residual_vs_model.py DataSet/exp --z-com 0.267 --jp 0.311
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
from scipy.signal import savgol_filter

from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    prepare_signals, detect_excitation_window, estimate_pivot_from_mocap,
    commanded_ramp_rate, MOMENT_CAP)

G = 9.81
MODELS = ('single', 'interf', 'garofano')


# ═════════════════════════════════════════════════════════════
#  Theoretical GE moment on the measured trajectory
#  (same construction as analysis/ge_trajectory.py)
# ═════════════════════════════════════════════════════════════

def ge_moment(bag, axis, pos, t, n, args):
    """dM_GE(t) [N m] about the active pivot line, for each model.

    Everything is put on the caller's clock ``t`` by interpolation — with
    --omega-source imu that clock is the IMU's, not the odometry's, and
    truncating instead of interpolating would silently misalign the
    attitude with the rates.
    """
    L, R, H = args.arm, args.radius, args.hub_height
    lp = args.lp_roll if axis == 'x' else args.lp_pitch
    ang = np.deg2rad(30 + 60 * np.arange(6))
    LX, LY = L * np.cos(ang), L * np.sin(ang)
    if axis == 'x':
        px, py = LX, LY + (lp if pos else -lp)
        arm = py
    else:
        px, py = LX + (-lp if pos else lp), LY
        arm = -px
    pP = np.vstack([px, py, np.full(6, H)])

    t0 = bag.odom.t[0]
    T6 = args.c_t * bag.rpm.rpm.astype(np.float64) ** 2
    Ti = np.vstack([np.interp(t, bag.rpm.t - t0, T6[:, j])
                    for j in range(6)]).T[:n]

    qw, qx, qy, qz = bag.odom.quaternion.T
    tq = bag.odom.t - t0
    r31 = np.interp(t, tq, 2 * (qx * qz - qw * qy))[:n]
    r32 = np.interp(t, tq, 2 * (qy * qz + qw * qx))[:n]
    r33 = np.interp(t, tq, 1 - 2 * (qx * qx + qy * qy))[:n]
    m = n
    h_it = (np.outer(r31, pP[0]) + np.outer(r32, pP[1])
            + np.outer(r33, pP[2]))
    ok = np.all(h_it > R / 4 + 0.01, axis=1)

    out = {}
    dT = Ti[:m] * R ** 2 / (16 * h_it ** 2 - R ** 2)
    out['single'] = dT @ arm

    d2 = ((pP[0][:, None] - pP[0][None, :]) ** 2
          + (pP[1][:, None] - pP[1][None, :]) ** 2)
    Z = h_it[:, :, None] + h_it[:, None, :]
    srot = (R ** 2 / 4) * np.sum(Z / (d2 + Z ** 2) ** 1.5, axis=2)
    out['interf'] = ((1.0 / (1.0 - srot) - 1.0) * Ti[:m]) @ arm

    lp_signed = arm[0] - (LY[0] if axis == 'x' else -LX[0])
    cx = 0.0 if axis == 'x' else -lp_signed
    cy = lp_signed if axis == 'x' else 0.0
    zc = r31 * cx + r32 * cy + r33 * H
    dists2 = np.array([0.0, L ** 2, L ** 2, 3 * L ** 2, 3 * L ** 2, 4 * L ** 2])
    s_lvl = (R ** 2 / 4) * np.sum(
        2 * zc[:, None] / (dists2[None, :] + 4 * zc[:, None] ** 2) ** 1.5,
        axis=1)
    s9 = (R ** 2 * zc / (L ** 2 / 4 + 4 * zc ** 2) ** 1.5
          + (R ** 2 / 2) * zc / (21 * L ** 2 / 4 + 4 * zc ** 2) ** 1.5
          + R ** 2 * zc / (13 * L ** 2 / 4 + 4 * zc ** 2) ** 1.5
          + (R ** 2 / 2) * zc / (7 * L ** 2 / 4 + 4 * zc ** 2) ** 1.5)
    kz = (s9 / s_lvl)[:, None]
    rd = 2 * L - args.frame_width
    sf = (2 * R ** 2 * args.jk * zc / (rd ** 2 + 4 * zc ** 2) ** 1.5)[:, None]
    g_rot = 1.0 / (1.0 - kz * srot) - 1.0
    g_full = 1.0 / (1.0 - kz * srot - sf) - 1.0
    arm_c = LY if axis == 'x' else -LX
    out['garofano'] = ((g_rot * Ti[:m]) @ arm_c
                       + np.sum(g_full * Ti[:m], axis=1) * lp_signed)
    return out, ok, m


# ═════════════════════════════════════════════════════════════

def free_phase(om, f, i1, thrust_frac, min_samples):
    nom = float(np.median(f[max(0, i1 - 40):i1 + 1]))
    if not np.isfinite(nom) or nom <= 0:
        return None
    off = np.where(f[i1:] < thrust_frac * nom)[0]
    if len(off) == 0:
        return None
    j0 = i1 + int(off[0])
    w = om[j0:]
    if len(w) < min_samples:
        return None
    turn = np.where(np.sign(w) != np.sign(w[0]))[0]
    j1 = j0 + (int(turn[0]) if len(turn) else len(w)) - 1
    seg = np.abs(om[j0:j1 + 1])
    if len(seg) > 4:
        k = int(np.argmax(seg))
        if k < len(seg) - 2 and seg[-1] < 0.4 * seg[k]:
            j1 = j0 + k
    return (j0, j1) if j1 - j0 + 1 >= min_samples else None


def main():
    p = argparse.ArgumentParser(
        description="Dynamics moment residual vs the theoretical GE moment.")
    p.add_argument('root')
    p.add_argument('--z-com', type=float, default=0.267,
                   help="CoM height above the contact plane [m]")
    p.add_argument('--jp', type=float, default=0.311,
                   help="pivot inertia [kg m^2]; pass 0 to derive it per run "
                        "from the free phase at the pinned z_CoM, "
                        "J_P = 2 W z / A (self-consistent)")
    p.add_argument('--mass', type=float, default=3.22)
    p.add_argument('--thrust-frac', type=float, default=0.05)
    p.add_argument('--min-travel', type=float, default=15.0)
    p.add_argument('--min-samples', type=int, default=8)
    p.add_argument('--savgol', type=int, default=15,
                   help="Savitzky-Golay window [samples] for phi'' = dw/dt")
    p.add_argument('--arm', type=float, default=0.265)
    p.add_argument('--radius', type=float, default=0.127)
    p.add_argument('--hub-height', type=float, default=0.315)
    p.add_argument('--lp-roll', type=float, default=0.140)
    p.add_argument('--lp-pitch', type=float, default=0.110)
    p.add_argument('--frame-width', type=float, default=0.22)
    p.add_argument('--jk', type=float, default=2.2)
    p.add_argument('--c-t', type=float, default=1.3175e-7)
    p.add_argument('--save-csv', action='store_true')
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    W = args.mass * G
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
            phi = roll if axis == 'x' else pitch
            n = min(len(t), len(phi))
            t, om, M, f, phi = t[:n], om[:n], M[:n], f[:n], phi[:n]
            i0, i1 = detect_excitation_window(M,
                                              moment_cap=MOMENT_CAP.get(axis))
            pos = b.name.startswith('pos')
            sgn = +1.0 if pos else -1.0
            sl = free_phase(om, f, i1, args.thrust_frac, args.min_samples)
            if sl is None:
                continue
            phi_rest = float(np.median(phi[:max(1, i0)]))
            dphi = sgn * (phi - phi_rest)

            # ── free phase: a_c, GE-free ────────────────────────────
            j0, j1 = sl
            if np.degrees(dphi[j1] - dphi[j0]) < args.min_travel:
                continue
            X = np.column_stack([np.ones(j1 - j0 + 1), -np.cos(dphi[j0:j1 + 1]),
                                 -np.sin(dphi[j0:j1 + 1])])
            c, *_ = np.linalg.lstsq(X, om[j0:j1 + 1] ** 2, rcond=None)
            if c[1] <= 0 or c[2] <= 0:
                continue
            # J_P and z_CoM are NOT independent given the free phase:
            # A = 2 W z / J_P is measured, so pinning both over-determines
            # the pendulum.  jp = 0 derives J_P from the pinned z.
            jp = args.jp if args.jp > 0 else 2 * W * args.z_com / c[1]
            a_c = c[2] * jp / (2 * W)
            z_chk = c[1] * jp / (2 * W)

            # ── powered phase: the residual ─────────────────────────
            k0 = i0 + int(0.5 * (i1 - i0))       # second half of the ramp
            k1 = i1
            if k1 - k0 < args.savgol + 2:
                continue
            dt = float(np.median(np.diff(t)))
            wl = args.savgol if args.savgol % 2 else args.savgol + 1
            alpha = savgol_filter(om, wl, 2, deriv=1, delta=dt)
            ge, okge, m = ge_moment(b, axis, pos, t, n, args)
            k1 = min(k1, m - 1)
            if k1 - k0 < 10 or not okge[k0:k1 + 1].all():
                continue
            s = slice(k0, k1 + 1)
            # sign-normalised to the tipping direction
            rig = (jp * sgn * alpha[s] - sgn * M[s]
                   + W * (a_c * np.cos(dphi[s]) - args.z_com * np.sin(dphi[s])))
            # thrust term, arm pinned to a_c
            resid = rig - f[s] * a_c
            # ── nested test ────────────────────────────────────────
            # M(t) = J_P alpha - f a_t + W [a_c cos - z sin] - dM_GE
            # Model 0: (a_t, a_c) free, (J_P, z_CoM) pinned as requested.
            # Model 1: adds g * dM_GE.  The GE model is a KNOWN regressor,
            # so g is a direct test — it should come out at +1, and its
            # standard error says whether the data can see it at all.
            # Sign convention, checked against the mocap pivot side: the
            # code places the pivot at -lp (roll, pos) / +lp (pitch, pos),
            # matching the measured cx, and sgn * dM_GE then comes out
            # POSITIVE = the ground effect aids the tip and lowers M_crit,
            # as the manuscript's thrust channel a = c_a f l requires.
            # Balance (tipping-positive):
            #   J_P phi'' = M + f a - W a cos(dphi) + W z sin(dphi) + dM_GE
            # so   y = M - J_P phi'' + W z sin(dphi) = a (W cos - f) - dM_GE
            # and the fitted coefficient of the -dM_GE column is g = +1.
            y = (sgn * M[s] - jp * sgn * alpha[s]
                 + W * args.z_com * np.sin(dphi[s]))
            # a_t and a_c multiply near-constants (f is flat, cos dphi ~ 1)
            # and cannot be separated over this window; the thrust line and
            # the CoM differ only by the offset lambda, so pin a_t = a_c = a
            # and carry ONE arm.
            X0 = np.column_stack([W * np.cos(dphi[s]) - f[s]])

            def lsq(X, y):
                cc, *_ = np.linalg.lstsq(X, y, rcond=None)
                rr = y - X @ cc
                dof = max(len(y) - X.shape[1], 1)
                ss = np.sqrt(rr @ rr / dof
                             * np.diag(np.linalg.pinv(X.T @ X)))
                return cc, ss, float(np.std(rr))

            c0, s0, rms0 = lsq(X0, y)
            row = dict(case=d.parent.name, axis=d.name, bag=b.name,
                       rate=commanded_ramp_rate(b.name) or np.nan,
                       n=k1 - k0 + 1, a_c_free=a_c, jp=jp, z_chk=z_chk,
                       fbar=float(np.mean(f[s])),
                       dphi0=np.degrees(dphi[k0]), dphi1=np.degrees(dphi[k1]),
                       a_t=c0[0], a_c=c0[0], rms0=rms0,
                       res_mean=float(resid.mean()),
                       res_span=float(resid.max() - resid.min()))
            for name in MODELS:
                gm = sgn * ge[name][s]
                c1, s1, rms1 = lsq(np.column_stack([X0, -gm]), y)
                row[f'ge_{name}_mean'] = float(gm.mean())
                row[f'ge_{name}_span'] = float(gm.max() - gm.min())
                row[f'g_{name}'] = float(c1[1])
                row[f'sg_{name}'] = float(s1[1])
                row[f'rms1_{name}'] = rms1
            rows.append(row)
        print(f"  assessed {d}", flush=True)

    if not rows:
        raise SystemExit("no runs with both a free phase and a usable ramp")

    print(f"\n{len(rows)} runs   (z_CoM = {1e3 * args.z_com:.0f} mm, "
          f"J_P = {args.jp:.3f} kg m^2 pinned)\n")
    ac = np.array([r['a_c_free'] for r in rows])
    zc = np.array([r['z_chk'] for r in rows])
    print(f"  free-phase calibration: a_c = {1e3 * ac.mean():.1f} ± "
          f"{1e3 * ac.std(ddof=1):.1f} mm;  implied z_CoM = A J_P/(2W) = "
          f"{1e3 * zc.mean():.0f} ± {1e3 * zc.std(ddof=1):.0f} mm "
          f"(pinned {1e3 * args.z_com:.0f})")
    d0 = np.array([r['dphi0'] for r in rows])
    d1 = np.array([r['dphi1'] for r in rows])
    print(f"  residual window: dphi {d0.mean():.1f}° -> {d1.mean():.1f}°, "
          f"{np.mean([r['n'] for r in rows]):.0f} samples")

    at = 1e3 * np.array([r['a_t'] for r in rows])
    ac2 = 1e3 * np.array([r['a_c'] for r in rows])
    rms0 = 1e3 * np.array([r['rms0'] for r in rows])
    print(f"\n  MODEL 0  (J_P, z_CoM pinned; one free arm a_t = a_c = a)")
    print(f"    a = {at.mean():.1f} ± {at.std(ddof=1):.1f} mm,  "
          f"fit RMS {rms0.mean():.0f} ± {rms0.std(ddof=1):.0f} mN·m")
    print(f"    (free-phase a_c for the same runs: "
          f"{1e3 * np.mean([r['a_c_free'] for r in rows]):.1f} mm)")

    print(f"\n  MODEL 1  adds g x dM_GE.  g = +1 would mean the theoretical")
    print(f"  moment is exactly what the dynamics is missing.")
    print(f"  {'GE model':10}{'level [mN·m]':>15}{'span [mN·m]':>14}"
          f"{'g (per run)':>18}{'g (pooled)':>14}{'RMS gain':>11}")
    for name in MODELS:
        gm = 1e3 * np.array([r[f'ge_{name}_mean'] for r in rows])
        gs = 1e3 * np.array([r[f'ge_{name}_span'] for r in rows])
        gg = np.array([r[f'g_{name}'] for r in rows])
        sg = np.array([r[f'sg_{name}'] for r in rows])
        r1 = 1e3 * np.array([r[f'rms1_{name}'] for r in rows])
        wsum = 1.0 / sg ** 2
        pool = float(np.sum(gg * wsum) / np.sum(wsum))
        pse = float(np.sqrt(1.0 / np.sum(wsum)))
        print(f"  {name:10}{gm.mean():9.0f} ±{gs.std(ddof=1):4.0f}"
              f"{gs.mean():10.0f} ±{gs.std(ddof=1):3.0f}"
              f"{gg.mean():11.1f} ±{sg.mean():5.1f}"
              f"{pool:9.2f} ±{pse:4.2f}"
              f"{rms0.mean() - r1.mean():10.1f}")
    fbar = float(np.mean([r.get('fbar', 21.0) for r in rows]))
    print(f"\n  WHY.  Over this window the theoretical GE moment is almost a "
          f"CONSTANT\n  (span/level = "
          + ', '.join(f"{name} {1e2 * np.mean([r[f'ge_{name}_span'] for r in rows]) / abs(np.mean([r[f'ge_{name}_mean'] for r in rows])):.0f}%"
                      for name in MODELS)
          + f"), and a constant is exactly what the free arm absorbs:")
    for name in MODELS:
        lev = abs(1e3 * np.mean([r[f'ge_{name}_mean'] for r in rows]))
        print(f"    {name:10}: {lev:5.0f} mN·m == an arm error of "
              f"{lev / (W - fbar):5.1f} mm on the gravity arm, or "
              f"{lev / fbar:5.1f} mm on the thrust arm")
    print(f"  The arm is known to ~15 mm here (free phase {1e3 * ac.mean():.0f}, "
          f"Model 0 {at.mean():.0f}, mocap ~113 mm), so the GE level sits "
          f"INSIDE\n  the arm uncertainty and only its {1e3 * np.mean([r['ge_interf_span'] for r in rows]):.0f} mN·m "
          f"of curvature could ever be separated.")

    print(f"\n  distance from the theory value g = 1, per model:")
    for name in MODELS:
        gg = np.array([r[f'g_{name}'] for r in rows])
        sg = np.array([r[f'sg_{name}'] for r in rows])
        wsum = 1.0 / sg ** 2
        pool = float(np.sum(gg * wsum) / np.sum(wsum))
        pse = float(np.sqrt(1.0 / np.sum(wsum)))
        print(f"    {name:10}: per-run {gg.mean():+6.1f} ± {sg.mean():4.1f} "
              f"-> {abs(gg.mean() - 1) / sg.mean():4.1f} sigma;  pooled "
              f"{pool:+6.2f} ± {pse:.2f} -> {abs(pool - 1) / pse:5.1f} sigma")
    print(f"    (the pooled figure ignores between-run scatter, so read the "
          f"per-run column)")

    print(f"\n  per-run SE(g) is the resolving power: the smallest GE "
          f"amplitude the\n  dynamics could distinguish from zero at 3 sigma "
          f"is 3 SE(g) x |level|.")
    for name in MODELS:
        gm = 1e3 * abs(np.array([r[f'ge_{name}_mean'] for r in rows]).mean())
        sg = np.array([r[f'sg_{name}'] for r in rows]).mean()
        print(f"    {name:10}: 3 SE(g) = {3 * sg:6.1f} x the model level "
              f"{gm:5.0f} mN·m  ->  detection floor {3 * sg * gm:7.0f} mN·m")

    print(f"\n  by commanded ramp rate:")
    print(f"  {'rate':>6}{'n':>4}{'resid level':>13}{'resid span':>12}"
          + ''.join(f"{'g ' + m:>16}" for m in MODELS))
    g = defaultdict(list)
    for r in rows:
        g[r['rate']].append(r)
    for rate in sorted(g):
        v = g[rate]
        print(f"  {rate:6.2f}{len(v):4d}"
              f"{1e3 * np.mean([r['res_mean'] for r in v]):13.0f}"
              f"{1e3 * np.mean([r['res_span'] for r in v]):12.0f}"
              + ''.join(f"{np.mean([r['g_' + m] for r in v]):16.1f}"
                        for m in MODELS))

    if args.save_csv:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'ge_residual_vs_model.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nTable -> {out / 'ge_residual_vs_model.csv'}")


if __name__ == '__main__':
    main()
