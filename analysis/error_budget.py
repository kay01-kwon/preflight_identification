#!/usr/bin/env python3
"""End-to-end error budget: from the residual forcing rho to millimetres.

Sections fid-ideal / fid-transfer bound every perturbation channel in
moment units (mN.m).  This script closes the last link and expresses
each channel in the currency of the deliverable, so the reader never
has to translate: how far does each residual move the identified CoM
offset?

The chain is closed in three exact steps per run.

1. FORCING.  Each perturbation is evaluated along the *measured*
   trajectory over the fit window [onset, window end] and projected out
   of the estimator span {1, tau, dphi} -- an algebraic identity, these
   being the estimator's exact per-run degrees of freedom.  What
   survives is rho(tau).

2. PROPAGATION.  With zero initial conditions at the onset the
   deviation dynamics J_P e'' - D e = rho has no homogeneous part
   (h_phi(0) = 0 kills the Leibniz boundary term in e'), so the rate
   deviation is the Duhamel integral alone,

       domega(tau) = (1/J_P) \int_0^tau cosh(C2 (tau - s)) rho(s) ds,

   with J_P = W z_CoM / C2^2 = 1 / (K C2^2) from the calibrated
   constants -- no independent inertia estimate enters.

3. TRANSLATION.  The onset is the argmin over t0 of the two-segment
   residual.  Perturbing the data by domega and linearising the
   stationarity condition (Gauss-Newton) about the unperturbed optimum,
   with d omega_hat / d t0 = -C1 C2 sinh(C2 tau),

       dt0        = -<domega, sinh(C2 tau)> / (C1 C2 ||sinh(C2 tau)||^2)
       dM_crit    = Mdot * dt0 = -<domega, sinh> / (K C2 ||sinh||^2)

   The ramp rate CANCELS (C1 = K Mdot): the exchange rate from rho to
   critical moment is a constant of the calibration, not of the run.

   The deliverable then follows the pipeline exactly:
       M_ff  = sign * 0.5 * (M_pos + M_neg),   sign = -1 for pitch
       offset = M_ff / W
   so an error that enters both tip directions with the same signed
   moment cancels in the pivot-free average, and only the common part
   in |M| survives.

A fourth number is reported for the C2 question specifically: since
d ln(cosh x - 1) / d ln x = x coth(x/2) >= 2 for all x > 0, a relative
rate deviation eps can masquerade as at most an eps/2 relative change
of the exponent.  This bounds how much of rho could be mistaken for a
mis-calibrated C2 if C2 were re-fitted per run (it is not).

Usage
-----
python analysis/error_budget.py DataSet/exp [--output-dir DIR]
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
    extract_piecewise_batch, detect_excitation_window, commanded_ramp_rate,
    estimate_rig_constants, prepare_signals, MOMENT_CAP)

# geometry / aero constants, identical to ge_trajectory.py and
# small_angle_trajectory.py so the components stay comparable
W_DEFAULT = 31.59          # N
Z_COM = 0.30               # m -- deliberately ABOVE the CAD value
# CAD gives z_CoM = 0.261 m; 0.30 is kept here on purpose.  The
# gravity channel of rho is the out-of-span part of W z sin(phi),
# whose curvature grows with z, so overstating z overstates the
# bound.  This is a budget, not an estimate -- do not 'correct'
# it to 0.261 without also relabelling what the table reports.
LP = dict(x=0.140, y=0.110)
OFFSET_MARGIN = 0.020      # m, worst |CoM offset| added to the gravity arm
R_ROTOR = 0.127            # m
H_HUB = 0.315              # m
L_ARM = 0.265              # m
C_T = 1.3175e-7
BETA_M = 0.0345            # 1/rad, interference model (analysis/ge_linearity)

CHANNELS = ('gravity', 'ge', 'bilinear', 'ramp')


def out_of_span(A, y):
    """Remove the estimator's per-run degrees of freedom {1, tau, dphi}."""
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ c


def duhamel(tau, rho, c2, j_p):
    """domega(tau_i) = (1/J_P) int_0^tau_i cosh(C2 (tau_i - s)) rho(s) ds.

    Trapezoidal quadrature of the exact kernel -- no series truncation.
    """
    n = len(tau)
    out = np.zeros(n)
    for i in range(1, n):
        s = tau[:i + 1]
        integrand = np.cosh(np.clip(c2 * (tau[i] - s), 0, 30)) * rho[:i + 1]
        out[i] = np.trapezoid(integrand, s) / j_p
    return out


def _rot_matrices(qw, qx, qy, qz):
    """Stack of body-to-world rotation matrices from a quaternion series."""
    return np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
                  2 * (qx * qz + qw * qy)], axis=-1),
        np.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
                  2 * (qy * qz - qw * qx)], axis=-1),
        np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                  1 - 2 * (qx * qx + qy * qy)], axis=-1),
    ], axis=-2)


def ge_moment(bag, sig, axis, n, pos, q_rest=None):
    """Pivot ground-effect moment along the trajectory (interference model).

    Image superposition over rotor-centre distances at each rotor's own
    tilted height; no empirical constant.  Mirrors the 'interf' branch of
    analysis/ge_trajectory.py.

    Ground effect is set by rotor height above the GROUND, so the heights
    must be referred to the resting plane, not to true vertical: the pad
    is not level (roll +0.5 deg, pitch -1.5 deg, see
    analysis/attitude_reference.py).  Pass ``q_rest``, the quaternion
    measured before the ramp, and the heights are taken along the ground
    normal n = R(q_rest) e_z instead of along world z.  Omitting it keeps
    the original world-vertical convention.

    (Gravity is the other way round -- it acts along true vertical and so
    takes the absolute attitude.  The two references are not
    interchangeable.)
    """
    ang = np.deg2rad(30 + 60 * np.arange(6))
    lx, ly = L_ARM * np.cos(ang), L_ARM * np.sin(ang)
    lp = LP[axis]
    if axis == 'x':
        px, py = lx, ly + (lp if pos else -lp)
        arm = py
    else:
        px, py = lx + (-lp if pos else lp), ly
        arm = -px
    p_p = np.vstack([px, py, np.full(6, H_HUB)])

    t = sig['t']
    t0 = bag.odom.t[0]
    t6 = C_T * bag.rpm.rpm.astype(np.float64) ** 2
    thrust = np.vstack([np.interp(t, bag.rpm.t - t0, t6[:, j])
                        for j in range(6)]).T

    qw, qx, qy, qz = bag.odom.quaternion.T
    if q_rest is None:
        # height along world z: the third row of R(t)
        nx = 2 * (qx * qz - qw * qy)
        ny = 2 * (qy * qz + qw * qx)
        nz = 1 - 2 * (qx * qx + qy * qy)
        h_it = (np.outer(nx[:n], p_p[0]) + np.outer(ny[:n], p_p[1])
                + np.outer(nz[:n], p_p[2]))
    else:
        # height along the ground normal: h = (R(q_rest) e_z)^T R(t) p
        rw, rx, ry, rz = q_rest
        gn = np.array([2 * (rx * rz + rw * ry),
                       2 * (ry * rz - rw * rx),
                       1 - 2 * (rx * rx + ry * ry)])          # R0[:, 2]
        rot = _rot_matrices(qw[:n], qx[:n], qy[:n], qz[:n])   # (n, 3, 3)
        h_it = np.einsum('k,nkj,jm->nm', gn, rot, p_p)
    if np.any(h_it <= R_ROTOR / 4 + 0.01):
        return None
    d2 = ((p_p[0][:, None] - p_p[0][None, :]) ** 2
          + (p_p[1][:, None] - p_p[1][None, :]) ** 2)
    zz = h_it[:, :, None] + h_it[:, None, :]
    srot = (R_ROTOR ** 2 / 4) * np.sum(zz / (d2 + zz ** 2) ** 1.5, axis=2)
    gain = 1.0 / (1.0 - srot) - 1.0
    return (gain * thrust[:n]) @ arm


def check_weight():
    """Verify the reduction dM_crit = -<rho>_w with w >= 0 and int w = 1.

    Exchanging the order of integration in the translation formula turns
    the whole propagation into a weighted average of the forcing,

        dM_crit = -int_0^T rho(s) w(s) ds,
        w(s) = int_s^T sinh(C2 tau) cosh(C2 (tau-s)) dtau
               / (J_P K C2 ||sinh||^2).

    The weight is non-negative (both factors are positive on s < tau)
    and normalized -- int w = 1 is exactly the constant-forcing identity
    dM_crit = -rho_0.  Hence |dM_crit| <= max|rho| unconditionally, with
    no assumption on the shape of rho.  This checks both properties on
    the discrete operators the budget actually uses, and reports how far
    a sign-changing forcing falls below the bound.
    """
    print("reduction check:  dM_crit = -<rho>_w,   w >= 0,   int w = 1\n")
    print(f"{'C2':>5} {'K':>7} {'T[s]':>6} {'N':>4} | {'dM(rho=1)':>11} "
          f"{'int w':>9} {'min w':>10} | {'<rho>_w/max|rho|':>17}")
    for c2, k, t_end, n in [(8.0, 0.060, 0.30, 31), (6.125, 0.110, 0.45, 46),
                            (3.5, 0.300, 0.90, 91), (4.5, 0.520, 0.70, 71),
                            (6.625, 0.180, 0.55, 56)]:
        tau = np.linspace(0.0, t_end, n)
        j_p = 1.0 / (k * c2 ** 2)
        sh = np.sinh(np.clip(c2 * tau, 0, 30))
        den = k * c2 * float(sh @ sh)

        def d_mcrit(rho):
            return -float(duhamel(tau, rho, c2, j_p) @ sh) / den

        one = d_mcrit(np.ones_like(tau))
        eye = np.eye(n)
        w = -np.array([d_mcrit(eye[i]) for i in range(n)]) / (tau[1] - tau[0])
        rho = np.sin(6 * np.pi * tau / t_end) * (tau / t_end) ** 4
        ratio = abs(d_mcrit(rho)) / max(np.abs(rho).max(), 1e-12)
        print(f"{c2:5.2f} {k:7.3f} {t_end:6.2f} {n:4d} | {one:11.6f} "
              f"{w.sum() * (tau[1] - tau[0]):9.5f} {w.min():10.2e} | "
              f"{ratio:17.4f}")
    print("\ndM(rho=1) = -1 confirms int w = 1; min w >= 0 confirms the weight")
    print("is non-negative, so |dM_crit| <= max|rho| holds unconditionally.")


def main():
    p = argparse.ArgumentParser(
        description="rho -> onset -> M_crit -> CoM offset error budget.")
    p.add_argument('--check', action='store_true',
                   help="verify the weighted-average reduction and exit "
                        "(no dataset needed)")
    p.add_argument('root', nargs='?', help="dataset root (e.g. DataSet/exp)")
    p.add_argument('--weight', type=float, default=W_DEFAULT)
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    if args.check:
        check_weight()
        return
    if not args.root:
        p.error("dataset root is required unless --check is given")

    weight = args.weight
    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root
    dirs = sorted(d for d in root.glob('case_*/M[xy]') if d.is_dir())
    if not dirs:
        raise SystemExit(f"no datasets under {root}")

    rows = []
    for d in dirs:
        axis = 'x' if d.name == 'Mx' else 'y'
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            c2, k_gain = estimate_rig_constants(bags, axis)
            crits, _ = extract_piecewise_batch(bags, axis, cosh_c2=c2,
                                               ramp_gain=k_gain)
        wz = 1.0 / k_gain
        j_p = wz / c2 ** 2
        arm = LP[axis] + OFFSET_MARGIN
        bag_by = {b.name: b for b in bags}
        print(f"  {d}: C2={c2:.3f} rad/s, K={k_gain:.4f} "
              f"(W z_CoM={wz:.2f} N.m, J_P={j_p:.3f} kg.m^2)")

        for crit in crits:
            bag = bag_by[crit.bag_name]
            sig = prepare_signals(bag, axis)
            i0, i1 = detect_excitation_window(crit.moment,
                                              moment_cap=MOMENT_CAP.get(axis))
            j = crit.onset_idx
            if i1 - j < 10:
                continue
            roll, pitch = math_tools.quaternion_to_euler_vectorized(
                bag.odom.quaternion)
            phi_all = roll if axis == 'x' else pitch
            n = min(len(phi_all), len(crit.t))
            i1 = min(i1, n - 1)
            if i1 - j < 10:
                continue
            sl = slice(j, i1 + 1)

            tau = crit.t[sl] - crit.t[j]
            phi = phi_all[sl]
            dphi = phi - phi[0]
            mom = crit.moment[sl]
            m_dot = float(np.polyfit(tau, mom, 1)[0])
            if abs(m_dot) < 1e-6:
                continue
            basis = np.vstack([np.ones_like(tau), tau, dphi]).T

            rho = {}
            # (a) gravity curvature: the exact pivot gravity moment minus
            #     whatever the estimator can absorb
            rho['gravity'] = out_of_span(
                basis, -weight * arm * np.cos(phi) + weight * Z_COM * np.sin(phi))
            # (b) ground effect, interference model, exact along the trajectory
            ge = ge_moment(bag, sig, axis, n, crit.bag_name.startswith('pos'))
            rho['ge'] = (out_of_span(basis, ge[sl]) if ge is not None
                         else np.zeros_like(tau))
            # (c) bilinear GE term beta_M * (Mdot tau) * dphi
            rho['bilinear'] = out_of_span(basis,
                                          BETA_M * (mom - mom[0]) * dphi)
            # (d) ramp execution nonlinearity (incoherent, noise-like)
            rho['ramp'] = out_of_span(basis, mom - np.polyval(
                np.polyfit(tau, mom, 1), tau))

            sh = np.sinh(np.clip(c2 * tau, 0, 30))
            denom = k_gain * c2 * float(sh @ sh)
            # nominal rate at the window edge, the normalizer used by the
            # empirical fit metric
            nom_end = abs(k_gain * m_dot
                          * (np.cosh(np.clip(c2 * tau[-1], 0, 30)) - 1.0))

            rec = dict(case=d.parent.name, axis=d.name, bag=crit.bag_name,
                       dir='pos' if crit.bag_name.startswith('pos') else 'neg',
                       rate=commanded_ramp_rate(crit.bag_name) or np.nan,
                       c2=c2, k=k_gain, m_dot=m_dot,
                       n_post=int(i1 - j + 1),
                       c2_tau_end=float(c2 * tau[-1]),
                       dphi_end_deg=float(np.rad2deg(dphi[-1])))
            for name in CHANNELS:
                dw = duhamel(tau, rho[name], c2, j_p)
                d_mcrit = -float(dw @ sh) / denom
                rec[f'rho_{name}_mNm'] = 1e3 * float(np.max(np.abs(rho[name])))
                rec[f'dw_{name}_pct'] = (100.0 * float(np.max(np.abs(dw)))
                                         / nom_end if nom_end > 0 else np.nan)
                rec[f'dM_{name}_mNm'] = 1e3 * d_mcrit
            rec['dM_total_mNm'] = sum(rec[f'dM_{c}_mNm'] for c in CHANNELS)
            rows.append(rec)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'error_budget_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- per-run summary -------------------------------------------
    print(f"\n{len(rows)} gate-passing runs;  "
          f"C2*tau_end in [{min(r['c2_tau_end'] for r in rows):.2f}, "
          f"{max(r['c2_tau_end'] for r in rows):.2f}]")
    hdr = (f"\n{'channel':10} | {'|rho| max [mN.m]':>22} | "
           f"{'shape dev [% of nom]':>22} | {'dM_crit [mN.m]':>22}")
    print(hdr)
    print(f"{'':10} | {'median':>10} {'max':>11} | {'median':>10} {'max':>11}"
          f" | {'median':>10} {'max':>11}")
    print('-' * len(hdr))
    for name in CHANNELS:
        rr = np.array([r[f'rho_{name}_mNm'] for r in rows])
        dw = np.array([r[f'dw_{name}_pct'] for r in rows])
        dm = np.abs([r[f'dM_{name}_mNm'] for r in rows])
        print(f"{name:10} | {np.median(rr):10.2f} {rr.max():11.2f} | "
              f"{np.median(dw):10.3f} {dw.max():11.3f} | "
              f"{np.median(dm):10.2f} {dm.max():11.2f}")

    # equivalent exponent perturbation: eps/2 with eps the shape deviation
    eps = np.array([sum(r[f'dw_{c}_pct'] for c in CHANNELS) for r in rows])
    print(f"\nequivalent |dC2/C2| <= eps/2 : median {np.median(eps) / 2:.3f}%, "
          f"max {eps.max() / 2:.3f}%   (x coth(x/2) >= 2)")

    # ---- deliverable: per-direction averages, then the pivot-free pair
    grp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp[(r['case'], r['axis'])][r['dir']].append(r)
    print(f"\n{'case':8} {'ax':3} | "
          + ' '.join(f"{c[:4]:>17}" for c in CHANNELS)
          + f" {'total':>17}")
    print(f"{'':8} {'':3} | "
          + ' '.join(f"{'dM+ / dM-':>17}" for _ in CHANNELS)
          + f" {'dMff [um.m]':>17}")
    ff_rows = []
    for key in sorted(grp):
        case, ax = key
        sign = -1.0 if ax == 'My' else 1.0
        line = f"{case:8} {ax:3} |"
        rec = dict(case=case, axis=ax)
        for name in CHANNELS + ('total',):
            col = f'dM_{name}_mNm'
            dp = float(np.mean([r[col] for r in grp[key]['pos']])) \
                if grp[key]['pos'] else np.nan
            dn = float(np.mean([r[col] for r in grp[key]['neg']])) \
                if grp[key]['neg'] else np.nan
            d_ff = sign * 0.5 * (dp + dn)
            rec[f'{name}_pos'] = dp
            rec[f'{name}_neg'] = dn
            rec[f'{name}_ff'] = d_ff
            rec[f'{name}_mm'] = d_ff / weight        # mN.m / N = um -> mm*1e-3
            if name != 'total':
                line += f" {dp:+7.1f}/{dn:+6.1f}"
            else:
                line += f"   {d_ff:+8.2f}"
        ff_rows.append(rec)
        print(line)

    with open(out / 'error_budget_groups.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ff_rows[0].keys()))
        w.writeheader()
        w.writerows(ff_rows)

    print(f"\n{'channel':10} | {'|dM_crit| per dir':>20} | "
          f"{'|dM_ff| after pairing':>22} | {'CoM offset':>18}")
    print(f"{'':10} | {'med / max [mN.m]':>20} | {'med / max [mN.m]':>22} | "
          f"{'med / max [mm]':>18}")
    for name in CHANNELS + ('total',):
        per = np.abs(np.concatenate(
            [[r[f'{name}_pos'] for r in ff_rows],
             [r[f'{name}_neg'] for r in ff_rows]]))
        ff = np.abs([r[f'{name}_ff'] for r in ff_rows])
        mm = ff / weight
        print(f"{name:10} | {np.median(per):8.2f} / {per.max():8.2f} | "
              f"{np.median(ff):10.2f} / {ff.max():9.2f} | "
              f"{np.median(mm):7.3f} / {mm.max():7.3f}")
    print(f"\nCancellation factor (per-direction -> pivot-free), total: "
          f"{np.median(np.abs(np.concatenate([[r['total_pos'] for r in ff_rows], [r['total_neg'] for r in ff_rows]]))) / max(np.median(np.abs([r['total_ff'] for r in ff_rows])), 1e-9):.1f}x")
    print(f"Tables -> {out}")


if __name__ == '__main__':
    main()
