#!/usr/bin/env python3
"""
Ground-effect moment: exact models, linearity in tilt, closed-form k_GE
=======================================================================
Pure-numeric (no dataset needed). Two models, forming a bracket:

--model single (lower bound): per-rotor superposition of the classical
  single-rotor Cheeseman--Bennett result at the exact tilted heights
      h_i(phi) = h*cos(phi) + s_i*sin(phi)
      gamma-1  = R^2 / (16 h_i^2 - R^2)
  Neglects rotor--rotor interference.

--model garofano (upper bound): attitude-dependent adaptation of the
  co-planar hexarotor model of Garofano-Soldado et al. (IEEE RA-L 9(2),
  2024; static test-bench, i.e. the same quasi-static regime as the
  contact-phase excitation). Per-rotor interference sum evaluated at the
  tilted rotor heights with exact pairwise distances,
      s_rot,j = (R^2/4) sum_n (z_j+z_n) / [d_jn^2 + (z_j+z_n)^2]^{3/2}
      gamma_j = [1 - k*s_rot,j]^{-1}
  plus the fountain-driven body lift of their Eq. (8),
      s_f = 2 R^2 J_k z_c / (r_d^2 + 4 z_c^2)^{3/2},  r_d = 2d - c
      dF_body = [1/(1-k*sbar-s_f) - 1/(1-k*sbar)] * f ,
  with k calibrated so the level interference sum reproduces their
  Eq. (9) for this geometry (k = 0.9647). Empirical constants (J_k) are
  transferred from the authors' vehicle: an informed estimate, not truth.

Pivot-moment decomposition reported per angle:
  Delta M_GE,P = a(phi) + b(phi) * M_x
  a = thrust channel (rotor-common + body lift), arm exactly l_p
      -> antisymmetric between tip directions, cancels in M_ff
  b = gamma_bar - 1, the moment-proportional coefficient
      -> ramp part absorbed by K; onset part = -b relative bias on M_ff

Usage
-----
python analysis/ge_linearity.py                      # single-rotor model
python analysis/ge_linearity.py --model garofano --per-degree
"""

import argparse

import numpy as np


def hexagon(arm):
    ang = np.deg2rad(30 + 60 * np.arange(6))
    return arm * np.cos(ang), arm * np.sin(ang)


def srot_matrix(z, pxy, R):
    """Interference sum per rotor for heights z (6,) and positions pxy (6,2)."""
    d2 = np.sum((pxy[:, None, :] - pxy[None, :, :]) ** 2, axis=2)
    Z = z[:, None] + z[None, :]
    return (R**2 / 4) * np.sum(Z / (d2 + Z**2) ** 1.5, axis=1)


def main():
    p = argparse.ArgumentParser(description="Exact GE linearity in tilt.")
    p.add_argument('--model', choices=['single', 'interf', 'garofano'],
                   default='single')
    p.add_argument('--arm', type=float, default=0.265, help="arm length L [m]")
    p.add_argument('--radius', type=float, default=0.127)
    p.add_argument('--hub-height', type=float, default=0.315)
    p.add_argument('--thrust', type=float, default=26.25,
                   help="total collective thrust f [N]")
    p.add_argument('--lp-roll', type=float, default=0.130)
    p.add_argument('--lp-pitch', type=float, default=0.100)
    p.add_argument('--frame-width', type=float, default=0.22,
                   help="central frame width c [m] (garofano fountain)")
    p.add_argument('--jk', type=float, default=2.2,
                   help="fountain constant J_k (garofano, transferred)")
    p.add_argument('--k-cal', type=float, default=0.9647,
                   help="level calibration of the interference sum to "
                        "Garofano-Soldado Eq. (9) for this geometry")
    p.add_argument('--mx', type=float, default=2.3,
                   help="applied moment for the b-channel total [N.m]")
    p.add_argument('--phi-star', type=float, default=2.0)
    p.add_argument('--phi-max', type=float, default=9.4)
    p.add_argument('--weight', type=float, default=31.59)
    p.add_argument('--z-com', type=float, default=0.30)
    p.add_argument('--per-degree', action='store_true')
    args = p.parse_args()

    L, R, h, f = args.arm, args.radius, args.hub_height, args.thrust
    LX, LY = hexagon(L)
    pxy = np.column_stack([LX, LY])
    Wz = args.weight * args.z_com
    st = np.deg2rad(args.phi_star)
    rd = 2 * args.arm - args.frame_width

    def k_of_z(z):
        """Matching constant k(z): Garofano-Soldado Eq. (9) rotor sum over
        the level rotor-centre sum, both at clearance z."""
        d = args.arm
        s9 = (R**2 * z / (d**2/4 + 4*z**2)**1.5
              + (R**2/2) * z / (21*d**2/4 + 4*z**2)**1.5
              + R**2 * z / (13*d**2/4 + 4*z**2)**1.5
              + (R**2/2) * z / (7*d**2/4 + 4*z**2)**1.5)
        slevel = srot_matrix(np.full(6, z), pxy, R)[0]
        return s9 / slevel

    def channels(phi, s_h, lp, arm_c, tm):
        """(a, b) of Delta M_GE = a + b*M_cmd at tilt phi.

        s_h   : height coefficients, z_i = h cos(phi) + s_h,i sin(phi)
                (roll: l_y+l_p; pitch: l_p-l_x — interior side rises)
        arm_c : body-centre moment arms b_i (l_y or -l_x) — the fountain
                force acts at the centre and enters only via the
                parallel-axis transfer F*l_p.
        tm    : allocation column dT_i/dM_cmd.
        """
        z = h * np.cos(phi) + s_h * np.sin(phi)
        if args.model == 'single':
            g1 = R**2 / (16 * z**2 - R**2)
            a = np.sum(arm_c * g1) * f / 6 + lp * np.sum(g1) * f / 6
            b = float(np.sum((arm_c + lp) * g1 * tm))
            return a, b
        if args.model == 'interf':
            # pure image superposition over rotor-centre distances: mutual
            # interference included, NO empirical constants (no k matching,
            # no fountain/body term) — Sanchez-style neighbour sum.
            srot = srot_matrix(z, pxy, R)
            g = 1.0 / (1.0 - srot) - 1.0
            a = (np.sum(arm_c * g) + lp * np.sum(g)) * f / 6
            b = float(np.sum((arm_c + lp) * g * tm))
            return a, b
        zc = h * np.cos(phi) + lp * np.sin(phi)
        k = k_of_z(zc)                             # re-evaluated per attitude
        srot = srot_matrix(z, pxy, R)
        g_rot = 1.0 / (1.0 - k * srot) - 1.0
        sf = 2 * R**2 * args.jk * zc / (rd**2 + 4 * zc**2) ** 1.5
        g_full = 1.0 / (1.0 - k * srot - sf) - 1.0
        # Eq. (2) of the derivation: interference through the centre arms,
        # fountain-inclusive gain only in the total vertical force
        a = (np.sum(arm_c * g_rot) + lp * np.sum(g_full)) * f / 6
        b = float(np.sum((arm_c * g_rot + lp * g_full) * tm))
        return a, b

    print(f"model = {args.model}"
          + ("" if args.model != 'garofano' else
             f"  (J_k={args.jk}, r_d={rd:.3f} m — J_k transferred from "
               f"Garofano-Soldado et al. 2024; k(z_c) matched per attitude)"))
    for name, lp, s_h, arm_c, Mx in (
            ('roll', args.lp_roll, LY + args.lp_roll, LY, args.mx),
            ('pitch', args.lp_pitch, args.lp_pitch - LX, -LX, args.mx)):
        tm = arm_c / np.sum(arm_c**2)              # allocation column
        ch = lambda x: channels(x, s_h, lp, arm_c, tm)
        a0, b0 = ch(0.0)
        eps = 1e-6
        (ap, bp_), (am, bm) = ch(st + eps), ch(st - eps)
        k_tot = (ap - am) / (2 * eps) + (bp_ - bm) / (2 * eps) * Mx
        print(f"\n=== {name} (l_p={lp:.3f} m, Mx={Mx:g} N.m) ===")
        print(f"  a(0) = {a0*1e3:7.1f} mN.m   a/(f*lp) = {100*a0/(f*lp):6.2f} %")
        print(f"  b(0) = {100*b0:6.3f} %      total(0) = {(a0+b0*Mx)*1e3:7.1f} mN.m")
        print(f"  k_GE tangent @{args.phi_star:g} deg = {k_tot*1e3:+7.1f} mN.m/rad "
              f"({100*abs(k_tot)/Wz:.2f}% of Wz)")
        for pmax in (5.0, args.phi_max):
            ps = np.linspace(0, np.deg2rad(pmax), 301)
            tot = np.array([ch(x) for x in ps])
            tv = tot[:, 0] + tot[:, 1] * Mx
            a_st, b_st = ch(st)
            lin = (a_st + b_st * Mx) + k_tot * (ps - st)
            dev = float(np.max(np.abs(tv - lin)))
            print(f"  [0,{pmax:4.1f} deg]: GE change {(tv[-1]-tv[0])*1e3:7.2f} "
                  f"mN.m, max dev from tangent {dev*1e3:5.2f} mN.m")
        if args.per_degree:
            print(f"  {'phi':>4} {'a/(f*lp)%':>10} {'b%':>7} {'total[mN.m]':>12}")
            for pd in range(0, int(np.ceil(args.phi_max)) + 1):
                a, b = ch(np.deg2rad(pd))
                print(f"  {pd:4d} {100*a/(f*lp):10.2f} {100*b:7.3f} "
                      f"{(a+b*Mx)*1e3:12.1f}")


if __name__ == '__main__':
    main()
