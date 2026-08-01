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
    p.add_argument('--model', choices=['single', 'garofano'], default='single')
    p.add_argument('--arm', type=float, default=0.265, help="arm length L [m]")
    p.add_argument('--radius', type=float, default=0.127)
    p.add_argument('--hub-height', type=float, default=0.315)
    p.add_argument('--thrust', type=float, default=26.25,
                   help="total collective thrust f [N]")
    p.add_argument('--lp-roll', type=float, default=0.140)
    p.add_argument('--lp-pitch', type=float, default=0.110)
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

    def channels(phi, lat, lp):
        """Return (a, b) of Delta M_GE,P = a + b*Mx at tilt phi."""
        s = lat + lp
        z = h * np.cos(phi) + s * np.sin(phi)      # interior side rises
        if args.model == 'single':
            g1 = R**2 / (16 * z**2 - R**2)
            gbar = float(np.mean(g1))              # uniform T weighting
            a_rot = gbar * f * lp + np.sum(s * (g1 - gbar)) * f / 6
            return a_rot, gbar
        srot = srot_matrix(z, pxy, R)
        g1 = 1.0 / (1.0 - args.k_cal * srot) - 1.0
        gbar = float(np.mean(g1))
        zc = h * np.cos(phi) + lp * np.sin(phi)
        sf = 2 * R**2 * args.jk * zc / (rd**2 + 4 * zc**2) ** 1.5
        ksb = args.k_cal * float(np.mean(srot))
        dFb = (1.0 / (1.0 - ksb - sf) - 1.0 / (1.0 - ksb)) * f
        a = gbar * f * lp + np.sum(s * (g1 - gbar)) * f / 6 + dFb * lp
        return a, gbar

    print(f"model = {args.model}"
          + ("" if args.model == 'single' else
             f"  (k={args.k_cal}, J_k={args.jk}, r_d={rd:.3f} m — empirical "
               f"constants transferred from Garofano-Soldado et al. 2024)"))
    for name, lp, lat, Mx in (('roll', args.lp_roll, LY, args.mx),
                              ('pitch', args.lp_pitch, LX, args.mx)):
        a0, b0 = channels(0.0, lat, lp)
        eps = 1e-6
        ap, bp_ = channels(st + eps, lat, lp)
        am, bm = channels(st - eps, lat, lp)
        ka = (ap - am) / (2 * eps)
        kb = (bp_ - bm) / (2 * eps)
        k_tot = ka + kb * Mx
        print(f"\n=== {name} (l_p={lp:.3f} m, Mx={Mx:g} N.m) ===")
        print(f"  a(0) = {a0*1e3:7.1f} mN.m   a/(f*lp) = {100*a0/(f*lp):6.2f} %")
        print(f"  b(0) = {100*b0:6.3f} %      total(0) = {(a0+b0*Mx)*1e3:7.1f} mN.m")
        print(f"  k_GE tangent @{args.phi_star:g} deg = {k_tot*1e3:+7.1f} mN.m/rad "
              f"({100*abs(k_tot)/Wz:.2f}% of Wz)")
        for pmax in (5.0, args.phi_max):
            ps = np.linspace(0, np.deg2rad(pmax), 301)
            tot = np.array([channels(x, lat, lp) for x in ps])
            tv = tot[:, 0] + tot[:, 1] * Mx
            a_st, b_st = channels(st, lat, lp)
            lin = (a_st + b_st * Mx) + k_tot * (ps - st)
            dev = float(np.max(np.abs(tv - lin)))
            print(f"  [0,{pmax:4.1f} deg]: GE change {(tv[-1]-tv[0])*1e3:7.2f} "
                  f"mN.m, max dev from tangent {dev*1e3:5.2f} mN.m")
        if args.per_degree:
            print(f"  {'phi':>4} {'a/(f*lp)%':>10} {'b%':>7} {'total[mN.m]':>12}")
            for pd in range(0, int(np.ceil(args.phi_max)) + 1):
                x = np.deg2rad(pd)
                a, b = channels(x, lat, lp)
                print(f"  {pd:4d} {100*a/(f*lp):10.2f} {100*b:7.3f} "
                      f"{(a+b*Mx)*1e3:12.1f}")


if __name__ == '__main__':
    main()
