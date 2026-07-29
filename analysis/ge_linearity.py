#!/usr/bin/env python3
"""
Ground-effect moment: exact model, linearity in tilt, closed-form k_GE
======================================================================
Pure-numeric (no dataset needed). Evaluates the exact per-rotor IGE model

    h_i(phi) = h*cos(phi) + s_i*sin(phi)      (interior side rises while
                                               tipping about the pivot)
    gamma-1  = R^2 / (16 h_i^2 - R^2)         (exact identity, no expansion)
    tau_GE   = sum_i a_i T_i (gamma_i - 1),   a_i = s_i = l_(y|x),i +- l_p

and reports, per axis:
  * tau_GE(0) and the multiplicative identity Delta M_GE,P = eta*(M + f*l_p)
    with eta = gamma(h)-1  (CoM- and l_p-independent ratio);
  * the tangent slope k_GE at the onset tilt and its closed level form
    k_GE ~= T_tot R^2 (3L^2 + 6 l_p^2) / (48 h^3);
  * the per-degree deviation from the tangent line over [0, phi_max] --
    the "GE is linear in phi" certificate.

Usage
-----
python analysis/ge_linearity.py                      # defaults (S550)
python analysis/ge_linearity.py --phi-max 9.4 --per-degree
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Exact GE linearity in tilt.")
    p.add_argument('--arm', type=float, default=0.265, help="arm length L [m]")
    p.add_argument('--radius', type=float, default=0.127,
                   help="propeller radius R [m] (10 in prop)")
    p.add_argument('--hub-height', type=float, default=0.315,
                   help="hub height h in the level configuration [m]")
    p.add_argument('--thrust', type=float, default=26.25,
                   help="total collective thrust during excitation [N]")
    p.add_argument('--lp-roll', type=float, default=0.140,
                   help="conservative pivot distance, roll [m]")
    p.add_argument('--lp-pitch', type=float, default=0.110,
                   help="conservative pivot distance, pitch [m]")
    p.add_argument('--phi-star', type=float, default=2.0,
                   help="onset (rest) tilt where the tangent is taken [deg]")
    p.add_argument('--phi-max', type=float, default=9.4,
                   help="largest tilt of the evaluation range [deg]")
    p.add_argument('--weight', type=float, default=31.59, help="W [N]")
    p.add_argument('--z-com', type=float, default=0.30, help="z_CoM [m]")
    p.add_argument('--per-degree', action='store_true',
                   help="print the per-degree exact-vs-linear table")
    args = p.parse_args()

    L, R, h, Ttot = args.arm, args.radius, args.hub_height, args.thrust
    T = np.full(6, Ttot / 6)
    ang = np.deg2rad(30 + 60 * np.arange(6))
    Wz = args.weight * args.z_com
    st = np.deg2rad(args.phi_star)

    g1 = R**2 / (16 * h**2 - R**2)
    print(f"eta = gamma(h)-1 = {g1*100:.3f}%   (l_p- and CoM-independent)")

    for name, lp, lat in (('roll', args.lp_roll, L * np.sin(ang)),
                          ('pitch', args.lp_pitch, L * np.cos(ang))):
        s = lat + lp
        a = s

        def tau(phi):
            hi = h * np.cos(phi) + s * np.sin(phi)
            return float(np.sum(a * T * R**2 / (16 * hi**2 - R**2)))

        eps = 1e-6
        k = (tau(st + eps) - tau(st - eps)) / (2 * eps)
        k_level = Ttot * R**2 * (3 * L**2 + 6 * lp**2) / (48 * h**3)
        print(f"\n=== {name} (l_p = {lp:.3f} m) ===")
        print(f"  tau_GE(0) = {tau(0)*1e3:6.2f} mN.m")
        print(f"  k_GE tangent @{args.phi_star:.0f} deg = {k*1e3:+7.1f} mN.m/rad "
              f"({100*abs(k)/Wz:.2f}% of Wz)")
        print(f"  k_GE closed level form        = {k_level*1e3:7.1f} mN.m/rad")
        for pmax in (5.0, args.phi_max):
            ps = np.linspace(0, np.deg2rad(pmax), 601)
            tv = np.array([tau(x) for x in ps])
            lin = tau(st) + k * (ps - st)
            dev = float(np.max(np.abs(tv - lin)))
            rng = float(tv[-1] - tv[0])
            print(f"  [0,{pmax:4.1f} deg]: GE change {rng*1e3:6.2f} mN.m, "
                  f"max dev from tangent {dev*1e3:5.2f} mN.m")
        if args.per_degree:
            print(f"  {'phi[deg]':>8} {'exact':>8} {'linear':>8} {'diff':>7}  [mN.m]")
            for pd in range(0, int(np.ceil(args.phi_max)) + 1):
                x = np.deg2rad(pd)
                ex, li = tau(x) * 1e3, (tau(st) + k * (x - st)) * 1e3
                print(f"  {pd:8d} {ex:8.2f} {li:8.2f} {ex-li:+7.2f}")


if __name__ == '__main__':
    main()
