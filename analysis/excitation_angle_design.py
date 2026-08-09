#!/usr/bin/env python3
"""
Excitation-angle / ramp-rate design tables (theoretical, no dataset)
====================================================================
For every candidate ramp rate and excursion-angle cap, propagates the
a-priori coherent remainder through the Duhamel integral along the
NOMINAL cosh trajectory and prints four tables:

  1. systematic shape deviation |d_omega/omega| at the window edge [%]
  2. maximum |d_omega| over the window (attained at the edge) [mrad/s]
  3. post-onset sample count N_post
  4. signal span at the edge and the noise-limited NRMSE floor

plus the cap angle at which the systematic deviation crosses the
--criterion level, per rate (the binding rate is the slowest one).

Model
-----
Nominal solution with onset conditions omega(0) = alpha(0) = 0:

    omega(tau) = C1*(cosh(C2*tau) - 1),          C1 = K*Mdot
    dphi(tau)  = (C1/C2)*(sinh(C2*tau) - C2*tau)

The window ends where dphi reaches the cap (monotone bijection):
sinh(x) - x = C2*cap/C1, x = C2*tau_end. The coherent remainder about
the onset operating point is the second-order Lagrange remainder of the
gravity moment, with the ground-effect curvature folded in as a
fractional margin:

    rho(s)  = 1/2 * G2 * dphi(s)^2,
    G2      = W*(l_p + |lambda_off|) * (1 + ge_margin)
              [ |G''(xi)| = |W*arm*cos(xi) - W*z*sin(xi)| <= W*arm:
                the sine term SUBTRACTS about the onset point and is
                dropped conservatively; ge_margin adds the exact-model
                ground-effect curvature (~5% of the gravity curvature) ]

and its propagated deviation is the exact convolution with the
unstable (anti-restoring) kernel (Duhamel):

    d_omega(tau_end) = (1/J) * int_0^tau_end cosh(C2*(tau_end-s)) rho(s) ds,
    J = W*z_CoM / C2^2 .

Incoherent components (ramp-execution / gyro noise) are NOT part of rho:
they set the noise floor of table 4 instead of a systematic deformation.

Usage
-----
python analysis/excitation_angle_design.py
python analysis/excitation_angle_design.py --criterion 1.0 --caps 2 3 4 5 6 7 8 9 10
"""

import argparse

import numpy as np

# numpy 2.0 renamed trapz to trapezoid and dropped the old name, so
# neither spelling works on both.  Bind whichever this numpy has.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

from scipy.optimize import brentq


def main():
    p = argparse.ArgumentParser(
        description="Theoretical shape-deviation tables vs rate and angle cap.")
    p.add_argument('--c2', type=float, default=5.0,
                   help="calibrated instability rate C2 [rad/s]")
    p.add_argument('--k', type=float, default=0.19,
                   help="calibrated ramp gain K [1/(N.m)]")
    p.add_argument('--weight', type=float, default=31.59, help="W [N]")
    p.add_argument('--z-com', type=float, default=0.30, help="z_CoM [m]")
    p.add_argument('--arm', type=float, default=0.160,
                   help="conservative gravity arm l_p+|off| [m] "
                        "(0.140+0.020 roll)")
    p.add_argument('--ge-margin', type=float, default=0.05,
                   help="GE curvature as a fraction of the gravity curvature")
    p.add_argument('--fs', type=float, default=130.0,
                   help="odometry sample rate [Hz]")
    p.add_argument('--gyro-noise', type=float, default=0.035,
                   help="in-motion gyro noise RMS [rad/s]")
    p.add_argument('--rates', type=float, nargs='+',
                   default=[0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20])
    p.add_argument('--caps', type=float, nargs='+',
                   default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                   help="excursion caps [deg]")
    p.add_argument('--criterion', type=float, default=1.0,
                   help="systematic-deviation criterion [%%] for the "
                        "per-rate admissible cap")
    args = p.parse_args()

    C2, K, W = args.c2, args.k, args.weight
    J = W * args.z_com / C2**2
    G2 = W * args.arm * (1.0 + args.ge_margin)

    def propagate(Md, cap_deg):
        """(tau_end, |d_omega|, omega_nom, rel%) at the window edge."""
        C1 = K * Md
        dm = np.deg2rad(cap_deg)
        te = brentq(lambda T: C1 * (np.sinh(C2 * T) / C2 - T) - dm,
                    1e-5, 20.0)
        s = np.linspace(0, te, 3000)
        dphi = C1 * (np.sinh(C2 * s) / C2 - s)
        rho = 0.5 * G2 * dphi**2
        dom = _trapezoid(np.cosh(C2 * (te - s)) * rho, s) / J
        omn = C1 * (np.cosh(C2 * te) - 1)
        return te, dom, omn, 100.0 * dom / omn

    hdr = "rate\\cap " + "".join(f"{c:>7g}" for c in args.caps)

    print(f"Constants: C2={C2:g}, K={K:g}, W={W:g} N, z={args.z_com:g} m, "
          f"G2={G2:.2f} N.m/rad^2 (arm {args.arm:g} m +{100*args.ge_margin:.0f}% GE), "
          f"J={J:.3f} kg.m^2, fs={args.fs:g} Hz")

    print("\n[1] Systematic shape deviation at the window edge [%]")
    print(hdr); print("-" * len(hdr))
    for Md in args.rates:
        print(f"{Md:7.2f} " + "".join(
            f"{propagate(Md, c)[3]:7.2f}" for c in args.caps))

    print("\n[2] Max |d_omega| over the window [mrad/s]")
    print(hdr); print("-" * len(hdr))
    for Md in args.rates:
        print(f"{Md:7.2f} " + "".join(
            f"{propagate(Md, c)[1]*1e3:7.2f}" for c in args.caps))

    print(f"\n[3] Post-onset sample count N_post (@{args.fs:g} Hz)")
    print(hdr); print("-" * len(hdr))
    for Md in args.rates:
        print(f"{Md:7.2f} " + "".join(
            f"{propagate(Md, c)[0]*args.fs:7.0f}" for c in args.caps))

    print("\n[4] Signal span omega_nom(edge) [rad/s] / noise-limited NRMSE floor [%]")
    print(hdr); print("-" * len(hdr))
    for Md in args.rates:
        row = f"{Md:7.2f} "
        for c in args.caps:
            omn = propagate(Md, c)[2]
            row += f"{100*args.gyro_noise/omn:7.1f}"
        print(row + "   (floor %)")

    print(f"\n[5] Cap at which the systematic deviation crosses "
          f"{args.criterion:g}% (binding rate = slowest)")
    for Md in args.rates:
        f = lambda c: propagate(Md, c)[3] - args.criterion
        lo, hi = min(args.caps), max(args.caps)
        if f(lo) > 0:
            print(f"  {Md:5.2f} N.m/s: < {lo:g} deg")
        elif f(hi) < 0:
            print(f"  {Md:5.2f} N.m/s: > {hi:g} deg")
        else:
            print(f"  {Md:5.2f} N.m/s: {brentq(f, lo, hi):5.2f} deg")


if __name__ == '__main__':
    main()
