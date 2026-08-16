#!/usr/bin/env python3
"""The onset-shift bound, taken from the band instead of from Cauchy-Schwarz.

The published route to (101) leaves delta t_c as an inner product,

    delta t_c = <e_omega, chi> / ||chi||^2 - (C - b) <1, chi> / ||chi||^2,

and then has nowhere to go, because e_omega is an unknown function.  The
band supplies the missing statement.  Sec. VI-E already establishes

    |e_omega(tau)| <= E(tau) = rho_bar sinh(C2 tau) / (J_P C2),

so e_omega is not unknown, it is confined to a set, and a linear
functional over a set has a supremum that can be computed.

The point of this file is that the supremum is EXACT, and for a reason
that is structural rather than lucky.  The onset sensitivity is

    chi(tau) = -C1 C2 sinh(C2 tau),

and the upper edge of the band is also proportional to sinh(C2 tau) --
it has to be, because E is the Duhamel integral of the CONSTANT rho_bar
through the same deviation dynamics whose impulse response defines chi.
Two consequences follow at once.

  chi does not change sign on the window, so the supremum of <e, chi>
  over the box |e| <= E is attained on the edge, at e = +-E.  No
  inequality is used and the bound is attained by an admissible
  perturbation: it is tight, not merely valid.

  E and |chi| are proportional, so the integral of sinh^2 that appears
  in the numerator and in ||chi||^2 is the SAME integral and cancels
  identically.  What is left is

      |delta t_c| <= rho_bar / Mdot,   hence   |Delta M_crit| <= rho_bar

  with no B(x), no window length and no ramp rate surviving.  That is
  (108), and this is where its rate-independence comes from.

Cauchy-Schwarz gives the same number here, for the same reason -- it is
tight exactly when the two functions are proportional -- so nothing is
lost by using it, but nothing is explained by it either.  The band route
says WHY the answer is rho_bar.

Two things the rewrite must not quietly drop:

  the band is TWO-SIDED.  e_omega changes sign inside the window, so
  starting from "somewhere between the nominal and the upper edge" would
  be wrong; the correct premise is |e_omega| <= E.  Measured below.

  E(tau) is not sup E.  Replacing the envelope by its supremum -- a
  constant -- costs a factor sinh(x)(cosh x - 1)/B(x), which is 1.7 to
  1.9 over the observed x.  The supremum is the right object for a
  pointwise figure and the wrong one inside this bound.

Usage: python analysis/onset_band_bound.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nonlinear_band import exact

RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)


def B(x):
    return 0.25 * np.sinh(2.0 * x) - 0.5 * x


def channels(s):
    """The onset shift the band allows, three ways, on one exact run."""
    tau, c1, c2 = s['tau'], s['c1'], s['c2']
    chi = -c1 * c2 * np.sinh(c2 * tau)
    n2 = np.trapz(chi ** 2, tau)
    # (i) the box supremum: chi has one sign, so the extremum is the edge
    box = np.trapz(s['E_sup'] * np.abs(chi), tau) / n2
    # (ii) Cauchy-Schwarz on the same envelope
    cs = np.sqrt(np.trapz(s['E_sup'] ** 2, tau)) / np.sqrt(n2)
    # (iii) the closed form the cancellation predicts
    closed = s['rb_sup'] / s['mdot']
    # (iv) what a CONSTANT envelope at its supremum would give
    flat = np.trapz(s['E_sup'].max() * np.abs(chi), tau) / n2
    # and what the run actually does
    real = np.trapz(s['e'] * chi, tau) / n2
    return box, cs, closed, flat, real


def main():
    print(f"\n  the exact nonlinear tip-over, {len(RATES)} ramp rates\n")
    print(f"  --- is the band one-sided or two-sided? ---\n")
    print(f"  {'Mdot':>6}{'x':>7}{'min e_w':>11}{'max e_w':>11}"
          f"{'sign change':>13}{'at tau/te':>11}")
    print(f"  {'N m/s':>6}{'':7}{'deg/s':>11}{'deg/s':>11}{'':13}{'':11}")
    sols = {}
    for m in RATES:
        s = exact(m)
        sols[m] = s
        e = s['e']
        sc = np.where(np.diff(np.sign(e)) != 0)[0]
        where = (f"{s['tau'][sc[0]] / s['te']:.3f}" if sc.size else '-')
        print(f"  {m:6.2f}{s['x']:7.3f}{np.rad2deg(e.min()):11.4f}"
              f"{np.rad2deg(e.max()):11.4f}"
              f"{'yes' if sc.size else 'no':>13}{where:>11}")
    print(f"\n  e_omega is negative first and positive later, so the premise")
    print(f"  must be |e_omega| <= E, not 0 <= e_omega <= E.")

    print(f"\n  --- the supremum over the band, three ways ---\n")
    print(f"  {'Mdot':>6}{'box sup':>11}{'Cauchy-S':>11}{'closed form':>13}"
          f"{'rho_bar/Mdot':>14}{'realised':>11}")
    print(f"  {'N m/s':>6}{'ms':>11}{'ms':>11}{'ms':>13}{'ms':>14}{'ms':>11}")
    for m in RATES:
        box, cs, closed, flat, real = channels(sols[m])
        print(f"  {m:6.2f}{1e3 * box:11.4f}{1e3 * cs:11.4f}"
              f"{1e3 * closed:13.4f}{1e3 * sols[m]['rb_sup'] / m:14.4f}"
              f"{1e3 * real:11.4f}")
    print(f"\n  the three agree to machine precision: the box supremum IS")
    print(f"  Cauchy-Schwarz here, because E and |chi| are proportional,")
    print(f"  and both equal rho_bar/Mdot because the sinh^2 integral")
    print(f"  cancels between the numerator and ||chi||^2.")

    print(f"\n  --- in the moment domain, which is what (108) claims ---\n")
    print(f"  {'Mdot':>6}{'bound':>11}{'rho_bar':>11}{'ratio':>9}"
          f"{'realised':>11}{'occupancy':>12}")
    print(f"  {'N m/s':>6}{'mN.m':>11}{'mN.m':>11}{'':9}{'mN.m':>11}{'':12}")
    for m in RATES:
        box, cs, closed, flat, real = channels(sols[m])
        print(f"  {m:6.2f}{1e3 * m * box:11.4f}{1e3 * sols[m]['rb_sup']:11.4f}"
              f"{m * box / sols[m]['rb_sup']:9.4f}"
              f"{1e3 * m * abs(real):11.4f}"
              f"{abs(real) / box:12.4f}")
    print(f"\n  the bound is rho_bar at every rate, exactly, and the ramp")
    print(f"  rate has cancelled out of it.  The realised shift is 2 to 20%")
    print(f"  of it, and that gap is entirely in rho_bar >= |rho| -- the")
    print(f"  propagation step contributes no slack at all.")

    print(f"\n  --- what using sup E instead of E(tau) would cost ---\n")
    print(f"  {'Mdot':>6}{'x':>7}{'E(tau)':>11}{'sup E':>11}{'inflation':>11}"
          f"{'predicted':>11}")
    print(f"  {'N m/s':>6}{'':7}{'ms':>11}{'ms':>11}{'':11}"
          f"{'sinh(cosh-1)/B':>16}")
    for m in RATES:
        box, cs, closed, flat, real = channels(sols[m])
        x = sols[m]['x']
        g = np.sinh(x) * (np.cosh(x) - 1.0) / B(x)
        print(f"  {m:6.2f}{x:7.3f}{1e3 * box:11.4f}{1e3 * flat:11.4f}"
              f"{flat / box:11.4f}{g:16.4f}")
    print(f"\n  So the envelope must enter the bound as E(tau).  Its")
    print(f"  supremum is the right object for a pointwise figure and the")
    print(f"  wrong one here: it throws away a factor of about two.")

    print(f"\n  --- the hypothesis (108) actually needs ---\n")
    print(f"  The published step pairs a RISING rho against a FALLING")
    print(f"  weight.  The weight falls, and integrates to one.  But rho")
    print(f"  does not rise: rho_GE goes as tau^4 and rho_phi as tau^6, so")
    print(f"  near the onset the restoring GE term wins and rho is")
    print(f"  negative first.  Each channel is monotone on its own.\n")
    print(f"  {'Mdot':>6}{'rho rising':>12}{'rho_phi':>10}{'|rho_GE|':>10}"
          f"{'w falling':>11}{'int w':>9}{'<|rho|>_w':>11}{'rho_bar':>9}")
    print(f"  {'N m/s':>6}{'fraction':>12}{'fraction':>10}{'fraction':>10}"
          f"{'fraction':>11}{'':9}{'mN.m':>11}{'mN.m':>9}")
    for m in RATES:
        s = sols[m]
        tau, c2 = s['tau'], s['c2']
        sh = np.sinh(c2 * tau)
        n2 = np.trapz(sh ** 2, tau)
        w = c2 * np.array([
            np.trapz(np.sinh(c2 * tau[i:])
                     * np.cosh(c2 * (tau[i:] - tau[i])), tau[i:])
            for i in range(len(tau))]) / n2
        mono = lambda v: float((np.diff(v) >= -1e-13).mean())
        aw = np.trapz(np.abs(s['rho']) * w, tau)
        print(f"  {m:6.2f}{mono(s['rho']):12.3f}{mono(s['r_phi']):10.3f}"
              f"{mono(np.abs(s['r_ge'])):10.3f}"
              f"{float((np.diff(w) <= 1e-13).mean()):11.3f}"
              f"{np.trapz(w, tau):9.4f}{1e3 * aw:11.4f}"
              f"{1e3 * s['rb_sup']:9.3f}")
    print(f"\n  So the repair is to split before bounding.  With int w = 1")
    print(f"  and w >= 0, |Delta M_crit| <= <|rho|>_w <= <rho_phi>_w +")
    print(f"  <|rho_GE|>_w, and Chebyshev now applies to each term on its")
    print(f"  own hypothesis, each being genuinely monotone.  That returns")
    print(f"  rho_phi,max/7 + rho_GE,max/5 -- the same (108), but with the")
    print(f"  two reduction factors justified by the step that uses them")
    print(f"  instead of imported past a hypothesis that does not hold for")
    print(f"  the sum.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
