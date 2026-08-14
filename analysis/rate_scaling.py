#!/usr/bin/env python3
"""Where the rate dependence of the relative rate bound actually lives.

It is tempting to attribute it to the window: a slow ramp keeps an
unstable plant excited for longer, so the forcing has more time to act.
That reading is right for the ABSOLUTE deviation, which grows from
0.053 to 0.332 rad/s across the sweep because sinh x grows ninefold.
It is wrong for the relative one.  omega_nom is the output of the same
plant, driven by the ramp instead of by rho, so the amplification is
common to both and cancels.  With Jp C2^2 = Wz the ratio collapses to

    relative = coth(x/2) * rho_bar * C2 / Mdot                   (113)

in which the window survives only through coth(x/2) -- 1.106 to 1.011
over the sweep, eight per cent, and in the favourable direction.

This also corrects an earlier three-factor split of the same quantity,
Psi * rho_bar / dM_win.  Since Psi = x coth(x/2) and dM_win = (Mdot/C2)x,
the x cancels; counting Psi as a 1.58x amplification and 1/dM_win as a
6.94x signal collapse charges one effect twice, once in each direction,
and makes the window look far more important than it is.

Usage: python analysis/rate_scaling.py
"""
import numpy as np
from scipy.optimize import brentq
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_form_check import R_phi, R_GE
W,g,z,a,bM,JC=31.59,9.81,0.30,0.160,0.03446,0.0537
phi=np.deg2rad(10.0); wz=W*z; m=W/g
jp=m*(z**2+a**2); jph=JC+jp; c2=np.sqrt(wz/jp); den=jp*c2
rp=0.5*W*a*phi**2; Lam=lambda u: np.sinh(u)-u
print(f"  Wz = Jp C2^2 identity  =>  relative = coth(x/2) * rho_bar * C2 / Mdot")
print(f"  the exponential cancels: e_w and omega_nom are outputs of the SAME")
print(f"  unstable plant, one driven by rho and the other by Mdot.\n")
print(f"  {'Mdot':>6}{'x':>7}{'coth(x/2)':>11}{'rho_bar':>10}{'C2/Mdot':>10}"
      f"{'exact rel':>11}{'coth*rho*C2/M':>15}")
V={}
for md in (1.20,0.45,0.10):
    x=brentq(lambda t: Lam(t)-phi*wz*c2/md,1e-9,40)
    rg=bM*phi*(6*phi*jph*md**2)**(1/3)
    rb=R_phi(x)*rp+R_GE(x)*rg
    e=rb*np.sinh(x)/den; wn=(md/wz)*(np.cosh(x)-1)
    ct=1/np.tanh(x/2)
    V[md]=(x,ct,rb,e,wn)
    print(f"  {md:6.2f}{x:7.3f}{ct:11.4f}{1e3*rb:10.3f}{c2/md:10.3f}"
          f"{100*e/wn:10.2f}%{100*ct*rb*c2/md:14.2f}%")

print(f"\n  slow / fast, the HONEST decomposition (two factors, not three):\n")
x0,c0,r0,e0,w0=V[1.20]; x1,c1,r1,e1,w1=V[0.10]
print(f"    coth(x/2)   {c0:.4f} -> {c1:.4f}   x{c1/c0:.3f}"
      f"   window length: 8%, and it HELPS")
print(f"    rho_bar     {1e3*r0:.2f} -> {1e3*r1:.2f}   x{r1/r0:.3f}"
      f"   disturbance smaller")
print(f"    1/Mdot                        x{1.20/0.10:.1f}"
      f"     <-- essentially the whole thing")
print(f"    product                       x{(c1/c0)*(r1/r0)*12:.2f}"
      f"   (measured x{(e1/w1)/(e0/w0):.2f})")

print(f"\n  why the three-factor split Psi * rho_bar / dM_win misleads:")
print(f"    Psi = x coth(x/2)  and  dM_win = (Mdot/C2) x  --  the x CANCELS.")
print(f"    Psi x1.58 and 1/dM_win x6.94 split ONE effect across two factors.\n")

print(f"  the physical reading: moment swept in one plant time constant\n")
print(f"  1/C2 = {1/c2:.3f} s\n")
print(f"  {'Mdot':>6}{'Mdot/C2 [mN.m]':>16}{'rho_bar [mN.m]':>16}{'ratio':>9}"
      f"{'x coth':>9}{'= relative':>12}")
for md in (1.20,0.45,0.10):
    x,ct,rb,e,wn=V[md]
    print(f"  {md:6.2f}{1e3*md/c2:16.1f}{1e3*rb:16.2f}{100*rb*c2/md:8.2f}%"
          f"{ct:9.4f}{100*ct*rb*c2/md:11.2f}%")

print(f"\n  and the ABSOLUTE error, where the exposure argument is exactly right:\n")
print(f"  {'Mdot':>6}{'sinh x':>10}{'rho_bar':>10}{'|e_w|':>10}")
for md in (1.20,0.10):
    x,ct,rb,e,wn=V[md]
    print(f"  {md:6.2f}{np.sinh(x):10.2f}{1e3*rb:10.2f}{e:10.4f}")
print(f"    sinh x  x{np.sinh(V[0.10][0])/np.sinh(V[1.20][0]):.2f}"
      f"   rho_bar x{V[0.10][2]/V[1.20][2]:.2f}"
      f"   ->  |e_w| x{V[0.10][3]/V[1.20][3]:.2f}")
