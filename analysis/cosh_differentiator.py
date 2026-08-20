#!/usr/bin/env python3
"""Why an analytic cosh/sinh derivative cannot settle the GE check.

Fitting omega over the window with the linearised solution family
{cosh(C2 tau), sinh(C2 tau), 1} -- linear in the three coefficients
once C2 is fixed -- and differentiating ANALYTICALLY is the
best-behaved differentiator available: no window, no edge, no
regime mixing at the moment cap, and the 4-7 Hz load-excited band of
analysis/omega_band_probe.py is averaged away by a three-parameter
global fit.  It is nevertheless the wrong instrument HERE, and the
reason is structural rather than numerical.

THE CIRCULARITY.  A ground-effect moment proportional to tilt is a
STIFFNESS term: in the linearised balance J_P omega_dot = W z phi +
(ramp) it adds to W z and therefore shows up in one place only, the
exponent C2 = sqrt(W z / J_P).  So:

  * C2 PINNED to the no-GE value forces the attitude dependence to
    zero.  The family IS the solution set of the linearised dynamics,
    so J_P omega_dot reproduces W z phi + const along it by
    construction and dM_GE comes out flat whatever the ground effect
    really does.  Measured on case_02/Mx/pos_Mx_01: slope -6.7
    mN.m/deg against the model's -3.1 -- near-perfect agreement that
    demonstrates nothing.

  * C2 FREE absorbs the attitude dependence INTO the exponent, where
    it is indistinguishable from an error in J_P or z_CoM.  Slope
    -25.3 at a fitted C2 of 4.76 rad/s.

Either way the attitude dependence is not recoverable by this route --
the same degeneracy recorded in ge_dynamics_check.py, reached from the
differentiator side.  The information about departures from the family
lives in the RESIDUAL of the pinned fit, not in its derivative, which
is what docs/access_tight_rms_bound.tex bounds.

A SIDE FINDING worth its own look.  Fitted freely on the excitation
windows themselves, C2 lands at a median of 4.72 rad/s (range
3.80-5.92 over the 14 bags of case_02/Mx) -- essentially the CAD
parallel-axis prediction sqrt(W z / J_P) = 4.97 -- and fits better
than the calibrated 6.125 on EVERY run.  That points the 23% exponent
discrepancy behind J_P = W z / C2^2 = 0.220 (below the rigid-body
floor 0.283) at the C2 calibration rather than at the windows.
Caveat: the three-parameter family with C2 free is partly degenerate
over a 0.8 s window, since a smaller exponent with a larger sinh
coefficient mimics a larger one, so this is a diagnostic, not a
conclusion.  analysis/c2_free_check.py runs the sweep.

Usage: PYTHONPATH=<stubs> python analysis/cosh_differentiator.py
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np
_R='/home/user/preflight_identification'
sys.path.insert(0,_R); sys.path.insert(0,_R+'/analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment, LP
from analysis.rate_derivative import omega_dot, omega_dot_poly
from analysis.ge_dynamics_check import MASS_KG, G, OFF_SIGN, OFF_MM, j_parallel

d=Path(_R+'/DataSet/exp/case_02/Mx'); axis='x'; case='case_02'; axn='Mx'
z=0.261; W=MASS_KG[case]*G
off=OFF_SIGN[axn]*OFF_MM[(case,axn)]*1e-3
with contextlib.redirect_stdout(io.StringIO()):
    bags=load_excitation_dataset(d)
    c2,kg=cvp.estimate_rig_constants(bags,axis)
    crits,_=cvp.extract_piecewise_batch(bags,axis,cosh_c2=c2,ramp_gain=kg)
name=sorted(c.bag_name for c in crits if c.bag_name.startswith('pos'))[0]
crit=next(c for c in crits if c.bag_name==name); bag=next(b for b in bags if b.name==name)
s=1.0
piv=cvp.estimate_pivot_from_mocap(bag,crit.onset_time,axis)
lp=piv['pivot_abs']*1e-3 if not np.isnan(piv['pivot_abs']) else LP[axis]
arm=lp+s*off; j_p=j_parallel(axis,z,MASS_KG[case])
sig=cvp.prepare_signals(bag,axis)
roll,pitch=math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
phi_all=roll if axis=='x' else pitch
n=min(len(phi_all),len(sig['t']))
_,i1=cvp.detect_excitation_window(sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
j=crit.onset_idx; i1=min(i1,n-1); sl=slice(j,i1+1)
tau=sig['t'][sl]-sig['t'][j]; phi=s*(phi_all[sl]-phi_all[j])
m=s*sig['moment'][sl]; f=sig['f_col'][sl]
dt=float(np.median(np.diff(tau))); om_full=s*sig['omega'][:n]; om=om_full[sl]
ge_mod=s*ge_moment(bag,sig,axis,n,pos=True,window=sl)[sl]

def cosh_fit_dot(tau, om, c2v):
    """omega = a cosh + b sinh + c  (linear in a,b,c with C2 fixed);
    omega_dot = C2 (a sinh + b cosh), analytic."""
    A=np.column_stack([np.cosh(c2v*tau), np.sinh(c2v*tau), np.ones_like(tau)])
    co,*_=np.linalg.lstsq(A,om,rcond=None)
    fit=A@co
    return c2v*(co[0]*np.sinh(c2v*tau)+co[1]*np.cosh(c2v*tau)), fit, co

def slope(od):
    ge=j_p*od-m-f*lp+W*arm*np.cos(phi)-W*z*np.sin(phi)
    sd,id_=np.polyfit(phi,ge,1)
    return 1e3*sd*np.pi/180, 1e3*id_

sm,im=np.polyfit(phi,ge_mod,1)
print(f"\n  {case}/{axn}/{name}  C2(calibrated) = {c2:.3f} rad/s")
print(f"  model: slope {1e3*sm*np.pi/180:+.1f} mN.m/deg, "
      f"intercept {1e3*im:.1f} mN.m\n")
print(f"  {'differentiator':<36}{'fit RMS':>10}{'slope':>9}{'intercept':>11}")
for lab,od,fr in [
    ('SG w=9  order 2  (deployed)', omega_dot(om_full,dt,9,2)[sl], None),
    ('SG w=41 order 2', omega_dot(om_full,dt,41,2)[sl], None),
    ('SG w=41 order 7', omega_dot(om_full,dt,41,7)[sl], None),
    ('anchored poly order 5 (windowless)',
     omega_dot_poly(tau,om,order=5)[0] if isinstance(omega_dot_poly(tau,om,order=5),tuple)
     else omega_dot_poly(tau,om,order=5), None)]:
    sd,id_=slope(od)
    print(f"  {lab:<36}{'':>10}{sd:9.1f}{id_:11.1f}")

# cosh/sinh analytic, C2 pinned then free
odp,fit,co=cosh_fit_dot(tau,om,c2)
sd,id_=slope(odp)
print(f"  {'cosh+sinh fit, C2 PINNED '+f'({c2:.2f})':<36}"
      f"{np.rad2deg(np.sqrt(np.mean((om-fit)**2))):10.3f}{sd:9.1f}{id_:11.1f}")
best=None
for c2v in np.linspace(2.0,20.0,900):
    _,fit_,_=cosh_fit_dot(tau,om,c2v)
    r=float(np.mean((om-fit_)**2))
    if best is None or r<best[1]: best=(c2v,r)
c2f=best[0]; odf,fitf,_=cosh_fit_dot(tau,om,c2f)
sd,id_=slope(odf)
print(f"  {'cosh+sinh fit, C2 FREE '+f'({c2f:.2f})':<36}"
      f"{np.rad2deg(np.sqrt(np.mean((om-fitf)**2))):10.3f}{sd:9.1f}{id_:11.1f}")
print(f"\n  max tilt in window: {np.rad2deg(phi[-1]):.2f} deg")
