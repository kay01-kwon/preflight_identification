"""omega_dot from an analytic cosh/sinh fit, vs the SG differentiators."""
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
ALL=sorted(c.bag_name for c in crits)
name=ALL[0]
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


jp_par = j_parallel(axis, z, MASS_KG[case])
print(f"\n  calibrated C2 = {c2:.3f};  parallel-axis prediction "
      f"sqrt(Wz/J_P) = {(W*z/jp_par)**0.5:.3f} rad/s\n")
print(f"  {'bag':<14}{'C2 free':>9}{'RMS free':>10}{'RMS pinned':>12}{'phi_max':>9}")
free=[]
for bn in ALL:
    cr=next(c for c in crits if c.bag_name==bn)
    bg=next(b for b in bags if b.name==bn)
    ss=1.0 if bn.startswith('pos') else -1.0
    sg=cvp.prepare_signals(bg,axis)
    rr,pp=math_tools.quaternion_to_euler_vectorized(bg.odom.quaternion)
    pa=rr if axis=='x' else pp
    nn=min(len(pa),len(sg['t']))
    _,ii=cvp.detect_excitation_window(sg['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
    jj=cr.onset_idx; ii=min(ii,nn-1)
    if ii-jj<20: continue
    s2=slice(jj,ii+1)
    tt=sg['t'][s2]-sg['t'][jj]; oo=ss*sg['omega'][:nn][s2]
    ph=ss*(pa[s2]-pa[jj])
    _,fp,_=cosh_fit_dot(tt,oo,c2)
    b2=None
    for cv in np.linspace(2.0,20.0,900):
        _,ff_,_=cosh_fit_dot(tt,oo,cv)
        r=float(np.mean((oo-ff_)**2))
        if b2 is None or r<b2[1]: b2=(cv,r)
    _,ffit,_=cosh_fit_dot(tt,oo,b2[0])
    free.append(b2[0])
    print(f"  {bn:<14}{b2[0]:9.2f}"
          f"{np.rad2deg(np.sqrt(np.mean((oo-ffit)**2))):10.3f}"
          f"{np.rad2deg(np.sqrt(np.mean((oo-fp)**2))):12.3f}"
          f"{np.rad2deg(ph[-1]):9.2f}")
free=np.array(free)
print(f"\n  free C2: median {np.median(free):.2f}, range "
      f"{free.min():.2f}-{free.max():.2f}  (n={len(free)})")
