"""cap = Phi + tau_lag K Mdot C2 sqrt(B/x) + noise, checked per run."""
import contextlib, io, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G=9.81; Z=0.272; R_PHI=1/7; PHI_BOX=np.deg2rad(5.0)
TAU=0.012; NOISE=2.45e-4
J_CAD={'x':0.051085,'y':0.050564}
CASES={'S1':(0.0,3.066),'S2':(10.0,3.066),'S3':(5.0,3.066),
       'S4':(20.0,3.066),'S5':(20.0,3.066),'S6':(20.0,3.066),
       'S7':(20.0,3.066),'S8':(25.0,3.066),'S9':(32.0,3.066),
       'S11':(38.0,3.066),'S13':(25.0,3.220)}
OFF={'S1':{'Mx':0.0,'My':6.0},'S2':{'Mx':10.0,'My':0.0},
     'S3':{'Mx':5.0,'My':10.0},'S4':{'Mx':20.0,'My':20.0},
     'S5':{'Mx':20.0,'My':20.0},'S6':{'Mx':20.0,'My':20.0},
     'S7':{'Mx':20.0,'My':20.0},'S8':{'Mx':25.0,'My':25.0},
     'S9':{'Mx':32.0,'My':32.0},'S11':{'Mx':14.0,'My':38.0},
     'S13':{'Mx':25.0,'My':25.0}}
L={'Mx':0.110,'My':0.140}

def phi_term(tau, c2, k, rb):
    jp=1.0/(k*c2**2); N=len(tau)
    u=np.cosh(np.clip(c2*tau,0,30))-1.0
    ut=u-u.mean(); su2=float(ut@ut)
    ds=np.gradient(tau)
    T=tau[:,None]-tau[None,:]
    Km=np.where(T>=0,np.cosh(np.clip(c2*np.maximum(T,0),0,30))/jp,0.0)*ds[None,:]
    R=Km-Km.mean(axis=0)[None,:]-ut[:,None]*((ut@Km)/su2)[None,:]
    cn=np.sqrt((R**2).sum(axis=0))
    return min(rb*(cn/np.sqrt(N)).sum(),
               rb*np.sqrt(np.sum(np.abs(R).sum(axis=1)**2)/N))

g=defaultdict(list); worst=0.0
for case,(offmax,mass) in CASES.items():
    W=mass*G
    for simax,axis in (('Mx','x'),('My','y')):
        arm=L[simax]+OFF[case][simax]*1e-3
        jp=J_CAD[axis]+mass*(Z**2+L[simax]**2)
        c2=float(np.sqrt(W*Z/jp)); k=1.0/(W*Z)
        rb=R_PHI*0.5*W*arm*PHI_BOX**2
        with contextlib.redirect_stdout(io.StringIO()):
            bags=load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
        for bag in bags:
            rate=cvp.commanded_ramp_rate(bag.name)
            if rate is None: continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig=cvp.prepare_signals(bag,axis)
            i0,i1=cvp.detect_excitation_window(
                sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
            w=slice(i0,i1+1)
            t,om,mom=sig['t'][w],sig['omega'][w],sig['moment'][w]
            if len(t)<24: continue
            md=float(np.polyfit(t,mom,1)[0])
            pw=cvp.cosh_onset_fit(t,om,np.zeros_like(t),onset_guess=None,
                                  c2_fixed=c2,moment_floor=0.0,
                                  ramp_gain=k,ramp_rate=md)
            j=pw['onset_idx']
            if j<12 or len(om)-j<12: continue
            tau=t[j:]-t[j]
            r=om[j:]-pw['omega_pred'][j:]
            x=float(c2*tau[-1])
            B=0.25*np.sinh(2*x)-0.5*x
            lag=TAU*k*abs(md)*c2*np.sqrt(B/x)
            cap=phi_term(tau,c2,k,rb)+lag+NOISE
            meas=float(np.sqrt(np.mean(r**2)))
            worst=max(worst,meas/cap)
            g[rate].append((np.degrees(meas),np.degrees(cap)))
    print('done',case,flush=True)
print(f"\ntau_lag = {1e3*TAU:.0f} ms\n"
      f"{'rate':>6}{'n':>4}{'cap':>8}{'resid':>8}{'inside':>8}")
ti=tt=0
for rate in sorted(g):
    v=np.array(g[rate])
    ins=int((v[:,0]<=v[:,1]).sum()); ti+=ins; tt+=len(v)
    print(f"{rate:>6.2f}{len(v):>4}{v[:,1].mean():>8.3f}{v[:,0].mean():>8.3f}"
          f"{ins:>5}/{len(v)}")
print(f"\n{ti}/{tt} inside; worst usage {worst:.2f}")
