"""Measure delta_e_omega per its definition and set it against (107).

delta_e = e_omega - beta sinh(C2 tau), with e_omega the pinned-fit
deviation (amplitude C1 = K Mdot, theory constants) and beta the
end-matched coefficient (97b). Reported raw and low-band (<=5 Hz).
Cap: (107) with the box-tilt anchors; GE channel only on hardware.
Usage: python de_measure.py {sim|hw}
"""
import contextlib, io, sys, csv
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp

MODE = sys.argv[1]
G=9.81; R_PHI=1/7; R_GE=1/5

if MODE == 'sim':
    from utils.extractor import load_packed_dataset
    Z=0.272; PHI=np.deg2rad(5.0); BETA_M=0.0
    J_CAD={'x':0.051085,'y':0.050564}
    CASES={'S1':3.066,'S2':3.066,'S3':3.066,'S4':3.066,'S5':3.066,
           'S6':3.066,'S7':3.066,'S8':3.066,'S9':3.066,'S11':3.066,
           'S13':3.220}
    OFF={'S1':{'Mx':0.0,'My':6.0},'S2':{'Mx':10.0,'My':0.0},
         'S3':{'Mx':5.0,'My':10.0},'S4':{'Mx':20.0,'My':20.0},
         'S5':{'Mx':20.0,'My':20.0},'S6':{'Mx':20.0,'My':20.0},
         'S7':{'Mx':20.0,'My':20.0},'S8':{'Mx':25.0,'My':25.0},
         'S9':{'Mx':32.0,'My':32.0},'S11':{'Mx':14.0,'My':38.0},
         'S13':{'Mx':25.0,'My':25.0}}
    L={'Mx':0.110,'My':0.140}
    def datasets():
        for case, mass in CASES.items():
            for simax, axis in (('Mx','x'),('My','y')):
                arm=L[simax]+OFF[case][simax]*1e-3
                with contextlib.redirect_stdout(io.StringIO()):
                    bags=load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
                yield case, simax, axis, mass, arm, L[simax], bags
else:
    from utils.extractor import load_excitation_dataset
    from pathlib import Path
    Z=0.30; PHI=np.deg2rad(10.0); BETA_M=0.0345
    J_CAD={'x':0.0537,'y':0.0537}
    MASS={'case_01':3.066,'case_02':3.220,'case_03':3.220,
          'case_04':3.220,'case_05':3.220}
    ARM={'Mx':0.160,'My':0.130}; LP={'Mx':0.140,'My':0.110}
    def datasets():
        for d in sorted(Path('DataSet/exp').glob('case_*/M[xy]')):
            axis='x' if d.name=='Mx' else 'y'
            with contextlib.redirect_stdout(io.StringIO()):
                bags=load_excitation_dataset(d)
            yield (d.parent.name, d.name, axis, MASS[d.parent.name],
                   ARM[d.name], LP[d.name], bags)

rows=[]
for case, simax, axis, mass, arm, lp, bags in datasets():
    W=mass*G
    jp=J_CAD[axis]+mass*(Z**2+lp**2)
    c2=float(np.sqrt(W*Z/jp)); k=1.0/(W*Z)
    rb=R_PHI*0.5*W*arm*PHI**2
    for bag in bags:
        md=cvp.commanded_ramp_rate(bag.name)
        if md is None: continue
        with contextlib.redirect_stdout(io.StringIO()):
            sig=cvp.prepare_signals(bag,axis)
        i0,i1=cvp.detect_excitation_window(
            sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
        w=slice(i0,i1+1)
        t,om,mom=sig['t'][w],sig['omega'][w],sig['moment'][w]
        if len(t)<24: continue
        mds=float(np.polyfit(t,mom,1)[0])
        pw=cvp.cosh_onset_fit(t,om,np.zeros_like(t),onset_guess=None,
                              c2_fixed=c2,moment_floor=0.0,
                              ramp_gain=k,ramp_rate=mds)
        j=pw['onset_idx']
        if j<12 or len(om)-j<12: continue
        tau=t[j:]-t[j]
        e=om[j:]-pw['omega_pred'][j:]          # e_omega (pinned deviation)
        x_r=min(float(c2*tau[-1]),30.0)
        beta=e[-1]/np.sinh(x_r)
        de=e-beta*np.sinh(np.clip(c2*tau,0,30))
        dt=float(np.median(np.diff(tau)))
        dd=de-de.mean()
        F=np.fft.rfft(dd); fr=np.fft.rfftfreq(len(dd),d=dt)
        F[fr>5.0]=0.0
        lo=np.fft.irfft(F,n=len(dd))
        # (107) at the box tilt, per rate
        xd=brentq(lambda v: np.sinh(v)-v-PHI*W*Z*c2/md, 1e-3, 40)
        om_max=k*md*(np.cosh(xd)-1)+rb*np.sinh(xd)/(jp*c2)
        rd_phi=W*arm*PHI*om_max
        rd_ge=BETA_M*(md*PHI+md*(xd/c2)*om_max)
        cap=(rd_ge/(2*np.e)+rd_phi/12.0)*k
        rows.append(dict(case=case,ax=simax,rate=md,
                         de=np.degrees(float(np.sqrt(np.mean(de**2)))),
                         de_lo=np.degrees(float(np.sqrt(np.mean(lo**2)))),
                         cap=np.degrees(cap)))
    print('done',case,simax,flush=True)

with open(f'docs/{MODE}_de_runs.csv','w',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[0])); wr.writeheader()
    wr.writerows(rows)
g=defaultdict(list)
for r in rows: g[r['rate']].append(r)
print(f"\n[{MODE}]  {'rate':>6}{'n':>4}{'cap(107)':>10}{'RMS de':>9}"
      f"{'RMS de_lo':>10}{'in(raw)':>9}{'in(lo)':>8}")
tr=tl=tt=0
for rate in sorted(g):
    v=g[rate]
    ir=sum(r['de']<=r['cap'] for r in v)
    il=sum(r['de_lo']<=r['cap'] for r in v)
    tr+=ir; tl+=il; tt+=len(v)
    print(f"       {rate:>6.2f}{len(v):>4}"
          f"{np.mean([r['cap'] for r in v]):>10.3f}"
          f"{np.mean([r['de'] for r in v]):>9.3f}"
          f"{np.mean([r['de_lo'] for r in v]):>10.3f}"
          f"{ir:>6}/{len(v)}{il:>5}/{len(v)}")
print(f"\n[{MODE}] raw {tr}/{tt}, low-band {tl}/{tt}")
