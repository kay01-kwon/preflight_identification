#!/usr/bin/env python3
"""Static-threshold check figure and tables (GE on/off, benchmark included).

Reads mcrit_prediction.csv and nls_comparison_runs.csv from argv[1];
writes docs/fig_mcrit_static.pdf and prints the LaTeX table rows."""
import csv
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys; SC=sys.argv[1] if len(sys.argv)>1 else '.'
G=9.81
MASS={'case_01':3.066,'case_02':3.220,'case_03':3.220,'case_04':3.220,'case_05':3.220}
OFF={('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,('case_05','My'):-10.89}
OSGN={'Mx':1.0,'My':-1.0}
GE_LO,GE_HI=0.010,0.042
M=['cosh','nls','pelt_normal','pelt_rbf','cusum']
LBL={'cosh':'COSH','nls':'NLS','pelt_normal':'CPD-N','pelt_rbf':'CPD-R','cusum':'CUSUM'}
COL={'cosh':'#0072B2','nls':'#E69F00','pelt_normal':'#009E73','pelt_rbf':'#D55E00','cusum':'#CC79A7'}
MRK={'cosh':'o','nls':'s','pelt_normal':'^','pelt_rbf':'v','cusum':'D'}
ARMNAME={('Mx','pos'):'l_r',('Mx','neg'):'l_l',('My','pos'):'l_f',('My','neg'):'l_b'}

# group data: W, f_onset, COSH mean, offset term
pred=list(csv.DictReader(open(f'{SC}/mcrit_prediction.csv')))
data={}
for r in pred:
    W=MASS[r['case']]*G
    S=OSGN[r['axis']]*W*OFF[(r['case'],r['axis'])]*1e-3
    sgn=1.0 if r['dir']=='pos' else -1.0
    data[(r['case'],r['axis'],r['dir'])]=dict(W=W,f=float(r['f_onset']),S=S,sgn=sgn,
                                              M=float(r['M_ident']))
# benchmark directional means
bench=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f'{SC}/nls_comparison_runs.csv')):
    for m in M:
        if r.get(f'mcrit_{m}'):
            bench[(r['case'],r['axis'],r['dir'])][m].append(float(r[f'mcrit_{m}']))
bmean={k:{m:float(np.mean(v)) for m,v in d.items()} for k,d in bench.items()}

def fit_arms(gamma):
    fit={}
    for key in ARMNAME:
        g=[d for k,d in data.items() if (k[1],k[2])==key]
        A=np.array([d['W']-(1+gamma)*d['f'] for d in g])
        y=np.array([d['sgn']*(d['M']-d['S']) for d in g])
        fit[key]=float(A@y/(A@A))
    return fit

def resids(gamma,fit):
    out={}
    for k,d in data.items():
        l=fit[(k[1],k[2])]
        p=d['sgn']*(d['W']-(1+gamma)*d['f'])*l+d['S']
        out[k]=(p,1e3*(d['M']-p))
    return out

fitA=fit_arms(0.0); resA=resids(0.0,fitA)
gmid=0.5*(GE_LO+GE_HI)
fitB=fit_arms(gmid); resB=resids(gmid,fitB)
fitLo=fit_arms(GE_LO); fitHi=fit_arms(GE_HI)

print("arms [mm]:  scenario A (gamma=0)   B (gamma=2.6%)   lo(1.0%)  hi(4.2%)")
for key in sorted(ARMNAME):
    print(f"  {ARMNAME[key]}: {1e3*fitA[key]:6.1f}  {1e3*fitB[key]:6.1f}  "
          f"{1e3*fitLo[key]:6.1f}  {1e3*fitHi[key]:6.1f}")
for name,rr in (('A no-GE',resA),('B GE mid',resB)):
    a=np.abs(np.array([v[1] for v in rr.values()]))
    print(f"{name}: median {np.median(a):.1f}  p90 {np.percentile(a,90):.1f}  "
          f"max {a.max():.1f}  RMS {np.sqrt(np.mean(a**2)):.1f} mN·m")

# LaTeX table rows: per group
print("\n% table rows")
for k in sorted(data):
    d=data[k]; pA,rA=resA[k]; pB,rB=resB[k]
    bs=[bmean[k][m] for m in M[1:] if m in bmean[k]]
    c=k[0].replace('case_0','')
    print(f"{c} & {k[1][1]} & {'+' if k[2]=='pos' else '-'} & "
          f"${d['M']:+.3f}$ & ${pA:+.3f}$ & ${rA:+.0f}$ & ${pB:+.3f}$ & ${rB:+.0f}$ & "
          f"$[{min(bs):+.3f}, {max(bs):+.3f}]$ \\\\")

# figure: 1x5 cases, x slots = Mx-,Mx+,My-,My+
SLOTS=[('Mx','neg'),('Mx','pos'),('My','neg'),('My','pos')]
XL=[r'$M_{x,-}$',r'$M_{x,+}$',r'$M_{y,-}$',r'$M_{y,+}$']
cases=[f'case_0{i}' for i in range(1,6)]
fig,axes=plt.subplots(1,5,figsize=(12.5,3.2),sharey=True)
for ci,case in enumerate(cases):
    ax=axes[ci]
    ax.axhline(0,color='0.85',lw=0.7,zorder=0)
    for xi,(axname,dirn) in enumerate(SLOTS):
        k=(case,axname,dirn); d=data[k]
        # GE band with the gamma=1 arms held fixed: what applying the
        # thrust correction alone would do to the prediction
        lA=fitA[(axname,dirn)]
        pl=d['sgn']*(d['W']-(1+GE_LO)*d['f'])*lA+d['S']
        ph=d['sgn']*(d['W']-(1+GE_HI)*d['f'])*lA+d['S']
        lo,hi=sorted((pl,ph))
        ax.add_patch(plt.Rectangle((xi-0.34,lo),0.68,hi-lo,color='0.85',zorder=1))
        pA,_=resA[k]
        ax.plot([xi-0.34,xi+0.34],[pA,pA],color='k',lw=1.6,zorder=2)
        for mi,m in enumerate(M):
            val=d['M'] if m=='cosh' else bmean[k].get(m)
            if val is None: continue
            ax.plot(xi-0.24+mi*0.12,val,MRK[m],color=COL[m],ms=4.2,zorder=3,
                    label=LBL[m] if (ci==0 and xi==0) else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(XL,fontsize=8)
    ax.set_xlim(-0.6,3.6); ax.set_title(f'Case 0{ci+1}',fontsize=10)
    ax.grid(axis='y',alpha=0.25,lw=0.6); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel(r'$M_{\mathrm{crit}}$ [N$\cdot$m]')
h,l=axes[0].get_legend_handles_labels()
import matplotlib.lines as mlines, matplotlib.patches as mpatches
h+= [mlines.Line2D([],[],color='k',lw=1.6), mpatches.Patch(color='0.85')]
l+= ['static prediction ($\\gamma=1$, fitted arms)','GE band ($\\gamma-1\\in[1.0,4.2]\\%$, arms fixed)']
fig.legend(h,l,loc='upper center',ncols=7,fontsize=7.5,frameon=False,
           bbox_to_anchor=(0.5,1.08))
fig.tight_layout(rect=(0,0,1,0.96))
fig.savefig('/home/user/preflight_identification/docs/fig_mcrit_static.pdf',bbox_inches='tight')
print('figure saved')
