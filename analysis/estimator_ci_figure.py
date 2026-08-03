#!/usr/bin/env python3
"""Forest plot: CoM offset with Welch-t 95% CIs per estimator vs truth.

Reads nls_comparison_runs.csv (analysis/nls_comparison.py output) from the
directory given as argv[1]; writes docs/fig_estimator_ci.pdf and prints the
LaTeX cell strings used in the methodology tables."""
import csv, numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys; SC=sys.argv[1] if len(sys.argv)>1 else '.'
G=9.81
MASS={'case_01':3.066,'case_02':3.220,'case_03':3.220,'case_04':3.220,'case_05':3.220}
TRUTH={('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,('case_05','My'):-10.89}
SIGN={'Mx':1.0,'My':-1.0}
M=['cosh','nls','pelt_normal','pelt_rbf','cusum']
LBL={'cosh':'COSH','nls':'NLS','pelt_normal':'CPD-N','pelt_rbf':'CPD-R','cusum':'CUSUM'}
COL={'cosh':'#0072B2','nls':'#E69F00','pelt_normal':'#009E73','pelt_rbf':'#D55E00','cusum':'#CC79A7'}
MRK={'cosh':'o','nls':'s','pelt_normal':'^','pelt_rbf':'v','cusum':'D'}

rows=list(csv.DictReader(open(f'{SC}/nls_comparison_runs.csv')))
agg=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in M:
        if r.get(f'mcrit_{m}'): agg[(r['case'],r['axis'])][m][r['dir']].append(float(r[f'mcrit_{m}']))

def ci(key,m):
    g=agg[key][m]
    p,n=np.array(g['pos']),np.array(g['neg'])
    mff=.5*(p.mean()+n.mean())
    var=.25*(p.var(ddof=1)/len(p)+n.var(ddof=1)/len(n))
    num=(p.var(ddof=1)/len(p)+n.var(ddof=1)/len(n))**2
    den=(p.var(ddof=1)/len(p))**2/(len(p)-1)+(n.var(ddof=1)/len(n))**2/(len(n)-1)
    df=num/den
    half=stats.t.ppf(.975,df)*np.sqrt(var)
    W=MASS[key[0]]*G
    return SIGN[key[1]]*1e3*mff/W, 1e3*half/W, half  # lam, lam_half, Mff_half

keys=sorted(agg)
fig,ax=plt.subplots(figsize=(7.0,6.0))
ylab=[]
for i,key in enumerate(keys):
    y0=len(keys)-1-i
    tr=TRUTH[key]
    ax.axhline(y0-0.5,color='0.92',lw=0.6,zorder=0)
    for k,m in enumerate(M):
        lam,h,_=ci(key,m)
        y=y0+0.30-k*0.15
        ax.errorbar(lam,y,xerr=h,fmt=MRK[m],color=COL[m],ms=4.5,
                    elinewidth=1.2,capsize=2.0,zorder=3,
                    label=LBL[m] if i==0 else None)
    ax.plot(tr,y0,marker='|',color='k',ms=26,mew=2.2,zorder=4,
            label='load-cell truth' if i==0 else None)
    ylab.append(f"{key[0].replace('case_0','')}{key[1][1]}")
ax.set_yticks([len(keys)-1-i for i in range(len(keys))]); ax.set_yticklabels(ylab)
ax.set_xlabel('CoM offset [mm]'); ax.set_ylabel('case–axis')
ax.grid(axis='x',alpha=0.25,lw=0.6); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.legend(loc='upper left',fontsize=8,frameon=True,framealpha=0.9)
fig.tight_layout()
fig.savefig('/home/user/preflight_identification/docs/fig_estimator_ci.pdf')
print('figure saved')

# LaTeX strings: Table-2 cells (lam +- h) and Table-3 CI rows (Mff half, N*m)
print("\n% Table 2 cells: lam ± h")
for key in keys:
    c=key[0].replace('case_0',''); axn=key[1][1]
    cells=" & ".join((lambda t: f"${t[0]:+.2f}\\,{{\\scriptstyle\\pm{t[1]:.2f}}}$")(ci(key,m)) for m in M)
    print(f"{c} & {axn} & ${TRUTH[key]:+.2f}$ & {cells} \\\\")
print("\n% Table 3 CI rows (±CI95 of Mff, N*m) per method")
for m in M:
    cells=" & ".join(f"$\\pm{ci(key,m)[2]:.3f}$" for key in keys)
    print(f"{LBL[m]} & CI$_{{95}}$ & {cells} \\\\")
