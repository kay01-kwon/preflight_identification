#!/usr/bin/env python3
"""Per-case CoM-offset-error figure (1x5 subplots, x/y components, 5 methods).

Reads nls_comparison_runs.csv (analysis/nls_comparison.py output) from the
directory given as argv[1]; writes docs/fig_estimator_err.pdf."""
import csv, numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path; SC=sys.argv[1] if len(sys.argv)>1 else '.'

# figures belong to this repository, wherever it is checked out
REPO = Path(__file__).resolve().parents[1]
G=9.81
MASS={'case_01':3.066,'case_02':3.220,'case_03':3.220,'case_04':3.220,'case_05':3.220}
TRUTH={('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,('case_05','My'):-10.89}
SIGN={'Mx':1.0,'My':-1.0}
# Mx (roll) senses the y-offset; My (pitch) senses the x-offset
COMP={'Mx':'y','My':'x'}
M=['cosh','cosh_cad','nls','pelt_normal','pelt_rbf','cusum']
LBL={'cosh':'COSH','cosh_cad':'COSH-CAD','nls':'NLS','pelt_normal':'CPD-N','pelt_rbf':'CPD-R','cusum':'CUSUM'}
COL={'cosh':'#0072B2','cosh_cad':'#56B4E9','nls':'#E69F00','pelt_normal':'#009E73','pelt_rbf':'#D55E00','cusum':'#CC79A7'}
MRK={'cosh':'o','cosh_cad':'X','nls':'s','pelt_normal':'^','pelt_rbf':'v','cusum':'D'}

rows=list(csv.DictReader(open(f'{SC}/nls_comparison_runs.csv')))
agg=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in M:
        if r.get(f'mcrit_{m}'): agg[(r['case'],r['axis'])][m][r['dir']].append(float(r[f'mcrit_{m}']))

def est(key,m):
    g=agg[key][m]
    p,n=np.array(g['pos']),np.array(g['neg'])
    mff=.5*(p.mean()+n.mean())
    var=.25*(p.var(ddof=1)/len(p)+n.var(ddof=1)/len(n))
    num=(p.var(ddof=1)/len(p)+n.var(ddof=1)/len(n))**2
    den=(p.var(ddof=1)/len(p))**2/(len(p)-1)+(n.var(ddof=1)/len(n))**2/(len(n)-1)
    half=stats.t.ppf(.975,num/den)*np.sqrt(var)
    W=MASS[key[0]]*G
    return SIGN[key[1]]*1e3*mff/W, 1e3*half/W

cases=[f'case_0{i}' for i in range(1,6)]
fig,axes=plt.subplots(1,5,figsize=(12.5,3.0),sharey=True)
for ci_,case in enumerate(cases):
    ax=axes[ci_]
    ax.axhline(0,color='0.35',lw=0.9,zorder=1)
    # component x <- My, component y <- Mx
    for xi,axname in enumerate(['My','Mx']):
        key=(case,axname)
        for k,m in enumerate(M):
            lam,h=est(key,m)
            e=lam-TRUTH[key]
            ax.errorbar(xi+(k-2.5)*0.11,e,yerr=h,fmt=MRK[m],color=COL[m],
                        ms=4.5,elinewidth=1.2,capsize=2.0,zorder=3,
                        label=LBL[m] if (ci_==0 and xi==0) else None)
    ax.set_xticks([0,1]); ax.set_xticklabels([r'$x_{\mathrm{off}}$',r'$y_{\mathrm{off}}$'])
    ax.set_xlim(-0.55,1.55)
    ax.set_title(f'Case 0{ci_+1}',fontsize=10)
    ax.grid(axis='y',alpha=0.25,lw=0.6); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('CoM offset error [mm]')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncols=6,fontsize=8,
           frameon=False,bbox_to_anchor=(0.5,1.06))
fig.tight_layout(rect=(0,0,1,0.97))
fig.savefig(REPO / 'docs' / 'fig_estimator_err.pdf', bbox_inches='tight')
print('saved')
