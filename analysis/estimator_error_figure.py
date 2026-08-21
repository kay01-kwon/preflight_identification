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

# One shared x axis: ten case-component slots at unit spacing, the six
# estimators dodged inside each slot, and the case names annotated above
# the panel with light separators between cases -- rather than five
# subplots, which repeat the axis furniture and break the eye's ability
# to compare across cases.
fig,ax=plt.subplots(figsize=(12.5,3.4))
SLOT=1.0            # spacing between the x_off / y_off slots
GAP=0.55            # extra gap between cases
pos={}              # (case, axname) -> x centre
x=0.0
for case in cases:
    for axname in ('My','Mx'):          # x_off from pitch, y_off from roll
        pos[(case,axname)]=x
        x+=SLOT
    x+=GAP

ax.axhline(0,color='0.35',lw=0.9,zorder=1)
for case in cases:
    for axname in ('My','Mx'):
        key=(case,axname); xc=pos[key]
        for k,m in enumerate(M):
            lam,h=est(key,m)
            ax.errorbar(xc+(k-2.5)*0.115,lam-TRUTH[key],yerr=h,
                        fmt=MRK[m],color=COL[m],ms=5.5,elinewidth=1.2,
                        capsize=2.0,zorder=3,
                        label=LBL[m] if key==(cases[0],'My') else None)

ax.set_xticks([pos[(c,a)] for c in cases for a in ('My','Mx')])
ax.set_xticklabels([r'$x_{\mathrm{off}}$',r'$y_{\mathrm{off}}$']*len(cases),
                   fontsize=9)
ax.set_xlim(pos[(cases[0],'My')]-0.75, pos[(cases[-1],'Mx')]+0.75)
ax.set_ylabel('CoM offset error [mm]')
ax.grid(axis='both',alpha=0.25,lw=0.6); ax.set_axisbelow(True)
for sp in ('top','right'): ax.spines[sp].set_visible(False)

# case names above the panel, with separators in between
ylim=ax.get_ylim(); span=ylim[1]-ylim[0]
ax.set_ylim(ylim[0], ylim[1]+0.13*span)
ytxt=ylim[1]+0.045*span
for ci_,case in enumerate(cases):
    xc=0.5*(pos[(case,'My')]+pos[(case,'Mx')])
    ax.text(xc,ytxt,f'Case 0{ci_+1}',ha='center',va='bottom',fontsize=10)
    if ci_:
        xs=pos[(case,'My')]-0.5*(SLOT+GAP)
        ax.axvline(xs,color='0.85',lw=0.8,zorder=0)

handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=6,fontsize=8,
           frameon=False,bbox_to_anchor=(0.5,1.05))
fig.tight_layout(rect=(0,0,1,0.96))
fig.savefig(REPO / 'docs' / 'fig_estimator_err.pdf', bbox_inches='tight')
fig.savefig(REPO / 'docs' / 'fig_estimator_err.png', dpi=150,
            bbox_inches='tight')
print('saved')
