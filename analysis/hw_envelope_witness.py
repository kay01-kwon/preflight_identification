"""Hardware side of the envelope-witness comparison: free-fit residual
per run, and the a priori envelope cap with the manuscript's design
constants (10-deg cap, Z=0.30, ARM/LP of Sec VI-E)."""
import contextlib, io, sys, csv
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from pathlib import Path

rows=[]
for d in sorted(Path('DataSet/exp').glob('case_*/M[xy]')):
    axis='x' if d.name=='Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags=load_excitation_dataset(d)
    for bag in bags:
        md=cvp.commanded_ramp_rate(bag.name)
        if md is None: continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, pw = cvp.extract_piecewise(bag, axis, model='cosh',
                                                 cosh_c2=None, ramp_gain=None)
            rows.append(dict(case=d.parent.name, ax=d.name, rate=md,
                             res=float(np.degrees(pw['rmse']))))
        except Exception as e:
            print('fail', d, bag.name, e, flush=True)
    print('done', d, flush=True)
with open('docs/hw_env_witness_runs.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)
print(len(rows), 'hardware free fits')
