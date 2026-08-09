"""CV from the FULL pipeline for the three COSH constant choices, so the
numbers are directly comparable with the PNLS-centred run."""
import contextlib, io, sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import MASS_KG, G, ROOT

Z_CAD = 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
acc = {'cal': [], 'cad': []}
print(f"{'case':<9}{'ax':<4}{'cal':>8}{'CAD':>8}")
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    key = (d.parent.name, d.name); mass = MASS_KG[key[0]]
    j = J_CAD[axis] + mass * (Z_CAD**2 + LP[axis]**2)
    c2c, kc = float(np.sqrt(mass*G*Z_CAD/j)), 1.0/(mass*G*Z_CAD)
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        r_cal, _ = cvp.extract_piecewise_batch(bags, axis)
        r_cad, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2c,
                                               ramp_gain=kc)
    row = []
    for lab, cr in (('cal', r_cal), ('cad', r_cad)):
        p = np.array([c.onset_moment for c in cr if c.bag_name.startswith('pos')])
        n = np.array([c.onset_moment for c in cr if c.bag_name.startswith('neg')])
        cv = 50*(p.std(ddof=1)/abs(p.mean()) + n.std(ddof=1)/abs(n.mean()))
        acc[lab].append(cv); row.append(cv)
    print(f"{key[0]:<9}{key[1]:<4}{row[0]:7.1f}%{row[1]:7.1f}%")
for lab in ('cal', 'cad'):
    a = np.array(acc[lab])
    print(f"{lab:>13}  mean {a.mean():.1f}%  worst {a.max():.1f}%")
