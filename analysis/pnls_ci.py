"""CI and CV for the PNLS-centred constants, matching mff_table.py."""
import contextlib, io, sys, pickle
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import MASS_KG, G, ROOT

SC = Path(__file__).resolve().parent
CONST = pickle.load(open(SC / 'pnls_centred.pkl', 'rb'))['constants']
TRUTH = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
         ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
         ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
         ('case_05','My'):-10.89}
SIGN = {'Mx': +1.0, 'My': -1.0}
res, cvs = {}, []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    key = (d.parent.name, d.name)
    c2, k = CONST[key]
    w = MASS_KG[key[0]] * G
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        cr, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=k)
    p = np.array([c.onset_moment for c in cr if c.bag_name.startswith('pos')])
    n = np.array([c.onset_moment for c in cr if c.bag_name.startswith('neg')])
    mff = 0.5 * (p.mean() + n.mean())
    var = 0.25 * (p.var(ddof=1)/len(p) + n.var(ddof=1)/len(n))
    dfw = var**2 / ((0.25*p.var(ddof=1)/len(p))**2/(len(p)-1)
                    + (0.25*n.var(ddof=1)/len(n))**2/(len(n)-1))
    ci = stats.t.ppf(0.975, dfw) * np.sqrt(var)
    off = SIGN[key[1]] * 1e3 * mff / w
    cv = 50*(p.std(ddof=1)/abs(p.mean()) + n.std(ddof=1)/abs(n.mean()))
    cvs.append(cv)
    res[key] = (1e3*mff, 1e3*ci, off, 1e3*ci/w, off - TRUTH[key], cv)
    print(f"{key[0]:<9}{key[1]:<4}{1e3*mff:+10.1f} +-{1e3*ci:5.1f}"
          f"{off:+9.2f} +-{1e3*ci/w:4.2f}{TRUTH[key]:+8.2f}"
          f"{off-TRUTH[key]:+7.2f}{cv:7.1f}%")
e = np.array([v[4] for v in res.values()])
print(f"\nRMS {np.sqrt((e**2).mean()):.2f} mm  max {np.abs(e).max():.2f}  "
      f"mean {e.mean():+.2f}  CV mean {np.mean(cvs):.1f}%  worst {np.max(cvs):.1f}%")
pickle.dump(res, open(SC / 'pnls_ci.pkl', 'wb'))
