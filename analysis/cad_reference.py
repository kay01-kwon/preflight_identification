"""Run the identification with constants taken entirely from CAD.

CAD gives z_CoM = 0.256 m and the CoM inertias of Table 5, so the
parallel-axis theorem fixes J_P and hence BOTH constants:
    C2 = sqrt(W z / J_P),   K = 1/(W z).
Nothing is fitted.  This reports the rate-consistency CV and the
delivered CoM offset against the load-cell truth.
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import prepare, MASS_KG, G, ROOT, STRIDE

Z_CAD = 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
OFF = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
       ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
       ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
       ('case_05','My'):-10.89}
SGN = {'Mx': +1.0, 'My': -1.0}


def constants(axis, mass, z):
    j = J_CAD[axis] + mass * (z ** 2 + LP[axis] ** 2)
    w = mass * G
    return float(np.sqrt(w * z / j)), 1.0 / (w * z), j


def cv_of(prep, c2, k):
    g = {}
    for side, t, om, m, m_dot in prep:
        pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                c2_fixed=float(c2), moment_floor=0.0,
                                ramp_gain=float(k), ramp_rate=m_dot,
                                step_s=STRIDE * float(np.median(np.diff(t))))
        g.setdefault(side, []).append(float(m[pw['onset_idx']]))
    v = [float(np.std(x)) / abs(float(np.mean(x))) for x in g.values() if len(x) > 1]
    return 100 * float(np.mean(v)) if v else float('nan')


data = {}
for d in sorted(ROOT.glob('case_*/M[xy]')):
    ax = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    data[(d.parent.name, d.name)] = (ax, bags, prepare(bags, ax))

print(f"constants from CAD alone (z_CoM = {Z_CAD} m), nothing fitted:\n")
print(f"{'case':<9}{'ax':<4}{'J_P':>7}{'C2':>7}{'K':>8}{'CV%':>7}"
      f"{'truth':>8}{'ident':>8}{'err':>7}")
print('-' * 63)
errs, cvs = [], []
for key in sorted(data):
    ax, bags, prep = data[key]
    mass = MASS_KG[key[0]]
    c2, k, j = constants(ax, mass, Z_CAD)
    cv = cv_of(prep, c2, k); cvs.append(cv)
    with contextlib.redirect_stdout(io.StringIO()):
        cr, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2, ramp_gain=k)
    p = [c.onset_moment for c in cr if c.bag_name.startswith('pos')]
    n = [c.onset_moment for c in cr if c.bag_name.startswith('neg')]
    ident = 1e3 * SGN[key[1]] * 0.5 * (np.mean(p) + np.mean(n)) / (mass * G)
    e = ident - OFF[key]; errs.append(e)
    print(f"{key[0]:<9}{key[1]:<4}{j:7.3f}{c2:7.3f}{k:8.4f}{cv:7.1f}"
          f"{OFF[key]:8.2f}{ident:8.2f}{e:7.2f}")
e = np.array(errs)
print('-' * 63)
print(f"CAD-only    RMS {np.sqrt(np.mean(e**2)):.2f} mm   max|e| {np.abs(e).max():.2f} mm"
      f"   mean {e.mean():+.2f} mm   CV mean {np.mean(cvs):.1f}%  worst {np.max(cvs):.1f}%")
print(f"free (20 fitted numbers)  RMS 1.64   max|e| 3.36   mean -0.95   CV mean 2.4%  worst 4.2%")


# ---------------------------------------------------------------------
# Rate consistency with K deliberately mis-specified about the CAD point.
# The asymmetry is informative: halving K is much worse than doubling it,
# because the fitted constants prefer a K larger than CAD gives.
print("\nK mis-specified about the CAD point:\n")
print(f"{'case':<9}{'ax':<4}{'CAD':>8}{'K/2':>8}{'2K':>8}")
acc = {0.5: [], 1.0: [], 2.0: []}
for key in sorted(data):
    ax, _, prep = data[key]
    mass = MASS_KG[key[0]]
    c2, k, _ = constants(ax, mass, Z_CAD)
    row = []
    for f in (1.0, 0.5, 2.0):
        c = cv_of(prep, c2, f * k)
        acc[f].append(c)
        row.append(c)
    print(f"{key[0]:<9}{key[1]:<4}" + ''.join(f"{r:8.1f}" for r in row))
print(f"{'mean':<13}" + ''.join(f"{np.mean(acc[f]):8.1f}" for f in (1.0, 0.5, 2.0)))
print(f"{'worst':<13}" + ''.join(f"{np.max(acc[f]):8.1f}" for f in (1.0, 0.5, 2.0)))
