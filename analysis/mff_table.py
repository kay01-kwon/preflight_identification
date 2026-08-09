"""Feed-forward compensation moment and CoM offset, all five estimators.

Adds a sixth column to the benchmark of analysis/nls_comparison.py: COSH
run on constants derived from CAD alone (z_CoM = 0.261 m, Table 5
inertias, parallel-axis theorem; nothing fitted).  The per-run
estimators -- NLS, the two change-point detectors and CUSUM -- do not
consume (C2, K) at all, so only the COSH column can move; the table
shows by how much.

Reported per case/axis: the directional means of M_crit, the pivot-free
average M_ff = (mean+ + mean-)/2 which IS the moment to compensate, its
95% Welch interval, and the CoM offset lambda = +/- M_ff / W.
"""
import contextlib, io, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from analysis.pelt_crosscheck import classic_onset_index, _window, CLASSIC

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}
TRUTH_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
            ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
            ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
            ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
            ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
SIGN = {'Mx': +1.0, 'My': -1.0}
Z_CAD = 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
METHODS = ['cosh', 'cosh_cad', 'nls'] + CLASSIC

agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    key = (d.parent.name, d.name)
    mass = MASS_KG[key[0]]
    j = J_CAD[axis] + mass * (Z_CAD ** 2 + LP[axis] ** 2)
    c2_cad = float(np.sqrt(mass * G * Z_CAD / j))
    k_cad = 1.0 / (mass * G * Z_CAD)
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits_a, _ = cvp.extract_piecewise_batch(bags, axis)
        crits_c, _ = cvp.extract_piecewise_batch(bags, axis,
                                                 cosh_c2=c2_cad,
                                                 ramp_gain=k_cad)
    gated = {c.bag_name for c in crits_a}
    by_bag = {b.name: b for b in bags}
    res = {'cosh': {c.bag_name: c.onset_moment for c in crits_a},
           'cosh_cad': {c.bag_name: c.onset_moment for c in crits_c}}
    res.update({m: {} for m in METHODS if m not in res})
    for name in sorted(gated):
        bag = by_bag[name]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, _ = cvp.extract_piecewise(bag, axis, model='cosh',
                                                cosh_c2=None, ramp_gain=None)
            res['nls'][name] = crit.onset_moment
        except Exception:
            pass
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                base, i0, i1, win, guess, direction = _window(bag, axis)
            for m in CLASSIC:
                jj = classic_onset_index(m, base.omega[win], guess, direction)
                res[m][name] = float(base.moment[win][jj])
        except Exception:
            pass
    for name in sorted(gated):
        dirn = 'pos' if name.startswith('pos') else 'neg'
        for m in METHODS:
            if name in res[m]:
                agg[key][m][dirn].append(res[m][name])
    print(f"done {key[0]}/{key[1]}", flush=True)

LABEL = {'cosh': 'COSH (cal)', 'cosh_cad': 'COSH (CAD)', 'nls': 'NLS',
         'pelt_normal': 'CPD_N', 'pelt_rbf': 'CPD_R', 'cusum': 'CUSUM'}
print(f"\n{'case':8} {'ax':3} {'method':11} {'M_ff [mN.m]':>18} "
      f"{'offset [mm]':>13} {'truth':>7} {'err':>7}")
print('-' * 74)
err_by_method = defaultdict(list)
table = {}
for key in sorted(agg):
    w = MASS_KG[key[0]] * G
    for m in METHODS:
        g = agg[key][m]
        if 'pos' not in g or 'neg' not in g or len(g['pos']) < 2:
            continue
        p, n = np.array(g['pos']), np.array(g['neg'])
        mff = 0.5 * (p.mean() + n.mean())
        var = 0.25 * (p.var(ddof=1) / len(p) + n.var(ddof=1) / len(n))
        dfw = var ** 2 / (
            (0.25 * p.var(ddof=1) / len(p)) ** 2 / (len(p) - 1)
            + (0.25 * n.var(ddof=1) / len(n)) ** 2 / (len(n) - 1))
        ci = stats.t.ppf(0.975, dfw) * np.sqrt(var)
        off = SIGN[key[1]] * 1e3 * mff / w
        e = off - TRUTH_MM[key]
        err_by_method[m].append(e)
        table[(key, m)] = (1e3 * mff, 1e3 * ci, off, 1e3 * ci / w, e)
        print(f"{key[0]:8} {key[1]:3} {LABEL[m]:11} "
              f"{1e3*mff:+10.1f} +-{1e3*ci:5.1f} "
              f"{off:+9.2f} +-{1e3*ci/w:4.2f} {TRUTH_MM[key]:+7.2f} {e:+7.2f}")
    print()
print('-' * 74)
print(f"{'method':13}{'RMS [mm]':>10}{'max|e|':>9}{'mean':>8}"
      f"{'RMS [mN.m]':>12}")
for m in METHODS:
    e = np.array(err_by_method[m])
    if not len(e):
        continue
    print(f"{LABEL[m]:13}{np.sqrt((e**2).mean()):10.2f}{np.abs(e).max():9.2f}"
          f"{e.mean():+8.2f}{np.sqrt((e**2).mean())*31.59:12.1f}")
import pickle
with open(Path(__file__).resolve().parent / 'mff_table.pkl', 'wb') as f:
    pickle.dump(table, f)
