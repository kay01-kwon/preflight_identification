"""Consistency, not physical fidelity: CV of M_crit across the ramp sweep.

The calibration's stated purpose is to make the identified threshold
reproducible across ramp rates and tip directions -- not to measure
W z_CoM.  This tabulates the quantity that purpose names: the coefficient
of variation of |M_crit| within each tip direction, per dataset, for
(a) the free per-dataset calibration, (b) constants taken
entirely from CAD, and (c) K deliberately mis-specified about the CAD
point.
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import prepare, MASS_KG, G, ROOT, STRIDE

# CAD reference: z_CoM = 0.256 m and the Table 5 CoM inertias fix J_P by
# the parallel-axis theorem, hence both constants, with nothing fitted.
Z_CAD = 0.256
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}

CAL = {('case_01','Mx'):(8.000,0.0600), ('case_01','My'):(3.875,0.4900),
       ('case_02','Mx'):(6.125,0.1100), ('case_02','My'):(4.500,0.5200),
       ('case_03','Mx'):(3.500,0.3000), ('case_03','My'):(3.875,0.3600),
       ('case_04','Mx'):(3.750,0.3000), ('case_04','My'):(5.625,0.2000),
       ('case_05','Mx'):(6.875,0.0900), ('case_05','My'):(6.625,0.1800)}


def onsets(prepared, c2, k):
    """|M_crit| per run, grouped by tip direction, plus the ramp rates."""
    g, rates = {}, []
    for side, t, om, m, m_dot in prepared:
        pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                c2_fixed=float(c2), moment_floor=0.0,
                                ramp_gain=float(k), ramp_rate=m_dot,
                                step_s=STRIDE * float(np.median(np.diff(t))))
        g.setdefault(side, []).append(float(m[pw['onset_idx']]))
        rates.append(abs(m_dot))
    return g, rates


def cv(g):
    out = []
    for v in g.values():
        if len(v) >= 2:
            out.append(float(np.std(v)) / abs(float(np.mean(v))))
    return out


data = {}
for d in sorted(ROOT.glob('case_*/M[xy]')):
    ax = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    data[(d.parent.name, d.name)] = (ax, prepare(bags, ax))

VARIANTS = [('free', None), ('CAD', 1.0),
            ('K x0.5', 0.5), ('K x2', 2.0)]
hdr = (f"{'case':<9}{'ax':<4}{'n':>3}{'Mdot range':>13}"
       + ''.join(f"{lab:>11}" for lab, _ in VARIANTS))
print("CV of |M_crit| within each tip direction  [%], mean of pos/neg\n")
print(hdr); print('-' * len(hdr))
acc = {lab: [] for lab, _ in VARIANTS}
for key in sorted(data):
    ax, prep = data[key]
    w = MASS_KG[key[0]] * G
    row, rates = [], None
    for lab, fac in VARIANTS:
        if fac is None:
            c2, k = CAL[key]
        else:
            j = J_CAD[ax] + MASS_KG[key[0]] * (Z_CAD ** 2 + LP[ax] ** 2)
            k = fac / (w * Z_CAD)
            c2 = float(np.sqrt(w * Z_CAD / j))
        g, rates = onsets(prep, c2, k)
        c = cv(g)
        row.append(100 * float(np.mean(c)) if c else float('nan'))
        acc[lab].append(row[-1])
    n = sum(len(v) for v in g.values())
    print(f"{key[0]:<9}{key[1]:<4}{n:3d}"
          f"{min(rates):6.2f}-{max(rates):<6.2f}"
          + ''.join(f"{r:11.1f}" for r in row))
print('-' * len(hdr))
print(f"{'mean':<16}{'':13}" + ''.join(
    f"{np.nanmean(acc[lab]):11.1f}" for lab, _ in VARIANTS))
print(f"{'worst':<16}{'':13}" + ''.join(
    f"{np.nanmax(acc[lab]):11.1f}" for lab, _ in VARIANTS))
