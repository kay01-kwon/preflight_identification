#!/usr/bin/env python3
"""Deliverable under the parallel-axis parametrisation, as z_CoM varies.

The per-dataset calibration returns pivot inertias that violate the
parallel-axis lower bound J_P >= m (z_CoM^2 + l_p^2) as soon as
z_CoM >= 0.2 m -- by 24% for pitch at 0.20 m and by 59% at 0.30 m.  The
fix is to stop treating J_P as free.  The CAD inertias (manuscript
Table 5) and the landing-gear geometry are independent measurements, so

    J_P,axis = J_CAD,axis + m (z_CoM^2 + l_p,axis^2),
    C2 = sqrt(W z_CoM / J_P),      K = 1/(W z_CoM)

leaves z_CoM as the single free number for the whole experiment.

Two facts make this work.  First, C2 is stationary in z_CoM near the
plausible range: dC2/dz = 0 at z = sqrt((J_CAD + m l_p^2)/m), which is
0.188 m for roll and 0.167 m for pitch, so C2 varies by about 5% over
z in [0.14, 0.30].  Second, all the remaining z-dependence sits in
K = 1/(W z_CoM), and K is the low-sensitivity direction of the onset
search.  The delivered CoM offset is therefore nearly independent of
which z_CoM is assumed:

    z_CoM   J_roll  J_pitch   C2_r   C2_p       K |    RMS  max|e|   mean
     0.14    0.177    0.153   4.99   5.38  0.2261 |   1.67    3.40  -0.94
     0.18    0.219    0.194   5.10   5.42  0.1759 |   1.67    3.42  -0.92
     0.20    0.243    0.218   5.10   5.38  0.1583 |   1.66    3.42  -0.91
     0.25    0.315    0.291   5.00   5.21  0.1266 |   1.64    3.42  -0.87
     0.30    0.404    0.379   4.84   5.00  0.1055 |   1.62    3.40  -0.82
    free per-dataset calibration (20 numbers):        1.64    3.36  -0.95

So one physically admissible number reproduces twenty fitted ones, and
z_CoM >= 0.2 m -- the range the geometry actually supports -- is if
anything slightly better than the calibration score's own optimum
(z = 0.14 m).  The score and the deliverable are sensitive to different
directions, which is the same conclusion the (C2, K) ridge analysis
reaches.

Usage
-----
PYTHONPATH=<stubs> python analysis/pa_zscan.py
"""

import contextlib, io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/user/preflight_identification')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

ROOT = Path('/home/user/preflight_identification/DataSet/exp')
G = 9.81
MASS = {'case_01':3.066,'case_02':3.220,'case_03':3.220,'case_04':3.220,'case_05':3.220}
OFF  = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
        ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
        ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
        ('case_05','My'):-10.89}
SGN  = {'Mx':+1.0,'My':-1.0}
JCAD = {'x':0.051085,'y':0.050564}; LP = {'x':0.140,'y':0.110}

data = {}
for d in sorted(ROOT.glob('case_*/M[xy]')):
    with contextlib.redirect_stdout(io.StringIO()):
        data[(d.parent.name, d.name)] = ('x' if d.name=='Mx' else 'y',
                                         load_excitation_dataset(d))
print(f"{'z_CoM':>7} {'J_roll':>8} {'J_pitch':>8} {'C2_r':>6} {'C2_p':>6} "
      f"{'K':>7} | {'RMS':>6} {'max|e|':>7} {'mean':>6}")
print('-'*72)
for z in (0.14, 0.18, 0.20, 0.25, 0.30):
    errs = []
    for key,(axis,bags) in data.items():
        mass = MASS[key[0]]; w = mass*G
        jp = JCAD[axis] + mass*(z*z + LP[axis]**2)
        c2, k = float(np.sqrt(w*z/jp)), 1.0/(w*z)
        with contextlib.redirect_stdout(io.StringIO()):
            crits,_ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=k)
        p=[c.onset_moment for c in crits if c.bag_name.startswith('pos')]
        n=[c.onset_moment for c in crits if c.bag_name.startswith('neg')]
        if not p or not n: continue
        off = 1e3*SGN[key[1]]*0.5*(np.mean(p)+np.mean(n))/w
        errs.append(off - OFF[key])
    e = np.array(errs)
    jr = JCAD['x']+3.220*(z*z+LP['x']**2); jq = JCAD['y']+3.220*(z*z+LP['y']**2)
    print(f"{z:7.2f} {jr:8.3f} {jq:8.3f} {np.sqrt(31.59*z/jr):6.2f} "
          f"{np.sqrt(31.59*z/jq):6.2f} {1/(31.59*z):7.4f} | "
          f"{np.sqrt(np.mean(e**2)):6.2f} {np.abs(e).max():7.2f} {e.mean():6.2f}")
print('-'*72)
print("free per-dataset calibration (20 numbers):  RMS 1.64   max 3.36   mean -0.95")
