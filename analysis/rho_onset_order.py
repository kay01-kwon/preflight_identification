#!/usr/bin/env python3
"""Does rho really vanish at the anchor, and to what order?

Two framings coexist in the section:
  (1) ANCHORED  -- absorb the Taylor coefficients exactly at tau = 0.
      The remainder vanishes at the anchor by construction.
  (2) PROJECTED -- remove the least-squares projection onto
      span{1, tau, dphi} over the whole window.  This is what the
      estimator actually does; the residual is smaller in L2 but need
      not vanish pointwise at tau = 0.

Report, per channel: |rho(0)| / max|rho| in both framings, and the
fitted leading power of the anchored residual over the first third of
the window.

Measured (case_01/Mx + case_03/My): the projected residual is NOT small
at the anchor (0.17-0.52 of the window peak in median, up to 1.0), and
the fitted leading power is 1.2-1.7 rather than the theoretical 4
(bilinear GE) or 6 (gravity curvature) -- near the anchor the recorded
residual is noise-dominated.  The onset-order argument of
Sec. fid-budget is therefore a statement about the coherent modelling
error along the nominal trajectory, not one the record resolves; the
error budget itself forwards the projected peak and does not rely on
it.

Usage
-----
python analysis/rho_onset_order.py DataSet/exp/case_01/Mx [more dirs...]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import contextlib
import io
from pathlib import Path

import numpy as np

from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    extract_piecewise_batch, detect_excitation_window, estimate_rig_constants,
    prepare_signals, MOMENT_CAP)
from error_budget import ge_moment, out_of_span, LP, OFFSET_MARGIN, BETA_M

W, Z = 31.59, 0.30
res = {c: {'anc': [], 'prj': [], 'pow': []}
       for c in ('gravity', 'ge', 'bilinear')}

for d in [Path(p) for p in sys.argv[1:]]:
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2, k = estimate_rig_constants(bags, axis)
        crits, _ = extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=k)
    arm = LP[axis] + OFFSET_MARGIN
    bag_by = {b.name: b for b in bags}
    for crit in crits:
        bag = bag_by[crit.bag_name]
        sig = prepare_signals(bag, axis)
        _, i1 = detect_excitation_window(crit.moment,
                                         moment_cap=MOMENT_CAP.get(axis))
        j = crit.onset_idx
        roll, pitch = math_tools.quaternion_to_euler_vectorized(
            bag.odom.quaternion)
        ph_all = roll if axis == 'x' else pitch
        n = min(len(ph_all), len(crit.t))
        i1 = min(i1, n - 1)
        if i1 - j < 15:
            continue
        sl = slice(j, i1 + 1)
        tau = crit.t[sl] - crit.t[j]
        ph = ph_all[sl]
        dphi = ph - ph[0]
        mom = crit.moment[sl]
        A = np.vstack([np.ones_like(tau), tau, dphi]).T
        ge = ge_moment(bag, sig, axis, n,
                       crit.bag_name.startswith('pos'), window=sl)
        raw = {
            'gravity': -W * arm * np.cos(ph) + W * Z * np.sin(ph),
            'ge': ge[sl] if ge is not None else None,
            'bilinear': BETA_M * (mom - mom[0]) * dphi,
        }
        for name, y in raw.items():
            if y is None:
                continue
            # (1) anchored: subtract the tangent plane AT tau = 0
            g = np.gradient(y, tau)
            gp = np.gradient(dphi, tau)
            # d/dtau = partial_tau + partial_phi * dphi/dtau ; separate the
            # dphi direction by using the two-point slopes at the anchor
            anc = y - y[0] - g[0] * tau      # {1, tau} anchored at the onset
            prj = out_of_span(A, y)
            for tag, r in (('anc', anc), ('prj', prj)):
                m = np.max(np.abs(r))
                if m > 1e-12:
                    res[name][tag].append(abs(r[0]) / m)
            # leading power over the first third
            k3 = max(6, len(tau) // 3)
            msk = (tau[:k3] > 1e-3) & (np.abs(anc[:k3]) > 1e-12)
            if msk.sum() >= 5:
                p = np.polyfit(np.log(tau[:k3][msk]),
                               np.log(np.abs(anc[:k3][msk])), 1)[0]
                res[name]['pow'].append(p)

print(f"{'channel':10} {'|rho(0)|/max, anchored':>24} "
      f"{'|rho(0)|/max, projected':>25} {'leading power (anchored)':>26}")
for name, v in res.items():
    if not v['anc']:
        continue
    a, p, q = (np.array(v['anc']), np.array(v['prj']), np.array(v['pow']))
    print(f"{name:10} {np.median(a):12.2e} (max {a.max():.2e}) "
          f"{np.median(p):11.3f} (max {p.max():.3f}) "
          f"{np.median(q):15.2f}  [{np.percentile(q,10):.2f}, "
          f"{np.percentile(q,90):.2f}]")
