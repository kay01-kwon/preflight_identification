#!/usr/bin/env python3
"""Does the error budget's ground-effect entry depend on the GE model?

The fidelity argument runs: the trajectory belongs to the cosh family,
every perturbation is bounded in moment units, and the bounds are small
enough not to matter.  One entry of that budget is evaluated along the
image-superposition model -- so a reader may object that the budget
assumes what it should prove, and that the whole argument collapses if
the model's attitude dependence is wrong.

It does not, and the reason is structural rather than numerical.

The estimator carries three per-run degrees of freedom, {1, tau,
dphi}, and rho is what survives projection out of that span.  Write the
ground-effect moment along the trajectory as

    dM_GE(phi) = dM_GE(0) + beta * dphi + gamma * dphi^2 + ...

The level is in the span.  The ENTIRE linear attitude dependence is in
the span.  So beta -- the quantity the dynamic inversion could not
measure -- contributes exactly zero to rho, whatever its value.  Only
the curvature gamma and above reach the onset at all.

This script demonstrates that as an identity and then asks the question
that actually matters: how large would the CURVATURE have to be before
the ground-effect channel moves the identified CoM offset by 0.1 mm,
against a validation RMS of 1.64 mm?

Each run's modelled GE trace is decomposed on {1, dphi, dphi^2}, and
the gradient and curvature terms are separately amplified by a factor k
before being pushed through the same Duhamel propagation and onset
linearisation the budget uses.

Usage: PYTHONPATH=<stubs> python analysis/ge_budget_robustness.py [root]
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from error_budget import (out_of_span, duhamel, ge_moment, LP, W_DEFAULT)
from utils.extractor import load_excitation_dataset
from utils import math_tools
from critical_value_getter_piecewise import (
    extract_piecewise_batch, detect_excitation_window, prepare_signals,
    estimate_rig_constants, MOMENT_CAP)

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
KS = (1.0, 10.0, 100.0, 1000.0)

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        c2, k_gain = estimate_rig_constants(bags, axis)
        crits, _ = extract_piecewise_batch(bags, axis, cosh_c2=c2,
                                           ramp_gain=k_gain)
    j_p = (1.0 / k_gain) / c2 ** 2
    by = {b.name: b for b in bags}
    for crit in crits:
        bag = by[crit.bag_name]
        sig = prepare_signals(bag, axis)
        i0, i1 = detect_excitation_window(crit.moment,
                                          moment_cap=MOMENT_CAP.get(axis))
        j = crit.onset_idx
        roll, pitch = math_tools.quaternion_to_euler_vectorized(
            bag.odom.quaternion)
        phi_all = roll if axis == 'x' else pitch
        n = min(len(phi_all), len(crit.t))
        i1 = min(i1, n - 1)
        if i1 - j < 10:
            continue
        sl = slice(j, i1 + 1)
        tau = crit.t[sl] - crit.t[j]
        dphi = phi_all[sl] - phi_all[j]
        ge = ge_moment(bag, sig, axis, n,
                       crit.bag_name.startswith('pos'), window=sl)
        if ge is None:
            continue
        ge = ge[sl]
        basis = np.vstack([np.ones_like(tau), tau, dphi]).T

        # decompose the modelled GE trace into level / gradient /
        # curvature in the tilt excursion
        P = np.vstack([np.ones_like(dphi), dphi, dphi ** 2]).T
        coef, *_ = np.linalg.lstsq(P, ge, rcond=None)
        lin, quad = coef[1] * dphi, coef[2] * dphi ** 2

        sh = np.sinh(np.clip(c2 * tau, 0, 30))
        denom = k_gain * c2 * float(sh @ sh)

        def d_mcrit(trace):
            rho = out_of_span(basis, trace)
            return -float(duhamel(tau, rho, c2, j_p) @ sh) / denom

        rec = dict(base=1e3 * d_mcrit(ge))
        for k in KS:
            rec[f'grad{k:g}'] = 1e3 * d_mcrit(ge + (k - 1.0) * lin)
            rec[f'curv{k:g}'] = 1e3 * d_mcrit(ge + (k - 1.0) * quad)
        rows.append(rec)
    print(f"  {d.parent.name}/{d.name}: {len(rows)} runs", flush=True)

W = W_DEFAULT
print(f"\n{len(rows)} runs;  amplifying the modelled GE trace's own "
      f"gradient / curvature\n")
print(f"{'multiplier k':>14}{'|dM_crit| max':>16}{'CoM offset max':>17}")
print(f"{'':14}{'[mN.m]':>16}{'[mm]':>17}")
base = np.abs([r['base'] for r in rows]).max()
print(f"{'1 (as is)':>14}{base:16.4f}{base / W:17.5f}")
print("\n  gradient beta scaled:")
for k in KS:
    v = np.abs([r[f'grad{k:g}'] for r in rows]).max()
    print(f"{k:14.0f}{v:16.4f}{v / W:17.5f}")
print("\n  curvature gamma scaled:")
for k in KS:
    v = np.abs([r[f'curv{k:g}'] for r in rows]).max()
    print(f"{k:14.0f}{v:16.4f}{v / W:17.5f}")

g = np.abs([r['curv1'] for r in rows]).max()
if g > 0:
    need = 0.1 * W / g
    print(f"\nThe gradient is absorbed exactly: scaling beta by 1000 leaves")
    print(f"|dM_crit| unchanged, because beta*dphi lies in the span the")
    print(f"estimator already fits.  Only the curvature reaches the onset,")
    print(f"and it would have to be {need:.0f}x the modelled value to move")
    print(f"the identified offset by 0.1 mm -- against a validation RMS of")
    print(f"1.64 mm.")
