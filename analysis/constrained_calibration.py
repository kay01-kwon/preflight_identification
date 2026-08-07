#!/usr/bin/env python3
"""Do physically CONSISTENT constants reproduce the identification?

The per-dataset calibration estimates (C2, K) independently for each of
the ten case/axis datasets -- twenty free numbers.  Read literally,
K = 1/(W z_CoM) then implies z_CoM between 0.06 and 0.53 m, and the two
axes of the SAME vehicle disagree by up to 8x.  That cannot be right:
within one case the roll and pitch datasets are the same airframe, so
z_CoM must be common.  The individual constants are evidently sliding
along the flat (C2, K) ridge, whose well-determined direction is
J_P = 1/(K C2^2), not 1/K.

This script asks whether the method survives being held to the physics.
Reparametrise by quantities that MUST be shared:

    W z_CoM      one value for the whole vehicle          (1 number)
    J_P,roll     one value for every roll dataset         (1 number)
    J_P,pitch    one value for every pitch dataset        (1 number)

and derive the per-dataset constants from them,

    C2 = sqrt(W z_CoM / J_P,axis),      K = 1/(W z_CoM).

Twenty free numbers become three.  The objective is unchanged -- the
ramp-invariance score of estimate_rig_constants, summed over datasets --
and, because the roll datasets depend only on J_P,roll and the pitch
datasets only on J_P,pitch, the inner search separates: for each
W z_CoM the two inertias are optimised independently.

The identification is then rerun with the constrained constants and the
delivered CoM offsets are compared against the load-cell truth, next to
the free-calibration baseline.  If the accuracy survives, the constants
are no longer a fitted curve family with physical labels attached; if it
does not, the honest conclusion is that (C2, K) are effective constants
and 1/K must not be reported as W z_CoM.

Usage
-----
PYTHONPATH=<stubs> python analysis/constrained_calibration.py [outdir]
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}
OFF_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

# search boxes: z_CoM in [0.15, 0.35] m at W ~ 31.6 N; inertias bracket the
# parallel-axis estimate m (z^2 + l_p^2) + J_CoM
WZ_GRID = np.linspace(4.7, 11.1, 9)
J_GRID = {'x': np.linspace(0.15, 0.42, 10), 'y': np.linspace(0.07, 0.32, 10)}
STRIDE = 3


def prepare(bags, axis):
    """Cache the per-bag window signals the score needs."""
    out = []
    for bag in bags:
        try:
            sig = cvp.prepare_signals(bag, axis)
        except Exception:
            continue
        i0, i1 = cvp.detect_excitation_window(sig['moment'])
        win = slice(i0, i1 + 1)
        t, om, m = sig['t'][win], sig['omega'][win], sig['moment'][win]
        if len(t) < 24:
            continue
        side = 'neg' if bag.name.lower().startswith('neg') else 'pos'
        out.append((side, t - t[0], om, m, float(np.polyfit(t, m, 1)[0])))
    return out


def score(prepared, c2, k):
    """Ramp-invariance score: sum over tip directions of CV(|M_crit|)."""
    groups = {}
    for side, t, om, m, m_dot in prepared:
        pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                                c2_fixed=float(c2), moment_floor=0.0,
                                ramp_gain=float(k), ramp_rate=m_dot,
                                step_s=STRIDE * float(np.median(np.diff(t))))
        groups.setdefault(side, []).append(float(m[pw['onset_idx']]))
    s = 0.0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        mu = abs(float(np.mean(vals)))
        s += float(np.std(vals)) / mu if mu > 1e-9 else np.inf
    return s


def main():
    dirs = sorted(ROOT.glob('case_*/M[xy]'))
    data, free = {}, {}
    print("loading and running the FREE per-dataset calibration ...")
    for d in dirs:
        axis = 'x' if d.name == 'Mx' else 'y'
        key = (d.parent.name, d.name)
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_excitation_dataset(d)
            c2, k = cvp.estimate_rig_constants(bags, axis)
        data[key] = (axis, bags, prepare(bags, axis))
        free[key] = (c2, k)
        wz, jp = 1.0 / k, 1.0 / (k * c2 ** 2)
        print(f"  {key[0]}/{key[1]}: C2={c2:.3f} K={k:.4f} -> "
              f"W z_CoM={wz:6.2f} N.m (z={wz / (MASS_KG[key[0]] * G):.3f} m), "
              f"J_P={jp:.3f} kg.m^2")

    # ------------------------------------------------ constrained search
    print("\nconstrained search over (W z_CoM, J_roll, J_pitch) ...")
    best = (np.inf, None, None, None)
    trace = []
    for wz in WZ_GRID:
        k = 1.0 / wz
        tot, jbest = 0.0, {}
        for ax in ('x', 'y'):
            keys = [q for q in data if data[q][0] == ax]
            sub = np.inf
            for jp in J_GRID[ax]:
                c2 = float(np.sqrt(wz / jp))
                s = sum(score(data[q][2], c2, k) for q in keys)
                if s < sub:
                    sub, jbest[ax] = s, float(jp)
            tot += sub
        trace.append((float(wz), jbest['x'], jbest['y'], tot))
        print(f"  W z_CoM={wz:5.2f} (z={wz / 31.59:.3f} m)  "
              f"J_roll={jbest['x']:.3f}  J_pitch={jbest['y']:.3f}  "
              f"score={tot:.4f}")
        if tot < best[0]:
            best = (tot, float(wz), jbest['x'], jbest['y'])
    _, wz_b, jr_b, jp_b = best
    print(f"\n  best: W z_CoM={wz_b:.2f} N.m (z_CoM={wz_b / 31.59:.3f} m), "
          f"J_roll={jr_b:.3f}, J_pitch={jp_b:.3f} kg.m^2")

    con = {}
    for key, (axis, _, _) in data.items():
        jp = jr_b if axis == 'x' else jp_b
        con[key] = (float(np.sqrt(wz_b / jp)), 1.0 / wz_b)

    # ------------------------------------------- rerun the identification
    def deliver(consts, tag):
        rows = []
        for key, (axis, bags, _) in data.items():
            c2, k = consts[key]
            with contextlib.redirect_stdout(io.StringIO()):
                crits, _ = cvp.extract_piecewise_batch(
                    bags, axis, cosh_c2=c2, ramp_gain=k)
            pos = [c.onset_moment for c in crits
                   if c.bag_name.startswith('pos')]
            neg = [c.onset_moment for c in crits
                   if c.bag_name.startswith('neg')]
            if not pos or not neg:
                continue
            w = MASS_KG[key[0]] * G
            m_ff = OFF_SIGN[key[1]] * 0.5 * (np.mean(pos) + np.mean(neg))
            off = 1e3 * m_ff / w
            truth = OFF_MM[key]
            rows.append(dict(case=key[0], axis=key[1], tag=tag,
                             c2=c2, k=k, off_mm=off, truth_mm=truth,
                             err_mm=off - truth,
                             n_pos=len(pos), n_neg=len(neg)))
        return rows

    print("\nrerunning the identification ...")
    rows = deliver(free, 'free') + deliver(con, 'constrained')

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'constrained_calibration.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'case':9} {'ax':3} {'truth':>8} | "
          f"{'free off':>9} {'err':>7} | {'constr off':>11} {'err':>7}")
    print('-' * 62)
    for key in sorted({(r['case'], r['axis']) for r in rows}):
        fr = next(r for r in rows if (r['case'], r['axis']) == key
                  and r['tag'] == 'free')
        co = next(r for r in rows if (r['case'], r['axis']) == key
                  and r['tag'] == 'constrained')
        print(f"{key[0]:9} {key[1]:3} {fr['truth_mm']:8.2f} | "
              f"{fr['off_mm']:9.2f} {fr['err_mm']:7.2f} | "
              f"{co['off_mm']:11.2f} {co['err_mm']:7.2f}")
    print('-' * 62)
    for tag in ('free', 'constrained'):
        e = np.array([r['err_mm'] for r in rows if r['tag'] == tag])
        print(f"{tag:12}  RMS {np.sqrt(np.mean(e ** 2)):5.2f} mm   "
              f"max|e| {np.abs(e).max():5.2f} mm   "
              f"mean {e.mean():+5.2f} mm   (n={len(e)})")
    print(f"\nfree calibration: 20 free numbers; constrained: 3 "
          f"(W z_CoM, J_roll, J_pitch)")
    print(f"Tables -> {OUT}")


if __name__ == '__main__':
    main()
