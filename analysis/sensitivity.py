#!/usr/bin/env python3
"""
Sensitivity of the identified quantities to the rig constants (C₂, K)
=====================================================================
The closed-form onset uses two rig constants: the instability rate C₂ = √d and
the ramp gain K = 1/(W·z_CoM). Their selection objective (ramp-rate invariance
of M_crit) is flat along a ridge in (C₂, K), so the constants are individually
only loosely determined. What matters is how much the DELIVERABLES — M_crit,
the pivot-free moment offset M_ff, the pivot-based CoM offset and the mass —
move when the constants move. This script quantifies exactly that:

  * one-at-a-time (OAT): scale C₂ and K by 0.8…1.2 around the optimum and
    report every deliverable, with % changes;
  * a full 2-D grid over the physically plausible ranges.

Outputs CSVs (and an optional M_ff / CoM heatmap) ready for a paper table.

Usage
-----
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y --save-fig
python analysis/sensitivity.py DataSet/exp/case_05/My --axis y --c2 6.4 --k 0.19
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import (
    extract_piecewise,
    estimate_rig_constants,
    detect_excitation_window,
    cosh_onset_fit,
    estimate_pivot_from_mocap,
    compute_mass_and_offset,
)


def prepare(bags, axis):
    """Cache per-bag windowed signals + a base CriticalValueResult to clone."""
    runs = []
    for bag in bags:
        # one full extraction only to obtain the aligned signal arrays; its
        # onset fields are overwritten per (C₂, K) below
        base, _ = extract_piecewise(bag, axis, model='piecewise')
        i0, i1 = detect_excitation_window(base.moment)
        win = slice(i0, i1 + 1)
        t, om, M = base.t[win], base.omega[win], base.moment[win]
        runs.append(dict(
            bag=bag, base=base, i0=i0,
            t=t, omega=om, moment=M,
            m_dot=float(np.polyfit(t, M, 1)[0]),
            side='neg' if bag.name.lower().startswith('neg') else 'pos',
        ))
    return runs


def evaluate(runs, axis, c2, k, stride=2, known_mass=None):
    """All deliverables for one (C₂, K): per-direction M_crit, M_ff, CoM, mass."""
    crits, pivots, groups = [], [], {'neg': [], 'pos': []}
    for r in runs:
        pw = cosh_onset_fit(r['t'], r['omega'], r['moment'], onset_guess=None,
                            c2_fixed=float(c2), moment_floor=0.0,
                            ramp_gain=float(k), ramp_rate=r['m_dot'],
                            step_s=stride * float(np.median(np.diff(r['t']))))
        j = r['i0'] + pw['onset_idx']
        base = r['base']
        crit = replace(base, onset_idx=j,
                       onset_time=float(base.t[j]),
                       onset_thrust=float(base.f_col[j]),
                       onset_moment=float(base.moment[j]),
                       onset_omega=float(base.omega[j]))
        crits.append(crit)
        pivots.append(estimate_pivot_from_mocap(r['bag'], crit.onset_time, axis))
        groups[r['side']].append(crit.onset_moment)

    est = compute_mass_and_offset(crits, pivots, axis, known_mass=known_mass)
    m_neg = float(np.mean(groups['neg'])) if groups['neg'] else np.nan
    m_pos = float(np.mean(groups['pos'])) if groups['pos'] else np.nan
    return dict(
        c2=float(c2), k=float(k),
        M_neg=m_neg, M_pos=m_pos,
        M_neg_std=float(np.std(groups['neg'])) if groups['neg'] else np.nan,
        M_pos_std=float(np.std(groups['pos'])) if groups['pos'] else np.nan,
        M_ff=est['pair3_ff_mean'],
        com_mm=est['pair3_offset_mean'] * 1e3,
        mass=est['pair3_mass_mean'],
    )


def pct(v, ref):
    return (v - ref) / abs(ref) * 100.0 if ref and not np.isnan(ref) else np.nan


def main():
    p = argparse.ArgumentParser(
        description="Sensitivity of the deliverables to the rig constants.")
    p.add_argument('data_dir')
    p.add_argument('--axis', choices=['x', 'y'], required=True)
    p.add_argument('--c2', type=float, default=None,
                   help="reference C₂ (default: estimated optimum)")
    p.add_argument('--k', type=float, default=None,
                   help="reference K (default: estimated optimum)")
    p.add_argument('--scales', type=float, nargs='+',
                   default=[0.8, 0.9, 1.0, 1.1, 1.2],
                   help="OAT scale factors around the reference")
    p.add_argument('--c2-grid', type=float, nargs=3, default=[3.0, 8.0, 1.0],
                   metavar=('MIN', 'MAX', 'STEP'))
    p.add_argument('--k-grid', type=float, nargs=3, default=[0.10, 0.50, 0.10],
                   metavar=('MIN', 'MAX', 'STEP'))
    p.add_argument('--stride', type=int, default=2,
                   help="onset sweep stride in samples (2 ≈ 0.02 s at 100 Hz)")
    p.add_argument('--mass', type=float, default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--save-fig', action='store_true')
    args = p.parse_args()

    out = Path(args.output_dir) if args.output_dir else Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    bags = load_excitation_dataset(Path(args.data_dir))
    runs = prepare(bags, args.axis)

    c2_ref, k_ref = args.c2, args.k
    if c2_ref is None or k_ref is None:
        c2_est, k_est = estimate_rig_constants(bags, args.axis)
        c2_ref = c2_ref or c2_est
        k_ref = k_ref or k_est
    print(f"\nReference: C₂ = {c2_ref:.3f} rad/s, K = {k_ref:.4f}")

    ref = evaluate(runs, args.axis, c2_ref, k_ref,
                   stride=args.stride, known_mass=args.mass)
    print(f"  M_neg = {ref['M_neg']:+.4f} ± {ref['M_neg_std']:.4f}   "
          f"M_pos = {ref['M_pos']:+.4f} ± {ref['M_pos_std']:.4f}")
    print(f"  M_ff = {ref['M_ff']:+.4f} N·m   CoM = {ref['com_mm']:+.2f} mm   "
          f"mass = {ref['mass']:.3f} kg")

    # ── one-at-a-time ─────────────────────────────────────────
    oat_rows = []
    print(f"\n── OAT sensitivity (scales {args.scales}) ──")
    hdr = (f"{'param':6} {'scale':>5} {'value':>7} | {'M_ff':>8} {'Δ%':>6} | "
           f"{'CoM[mm]':>8} {'Δ%':>6} | {'mass':>6} {'Δ%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for pname in ('C2', 'K'):
        for s in args.scales:
            c2 = c2_ref * s if pname == 'C2' else c2_ref
            k = k_ref * s if pname == 'K' else k_ref
            r = (ref if s == 1.0 else
                 evaluate(runs, args.axis, c2, k,
                          stride=args.stride, known_mass=args.mass))
            row = dict(param=pname, scale=s, **r,
                       dMff_pct=pct(r['M_ff'], ref['M_ff']),
                       dcom_pct=pct(r['com_mm'], ref['com_mm']),
                       dmass_pct=pct(r['mass'], ref['mass']))
            oat_rows.append(row)
            val = c2 if pname == 'C2' else k
            print(f"{pname:6} {s:5.2f} {val:7.3f} | {r['M_ff']:+8.4f} "
                  f"{row['dMff_pct']:+6.1f} | {r['com_mm']:+8.2f} "
                  f"{row['dcom_pct']:+6.1f} | {r['mass']:6.3f} "
                  f"{row['dmass_pct']:+6.1f}")

    fp = out / f"sensitivity_oat_{args.axis}.csv"
    with open(fp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(oat_rows[0].keys()))
        w.writeheader()
        w.writerows(oat_rows)
    print(f"OAT table → {fp}")

    # ── 2-D grid ──────────────────────────────────────────────
    c2s = np.arange(args.c2_grid[0], args.c2_grid[1] + 1e-9, args.c2_grid[2])
    ks = np.arange(args.k_grid[0], args.k_grid[1] + 1e-9, args.k_grid[2])
    print(f"\n── 2-D grid: C₂ ∈ {[round(float(v), 2) for v in c2s]}, "
          f"K ∈ {[round(float(v), 2) for v in ks]} ──")
    grid_rows = []
    for c2 in c2s:
        for k in ks:
            r = evaluate(runs, args.axis, c2, k,
                         stride=args.stride, known_mass=args.mass)
            r['dMff_pct'] = pct(r['M_ff'], ref['M_ff'])
            r['dcom_pct'] = pct(r['com_mm'], ref['com_mm'])
            grid_rows.append(r)

    ff = np.array([r['M_ff'] for r in grid_rows])
    com = np.array([r['com_mm'] for r in grid_rows])
    print(f"  M_ff  range: {ff.min():+.4f} … {ff.max():+.4f} N·m "
          f"(span {100 * (ff.max() - ff.min()) / abs(ref['M_ff']):.1f}% of ref)")
    print(f"  CoM   range: {com.min():+.2f} … {com.max():+.2f} mm "
          f"(span {100 * (com.max() - com.min()) / abs(ref['com_mm']):.1f}% of ref)")

    fp = out / f"sensitivity_grid_{args.axis}.csv"
    with open(fp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)
    print(f"Grid table → {fp}")

    if args.save_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        FF = ff.reshape(len(c2s), len(ks))
        CM = com.reshape(len(c2s), len(ks))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, Z, title in ((axes[0], FF, r'$M_{ff}$ [N·m]'),
                             (axes[1], CM, 'CoM offset [mm]')):
            im = ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
                           extent=[ks[0], ks[-1], c2s[0], c2s[-1]])
            ax.plot(k_ref, c2_ref, 'r*', ms=14, label='reference')
            ax.set_xlabel('K'); ax.set_ylabel(r'$C_2$ [rad/s]')
            ax.set_title(title); ax.legend(loc='upper right')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Deliverable sensitivity to the rig constants "
                     f"(axis {args.axis})")
        fig.tight_layout()
        fp = out / f"sensitivity_grid_{args.axis}.png"
        fig.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"Figure → {fp}")


if __name__ == '__main__':
    main()
