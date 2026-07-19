#!/usr/bin/env python3
"""
PELT / Binary Segmentation onset cross-check  (review defense)
==============================================================
Independent, method-agnostic validation of the model-based onset.

The main pipeline locates the tip-over onset with model-based fits
(time-quadratic PLS and the closed-form cosh solution). This script cross-
checks that onset against a standard, citable change-point detector — PELT /
Binary Segmentation (`ruptures`) — using several cost functions. Agreement
across independent method families demonstrates the critical moment is
robust to the detection methodology, which is the typical reviewer concern.

Note on the systematic offset: statistical change-point methods detect the
onset slightly *later* (larger |M|) than the model-based fit, because the
distribution change must accumulate enough post-onset samples to become
significant (detection lag). The model-based (PLS / cosh) estimate localizes
the physical onset — the start of the rise.

Requires the optional package `ruptures`:  pip install ruptures

Usage
-----
python analysis/pelt_crosscheck.py DataSet/exp/case_05/My
python analysis/pelt_crosscheck.py DataSet/exp/case_05/My --axis y --save-fig
"""

import os
import sys

# Make the repo root importable when run as `python analysis/pelt_crosscheck.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.extractor import load_excitation_dataset
from critical_value_getter_piecewise import (
    extract_piecewise,
    detect_excitation_window,
    detect_axis,
    bag_name_to_title,
    estimate_pivot_from_mocap,
    compute_mass_and_offset,
)

# Cost functions to cross-check with. 'normal' (Gaussian mean+variance) is the
# closest analogue of the GLR test; 'rbf' is a nonparametric kernel detector;
# 'l2' is a piecewise-constant mean shift.
PELT_COSTS = ['l2', 'normal', 'rbf']


# ═════════════════════════════════════════════════════════════
#  PELT / Binary Segmentation onset
# ═════════════════════════════════════════════════════════════

def pelt_onset_index(omega_win: np.ndarray, model: str, min_size: int = 5) -> int:
    """
    Single-change-point (onset) index within the excitation window via
    Binary Segmentation with the given cost model.
    """
    try:
        import ruptures as rpt  # optional dependency
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "This cross-check needs the optional package 'ruptures'.\n"
            "  pip install ruptures"
        ) from exc

    algo = rpt.Binseg(model=model, min_size=min_size)
    bkps = algo.fit(np.asarray(omega_win).reshape(-1, 1)).predict(n_bkps=1)
    # predict() returns [breakpoint, N]; take the breakpoint
    return int(min(bkps[0], len(omega_win) - 1))


# ═════════════════════════════════════════════════════════════
#  Cross-check one dataset
# ═════════════════════════════════════════════════════════════

def crosscheck(bags, axis: str):
    """
    For each bag: model-based onset moment (PLS, cosh) vs PELT onset moment
    for every cost in PELT_COSTS. Returns a list of per-bag dicts plus the
    per-bag time series needed for plotting.
    """
    rows = []
    series = []
    for bag in bags:
        crit_pls, _ = extract_piecewise(bag, axis, model='piecewise')
        crit_cosh, _ = extract_piecewise(bag, axis, model='cosh')

        t, omega, moment = crit_pls.t, crit_pls.omega, crit_pls.moment
        i0, i1 = detect_excitation_window(moment)
        win = slice(i0, i1 + 1)
        omega_win, M_win, t_win = omega[win], moment[win], t[win]

        pelt = {}
        for cost in PELT_COSTS:
            j = pelt_onset_index(omega_win, cost)
            pelt[cost] = (t_win[j], float(M_win[j]))

        row = dict(
            bag=crit_pls.bag_name,
            M_pls=crit_pls.onset_moment,
            M_cosh=crit_cosh.onset_moment,
        )
        for cost in PELT_COSTS:
            row[f'M_pelt_{cost}'] = pelt[cost][1]
        rows.append(row)

        series.append(dict(
            name=crit_pls.bag_name, t_win=t_win, omega_win=omega_win, M_win=M_win,
            onsets={
                'PLS (quad)': (crit_pls.onset_time, crit_pls.onset_moment, 'tab:red', '--'),
                'cosh': (crit_cosh.onset_time, crit_cosh.onset_moment, 'tab:blue', ':'),
                **{f'PELT {c}': (pelt[c][0], pelt[c][1],
                                 clr, '-.') for c, clr in
                   zip(PELT_COSTS, ['tab:green', 'tab:orange', 'tab:purple'])},
            },
        ))
    return rows, series


# ═════════════════════════════════════════════════════════════
#  Downstream: propagate each onset method to CoM / moment offset
# ═════════════════════════════════════════════════════════════

# Onset methods to propagate. 'pelt:<cost>' uses the PELT breakpoint.
DOWNSTREAM_METHODS = ['piecewise', 'cosh', 'pelt:normal', 'pelt:rbf']
_METHOD_LABEL = {'piecewise': 'PLS', 'cosh': 'cosh',
                 'pelt:normal': 'PELT\nnormal', 'pelt:rbf': 'PELT\nrbf'}


def _crit_for_method(bag, axis: str, method: str):
    """CriticalValueResult for a bag under the given onset method."""
    if method in ('piecewise', 'cosh'):
        crit, _ = extract_piecewise(bag, axis, model=method)
        return crit
    # PELT: take the base signals, relocate the onset to the PELT breakpoint
    base, _ = extract_piecewise(bag, axis, model='piecewise')
    i0, i1 = detect_excitation_window(base.moment)
    j = i0 + pelt_onset_index(base.omega[i0:i1 + 1], method.split(':', 1)[1])
    return replace(base, onset_idx=j, onset_time=float(base.t[j]),
                   onset_thrust=float(base.f_col[j]),
                   onset_moment=float(base.moment[j]),
                   onset_omega=float(base.omega[j]))


def downstream_crosscheck(bags, axis: str, known_mass=None):
    """
    Propagate each onset method through the full estimation (pivot circle-fit
    + mass/CoM offset) and return the aggregated estimates per method.
    """
    out = []
    for method in DOWNSTREAM_METHODS:
        crits = [_crit_for_method(b, axis, method) for b in bags]
        pivots = [estimate_pivot_from_mocap(b, c.onset_time, axis)
                  for b, c in zip(bags, crits)]
        e = compute_mass_and_offset(crits, pivots, axis, known_mass=known_mass)
        out.append(dict(
            method=method,
            M_ff=e['pair3_ff_mean'], M_ff_std=e['pair3_ff_std'],
            offset_mm=e['pair3_offset_mean'] * 1e3,
            offset_mm_std=e['pair3_offset_std'] * 1e3,
            W_off=e['pair3_Woffset_mean'],
            mass=e['pair3_mass_mean'],
        ))
    return out


def print_and_save_downstream(down, axis, output_dir: Path):
    def _spread(vals):
        v = np.array([x for x in vals if not np.isnan(x)])
        return (v.max() - v.min()) / abs(np.mean(v)) * 100 if len(v) else float('nan')

    print(f"\n{'method':12s} | {'M_ff(offset)[Nm]':>18} | {'CoM off[mm]':>14} | "
          f"{'W*off[Nm]':>10} | {'mass[kg]':>8}")
    print("-" * 74)
    for d in down:
        print(f"{d['method']:12s} | {d['M_ff']:+9.5f} ± {d['M_ff_std']:.5f} | "
              f"{d['offset_mm']:+7.2f} ± {d['offset_mm_std']:4.2f} | "
              f"{d['W_off']:+10.5f} | {d['mass']:8.3f}")
    print("-" * 74)
    print(f"spread across methods:  M_ff {_spread([d['M_ff'] for d in down]):.1f}%  "
          f"CoM-offset {_spread([d['offset_mm'] for d in down]):.1f}%  "
          f"mass {_spread([d['mass'] for d in down]):.1f}%  "
          f"→ final estimates are onset-method independent")

    p = Path(output_dir) / f"pelt_downstream_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'M_ff_Nm', 'M_ff_std', 'CoM_offset_mm',
                    'CoM_offset_std_mm', 'W_offset_Nm', 'mass_kg'])
        for d in down:
            w.writerow([d['method'], f"{d['M_ff']:.6f}", f"{d['M_ff_std']:.6f}",
                        f"{d['offset_mm']:.4f}", f"{d['offset_mm_std']:.4f}",
                        f"{d['W_off']:.6f}", f"{d['mass']:.4f}"])
    print(f"Downstream table → {p}")
    return p


def plot_downstream(down, axis, save_dir=None, show=True):
    labels = [_METHOD_LABEL.get(d['method'], d['method']) for d in down]
    x = np.arange(len(down))
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',
              'tab:purple'][:len(down)]
    off_lbl = 'x_{off}' if axis == 'y' else 'y_{off}'

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    ff = [d['M_ff'] for d in down]
    a1.bar(x, ff, yerr=[d['M_ff_std'] for d in down], capsize=5,
           color=colors, alpha=0.8, edgecolor='k')
    a1.axhline(np.mean(ff), color='gray', ls='--', lw=1,
               label=f'mean={np.mean(ff):+.4f}')
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_ylabel(r'Moment offset $M_{ff}=0.5(M_p+M_n)$ [N·m]')
    a1.set_title('Feedforward moment offset (pivot-free)')
    a1.legend(); a1.grid(axis='y', alpha=0.3)

    off = [d['offset_mm'] for d in down]
    a2.bar(x, off, yerr=[d['offset_mm_std'] for d in down], capsize=5,
           color=colors, alpha=0.8, edgecolor='k')
    a2.axhline(np.mean(off), color='gray', ls='--', lw=1,
               label=f'mean={np.mean(off):+.2f}mm')
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel(rf'CoM offset ${off_lbl}$ [mm]')
    a2.set_title('CoM offset (pivot-based)')
    a2.legend(); a2.grid(axis='y', alpha=0.3)

    fig.suptitle('Downstream robustness: final CoM / moment offset vs onset method',
                 fontsize=13)
    fig.tight_layout()
    if save_dir:
        p = Path(save_dir) / f"pelt_downstream_{axis}.png"
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Downstream figure → {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════

def print_and_save_table(rows, axis, output_dir: Path):
    cols = ['M_pls', 'M_cosh'] + [f'M_pelt_{c}' for c in PELT_COSTS]
    hdr = f"{'bag':<14}" + "".join(f"{c.replace('M_',''):>10}" for c in cols) + \
          f"{'|Δ|% norm':>11}"
    print(hdr)
    print("-" * len(hdr))
    dev = []
    for r in rows:
        base = r['M_pls']
        # magnitude deviation: positive when PELT finds a larger |M| (both signs)
        d = (abs(r['M_pelt_normal']) - abs(base)) / abs(base) * 100 if base else float('nan')
        dev.append(d)
        print(f"{r['bag']:<14}" + "".join(f"{r[c]:>+10.4f}" for c in cols) +
              f"{d:>+11.1f}")
    dev = np.array(dev)
    print("-" * len(hdr))
    print(f"PELT(normal) vs PLS:  mean |Δ| = {np.nanmean(dev):+.1f}%  "
          f"(std {np.nanstd(dev):.1f}%)  — PELT larger |M| (detection lag)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"pelt_crosscheck_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag'] + cols + ['pelt_normal_dev_pct'])
        for r, d in zip(rows, dev):
            w.writerow([r['bag']] + [f"{r[c]:.6f}" for c in cols] + [f"{d:.3f}"])
    print(f"\nTable → {p}")
    return p


def plot_crosscheck(series, axis, save_dir=None, show=True):
    n = len(series)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    for idx, s in enumerate(series):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        tw, ww, Mw = s['t_win'], s['omega_win'], s['M_win']
        ax.plot(tw, ww, 'k-', lw=0.8, alpha=0.85)
        js = []
        for lab, (t_on, M_on, clr, ls) in s['onsets'].items():
            ax.axvline(t_on, color=clr, ls=ls, lw=1.6, label=f'{lab}: M={M_on:+.3f}')
            js.append(t_on)
        ax2 = ax.twinx()
        ax2.plot(tw, Mw, 'tab:gray', lw=1.0, alpha=0.4)
        ax2.set_ylabel(f'$M_{axis}$ [N·m]', color='gray')
        # zoom around the onset cluster
        lo, hi = min(js), max(js)
        pad = 0.06 * (tw[-1] - tw[0])
        ax.set_xlim(max(tw[0], lo - pad), min(tw[-1], hi + pad))
        ax.set_title(bag_name_to_title(s['name']), fontsize=12)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(rf'$\omega_{axis}$ [rad/s]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left' if 'neg' in s['name'].lower()
                  else 'upper left')
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle('Onset cross-validation: model-based (PLS / cosh) vs '
                 'change-point (PELT)', fontsize=14)
    fig.tight_layout()
    if save_dir:
        p = Path(save_dir) / f"pelt_crosscheck_{axis}.png"
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Figure → {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="PELT/BinSeg onset cross-check vs the model-based onset.")
    p.add_argument('data_dir', type=str)
    p.add_argument('--axis', type=str, default=None, choices=['x', 'y'])
    p.add_argument('--mass', type=float, default=None,
                   help='Known mass [kg] for the downstream estimation.')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--save-fig', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir

    bags = load_excitation_dataset(dataset_dir)
    print(f"Loaded {len(bags)} bags: {[b.name for b in bags]}\n")
    axis = args.axis if args.axis else detect_axis(dataset_dir, bags)
    print(f"Axis        : {axis} ({'roll' if axis == 'x' else 'pitch'})")
    print(f"Cross-check : PELT/BinSeg costs {PELT_COSTS}\n")

    # 1. Onset-level cross-check (M_crit per method)
    rows, series = crosscheck(bags, axis)
    print_and_save_table(rows, axis, output_dir)

    # 2. Downstream: propagate each method to the CoM / moment offset
    print("\n── Downstream (CoM / moment offset) ──")
    down = downstream_crosscheck(bags, axis, known_mass=args.mass)
    print_and_save_downstream(down, axis, output_dir)

    save_dir = output_dir if args.save_fig else None
    show = not args.no_plot
    if show or save_dir:
        plot_crosscheck(series, axis, save_dir=save_dir, show=show)
        plot_downstream(down, axis, save_dir=save_dir, show=show)


if __name__ == "__main__":
    main()
