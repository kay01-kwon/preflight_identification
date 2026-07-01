#!/usr/bin/env python3
"""
Moment Excitation Analysis — Critical Values + Pivot Estimation

Usage
-----
# Auto-detect axis, compute critical values + pivot + CoM
python critical_value_getter.py DataSet/exp/Mx

# Pitch direction
python critical_value_getter.py DataSet/exp/My

# With options
python critical_value_getter.py DataSet/exp/My --mass 3.066 --save-fig --no-plot
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.extractor import load_excitation_dataset, BagData, PoseData, OdometryData
from analysis.critical_value_extractor import (
    CriticalValueExtractor,
    CriticalValueResult,
)
from utils import math_tools


# ═════════════════════════════════════════════════════════════
#  Pivot Estimation via Mocap Circle Fit
# ═════════════════════════════════════════════════════════════

def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw from quaternion (ZYX convention)."""
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def fit_circle_cz_fixed(xy: np.ndarray, z: np.ndarray, cz: float = 0.0):
    """
    Fit circle with fixed center z-coordinate.

    (xy - cx)^2 + (z - cz)^2 = R^2
    → 2·cx·xy + d = xy^2 + (z - cz)^2   where d = R^2 - cx^2

    Returns: (cx, R, residual_std)
    """
    rhs = xy ** 2 + (z - cz) ** 2
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    beta, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    cx = beta[0]
    R = np.sqrt(max(beta[1] + cx ** 2, 0))
    residuals = np.sqrt((xy - cx) ** 2 + (z - cz) ** 2) - R
    return cx, R, float(np.std(residuals))


def estimate_pivot_from_mocap(
    bag: BagData,
    onset_time: float,
    axis: str,
    cz: float = 0.0,
) -> dict:
    """
    Estimate pivot distance from mocap circle fit on body-frame
    horizontal vs absolute-z trajectory.

    Window: onset → max |displacement| (body frame horizontal).

    Parameters
    ----------
    bag        : BagData with .pose (mocap) and .odom (for onset timing)
    onset_time : onset time relative to odom.t[0]
    axis       : 'x' (pitch → dx vs z) or 'y' (roll → dy vs z)
    cz         : fixed z-coordinate of circle center (ground level)

    Returns
    -------
    dict with: pivot_abs, R, residual, N, xy_fit, z_fit, cx
    """
    t0 = bag.odom.t[0]
    t_mc = bag.pose.t - t0
    px = bag.pose.position[:, 0]
    py = bag.pose.position[:, 1]
    pz = bag.pose.position[:, 2]

    # Initial yaw from mocap quaternion
    q0 = bag.pose.quaternion[0]  # [qw, qx, qy, qz]
    yaw0 = quat_to_yaw(q0[0], q0[1], q0[2], q0[3])

    # Idle position (before onset)
    idle_mask = t_mc < onset_time * 0.5
    if np.sum(idle_mask) < 10:
        idle_mask = np.arange(len(t_mc)) < 50
    px0 = np.mean(px[idle_mask])
    py0 = np.mean(py[idle_mask])

    # Yaw correction → body frame displacement
    c = np.cos(-yaw0)
    s = np.sin(-yaw0)
    dx_b = (px - px0) * c - (py - py0) * s  # body x
    dy_b = (px - px0) * s + (py - py0) * c  # body y

    # Select horizontal displacement based on axis
    if axis == 'y':
        # Pitch: use body x displacement
        d_horiz = dx_b
    else:
        # Roll: use body y displacement
        d_horiz = dy_b

    # Window: onset → max |displacement|
    mc_onset_idx = int(np.searchsorted(t_mc, onset_time))
    if mc_onset_idx >= len(t_mc) - 5:
        return dict(pivot_abs=np.nan, R=np.nan, residual=np.nan,
                    N=0, xy_fit=None, z_fit=None, cx=np.nan)

    mc_max_idx = mc_onset_idx + np.argmax(np.abs(d_horiz[mc_onset_idx:]))
    sl = slice(mc_onset_idx, mc_max_idx + 1)

    xy_fit = d_horiz[sl] * 1e3   # mm
    z_fit = pz[sl] * 1e3          # mm, absolute z

    if len(xy_fit) < 5:
        return dict(pivot_abs=np.nan, R=np.nan, residual=np.nan,
                    N=0, xy_fit=None, z_fit=None, cx=np.nan)

    cx, R, res = fit_circle_cz_fixed(xy_fit, z_fit, cz)

    return dict(
        pivot_abs=abs(cx),
        cx=cx,
        R=R,
        residual=res,
        N=len(xy_fit),
        xy_fit=xy_fit,
        z_fit=z_fit,
    )


def _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass=None):
    """
    Solve mass and offset from a single pos/neg pair.

    Returns (mass_kg, offset_m, W_offset_Nm) or (nan, nan, nan).
    """
    G = 9.81
    if pp is None or pn is None or pp < 1e-6 or pn < 1e-6:
        return np.nan, np.nan, np.nan

    if known_mass is not None:
        mg = known_mass * G
    else:
        mg = (fn * pn - Mn + fp * pp + Mp) / (pn + pp)

    m = mg / G

    if axis == 'y':
        off_n = pn * (fn / mg - Mn / (mg * pn) - 1)
        off_p = pp * (1 - fp / mg - Mp / (mg * pp))
        offset = -0.5 * (off_n + off_p)
    else:
        off_n = pn * (1 - fn / mg + Mn / (mg * pn))
        off_p = pp * (fp / mg + Mp / (mg * pp) - 1)
        offset = 0.5 * (off_n + off_p)

    W_offset = mg * offset

    return m, offset, W_offset


def compute_mass_and_offset(
    critical_results: list[CriticalValueResult],
    pivot_results: list[dict],
    axis: str,
    known_mass: Optional[float] = None,
) -> dict:
    """
    Compute mass and CoM offset from critical values + pivot distances.

    Computes both:
      - 3 same-trial pairs (pos_i, neg_i)
      - 9 all combinations (pos_i, neg_j)

    Each combination gets its own mass, offset, and W·offset.

    Returns dict with pair_3 and comb_9 sub-dicts.
    """
    pos_crits = [r for r in critical_results if 'pos' in r.bag_name.lower()]
    neg_crits = [r for r in critical_results if 'neg' in r.bag_name.lower()]
    pos_pivots = [p for r, p in zip(critical_results, pivot_results)
                  if 'pos' in r.bag_name.lower()]
    neg_pivots = [p for r, p in zip(critical_results, pivot_results)
                  if 'neg' in r.bag_name.lower()]

    n_pos = len(pos_crits)
    n_neg = len(neg_crits)
    n_pairs = min(n_pos, n_neg)

    G = 9.81

    # ── 3 same-trial pairs ──
    pair3_mass = []
    pair3_offset = []
    pair3_Woffset = []
    pair3_ff_onset = []
    pair3_labels = []

    for i in range(n_pairs):
        fp = pos_crits[i].onset_thrust
        fn = neg_crits[i].onset_thrust
        Mp = pos_crits[i].onset_moment
        Mn = neg_crits[i].onset_moment
        pp = pos_pivots[i]['pivot_abs'] * 1e-3 if not np.isnan(pos_pivots[i]['pivot_abs']) else None
        pn = neg_pivots[i]['pivot_abs'] * 1e-3 if not np.isnan(neg_pivots[i]['pivot_abs']) else None

        pair3_ff_onset.append(0.5 * (Mp + Mn))
        m, off, Woff = _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass)
        pair3_mass.append(m)
        pair3_offset.append(off)
        pair3_Woffset.append(Woff)
        pair3_labels.append(f"p{i+1}-n{i+1}")

    # ── 9 all combinations ──
    comb9_mass = []
    comb9_offset = []
    comb9_Woffset = []
    comb9_ff_onset = []
    comb9_labels = []

    for i in range(n_pos):
        for j in range(n_neg):
            fp = pos_crits[i].onset_thrust
            fn = neg_crits[j].onset_thrust
            Mp = pos_crits[i].onset_moment
            Mn = neg_crits[j].onset_moment
            pp = pos_pivots[i]['pivot_abs'] * 1e-3 if not np.isnan(pos_pivots[i]['pivot_abs']) else None
            pn = neg_pivots[j]['pivot_abs'] * 1e-3 if not np.isnan(neg_pivots[j]['pivot_abs']) else None

            comb9_ff_onset.append(0.5 * (Mp + Mn))
            m, off, Woff = _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass)
            comb9_mass.append(m)
            comb9_offset.append(off)
            comb9_Woffset.append(Woff)
            comb9_labels.append(f"p{i+1}-n{j+1}")

    # Convert to arrays, filter NaN
    def _stats(arr):
        arr = np.array(arr)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan, 0.0, np.array([])
        return float(np.mean(valid)), float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0, valid

    p3_m_mean, p3_m_std, _ = _stats(pair3_mass)
    p3_off_mean, p3_off_std, _ = _stats(pair3_offset)
    p3_Woff_mean, p3_Woff_std, _ = _stats(pair3_Woffset)
    p3_ff_mean, p3_ff_std, _ = _stats(pair3_ff_onset)

    c9_m_mean, c9_m_std, _ = _stats(comb9_mass)
    c9_off_mean, c9_off_std, _ = _stats(comb9_offset)
    c9_Woff_mean, c9_Woff_std, _ = _stats(comb9_Woffset)
    c9_ff_mean, c9_ff_std, _ = _stats(comb9_ff_onset)

    return dict(
        # 3 same-trial pairs
        pair3_mass=pair3_mass, pair3_offset=pair3_offset,
        pair3_Woffset=pair3_Woffset, pair3_ff_onset=pair3_ff_onset,
        pair3_labels=pair3_labels,
        pair3_mass_mean=p3_m_mean, pair3_mass_std=p3_m_std,
        pair3_offset_mean=p3_off_mean, pair3_offset_std=p3_off_std,
        pair3_Woffset_mean=p3_Woff_mean, pair3_Woffset_std=p3_Woff_std,
        pair3_ff_mean=p3_ff_mean, pair3_ff_std=p3_ff_std,
        # 9 all combinations
        comb9_mass=comb9_mass, comb9_offset=comb9_offset,
        comb9_Woffset=comb9_Woffset, comb9_ff_onset=comb9_ff_onset,
        comb9_labels=comb9_labels,
        comb9_mass_mean=c9_m_mean, comb9_mass_std=c9_m_std,
        comb9_offset_mean=c9_off_mean, comb9_offset_std=c9_off_std,
        comb9_Woffset_mean=c9_Woff_mean, comb9_Woffset_std=c9_Woff_std,
        comb9_ff_mean=c9_ff_mean, comb9_ff_std=c9_ff_std,
    )


# ═════════════════════════════════════════════════════════════
#  CSV Export — Estimation Results
# ═════════════════════════════════════════════════════════════

def save_estimation_csv(
    critical_results: list[CriticalValueResult],
    pivot_results: list[dict],
    estimation: dict,
    axis: str,
    output_dir: Path,
    known_mass: Optional[float] = None,
) -> Path:
    """
    Save CoM offset estimation results to CSV.

    Generates:
      1. com_estimation_summary_{axis}.csv  — per-trial critical values + pivot
      2. com_estimation_pairs_{axis}.csv    — 3 same-trial pair results
      3. com_estimation_combs_{axis}.csv    — 9 all-combination results
      4. com_estimation_result_{axis}.csv   — aggregated statistics
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offset_label = 'x_off' if axis == 'y' else 'y_off'

    # ── 1. Per-trial details ──
    detail_path = output_dir / f"com_estimation_summary_{axis}.csv"
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bag_name', 'direction',
            'f_crit_N', 'M_crit_Nm',
            'pivot_mm', 'pivot_R_mm', 'pivot_rmse_mm', 'pivot_N_pts',
        ])
        for crit, piv in zip(critical_results, pivot_results):
            direction = 'pos' if 'pos' in crit.bag_name.lower() else 'neg'
            piv_val = f"{piv['pivot_abs']:.2f}" if not np.isnan(piv['pivot_abs']) else ''
            R_val = f"{piv['R']:.2f}" if not np.isnan(piv['R']) else ''
            res_val = f"{piv['residual']:.4f}" if not np.isnan(piv['residual']) else ''
            writer.writerow([
                crit.bag_name, direction,
                f"{crit.onset_thrust:.6f}",
                f"{crit.onset_moment:.8f}",
                piv_val, R_val, res_val, piv['N'],
            ])
    print(f"  Per-trial details      → {detail_path}")

    # ── 2. 3 same-trial pairs ──
    pairs_path = output_dir / f"com_estimation_pairs_{axis}.csv"
    with open(pairs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pair', 'mass_kg', f'{offset_label}_mm',
            'W_offset_Nm', 'ff_onset_Nm',
        ])
        for i, label in enumerate(estimation['pair3_labels']):
            m = estimation['pair3_mass'][i]
            off = estimation['pair3_offset'][i]
            Woff = estimation['pair3_Woffset'][i]
            ff = estimation['pair3_ff_onset'][i]
            writer.writerow([
                label,
                f"{m:.6f}" if not np.isnan(m) else '',
                f"{off*1e3:.4f}" if not np.isnan(off) else '',
                f"{Woff:.8f}" if not np.isnan(Woff) else '',
                f"{ff:.8f}",
            ])
    print(f"  3 same-trial pairs     → {pairs_path}")

    # ── 3. 9 all combinations ──
    combs_path = output_dir / f"com_estimation_combs_{axis}.csv"
    with open(combs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'combination', 'mass_kg', f'{offset_label}_mm',
            'W_offset_Nm', 'ff_onset_Nm',
        ])
        for i, label in enumerate(estimation['comb9_labels']):
            m = estimation['comb9_mass'][i]
            off = estimation['comb9_offset'][i]
            Woff = estimation['comb9_Woffset'][i]
            ff = estimation['comb9_ff_onset'][i]
            writer.writerow([
                label,
                f"{m:.6f}" if not np.isnan(m) else '',
                f"{off*1e3:.4f}" if not np.isnan(off) else '',
                f"{Woff:.8f}" if not np.isnan(Woff) else '',
                f"{ff:.8f}",
            ])
    print(f"  9 all combinations     → {combs_path}")

    # ── 4. Aggregated results ──
    result_path = output_dir / f"com_estimation_result_{axis}.csv"

    pos_pivots = [p['pivot_abs'] for r, p in zip(critical_results, pivot_results)
                  if 'pos' in r.bag_name.lower() and not np.isnan(p['pivot_abs'])]
    neg_pivots = [p['pivot_abs'] for r, p in zip(critical_results, pivot_results)
                  if 'neg' in r.bag_name.lower() and not np.isnan(p['pivot_abs'])]

    with open(result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value', 'std', 'unit', 'note'])

        # Pivot
        if pos_pivots:
            writer.writerow(['pivot_pos', f"{np.mean(pos_pivots):.2f}",
                f"{np.std(pos_pivots,ddof=1):.2f}" if len(pos_pivots)>1 else '0.00',
                'mm', f'N={len(pos_pivots)}'])
        if neg_pivots:
            writer.writerow(['pivot_neg', f"{np.mean(neg_pivots):.2f}",
                f"{np.std(neg_pivots,ddof=1):.2f}" if len(neg_pivots)>1 else '0.00',
                'mm', f'N={len(neg_pivots)}'])

        # 3 pairs
        writer.writerow(['mass_3pair', f"{estimation['pair3_mass_mean']:.6f}",
            f"{estimation['pair3_mass_std']:.6f}", 'kg',
            f"known={known_mass}" if known_mass else 'estimated'])
        writer.writerow([f'{offset_label}_3pair',
            f"{estimation['pair3_offset_mean']*1e3:.4f}",
            f"{estimation['pair3_offset_std']*1e3:.4f}", 'mm', '3 same-trial pairs'])
        writer.writerow(['W_offset_3pair',
            f"{estimation['pair3_Woffset_mean']:.8f}",
            f"{estimation['pair3_Woffset_std']:.8f}", 'Nm', '3 same-trial pairs'])
        writer.writerow(['ff_onset_3pair',
            f"{estimation['pair3_ff_mean']:.8f}",
            f"{estimation['pair3_ff_std']:.8f}", 'Nm', '0.5*(Mp+Mn) 3 pairs'])

        # 9 combinations
        writer.writerow(['mass_9comb', f"{estimation['comb9_mass_mean']:.6f}",
            f"{estimation['comb9_mass_std']:.6f}", 'kg',
            f"known={known_mass}" if known_mass else 'estimated'])
        writer.writerow([f'{offset_label}_9comb',
            f"{estimation['comb9_offset_mean']*1e3:.4f}",
            f"{estimation['comb9_offset_std']*1e3:.4f}", 'mm', '9 all combinations'])
        writer.writerow(['W_offset_9comb',
            f"{estimation['comb9_Woffset_mean']:.8f}",
            f"{estimation['comb9_Woffset_std']:.8f}", 'Nm', '9 all combinations'])
        writer.writerow(['ff_onset_9comb',
            f"{estimation['comb9_ff_mean']:.8f}",
            f"{estimation['comb9_ff_std']:.8f}", 'Nm', '0.5*(Mp+Mn) 9 combs'])

    print(f"  Aggregated result      → {result_path}")
    return result_path


# ═════════════════════════════════════════════════════════════
#  Pivot Circle Fit Plotting
# ═════════════════════════════════════════════════════════════

def plot_pivot_fits(
    bags: list[BagData],
    critical_results: list[CriticalValueResult],
    pivot_results: list[dict],
    axis: str,
    save_dir: Optional[Path] = None,
    show: bool = True,
):
    """Plot circle fits for all bags in a single figure."""
    n = len(bags)
    if n == 0:
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes_arr = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    horiz_label = 'dx_body' if axis == 'y' else 'dy_body'

    for idx, (bag, crit, piv) in enumerate(zip(bags, critical_results, pivot_results)):
        r, c = divmod(idx, cols)
        ax = axes_arr[r][c]

        if piv['xy_fit'] is None:
            ax.set_title(f'{bag.name}\nInsufficient data')
            continue

        xy_fit = piv['xy_fit']
        z_fit = piv['z_fit']
        cx = piv['cx']
        R = piv['R']
        res = piv['residual']

        ax.plot(xy_fit, z_fit, '.', color='tab:blue', ms=3,
                label=f'data (N={piv["N"]})')
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + R * np.cos(theta), R * np.sin(theta),
                'r-', lw=1.5, alpha=0.5)
        ax.plot(cx, 0, 'r+', ms=15, mew=2,
                label=f'pivot |cx|={piv["pivot_abs"]:.1f}mm')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower left')
        ax.set_title(f'{bag.name}\n|cx|={piv["pivot_abs"]:.1f}mm  R={R:.1f}mm  res={res:.2f}mm')
        ax.set_xlabel(f'{horiz_label} [mm]')
        ax.set_ylabel('z [mm]')

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes_arr[r][c].set_visible(False)

    plt.suptitle(f'Mocap Circle Fit (cz=0, onset→max|d|) — axis={axis}', fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"pivot_circle_fit_{axis}.png"
        fig.savefig(fig_path, dpi=600, bbox_inches='tight')
        print(f"  Pivot plot → {fig_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  Estimation Result Plots
# ═════════════════════════════════════════════════════════════

def plot_estimation_results(
    estimation: dict,
    axis: str,
    save_dir: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot estimation results:
      1. Scatter: offset vs W·offset for 3-pair and 9-comb
      2. Bar chart: mass, offset, W·offset for each combination

    Parameters
    ----------
    estimation : dict from compute_mass_and_offset()
    axis       : 'x' or 'y'
    """
    G = 9.81
    offset_label = '$x_{off}$' if axis == 'y' else '$y_{off}$'
    moment_label = '$M_y^{ff}$' if axis == 'y' else '$M_x^{ff}$'
    offset_unit_label = f'{"x" if axis == "y" else "y"}_off [mm]'
    moment_unit_label = f'{"M_y" if axis == "y" else "M_x"} ff [N·m]'

    # ── Figure 1: Scatter — offset [mm] vs W·offset [N·m] ──
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # 3 pairs
    ax = axes1[0]
    off_3 = np.array(estimation['pair3_offset']) * 1e3  # mm
    Woff_3 = np.array(estimation['pair3_Woffset'])       # N·m
    labels_3 = estimation['pair3_labels']
    valid_3 = ~np.isnan(off_3) & ~np.isnan(Woff_3)

    if np.any(valid_3):
        ax.scatter(off_3[valid_3], Woff_3[valid_3], c='tab:blue', s=100, zorder=3, edgecolors='k')
        for i in np.where(valid_3)[0]:
            ax.annotate(labels_3[i], (off_3[i], Woff_3[i]),
                        textcoords='offset points', xytext=(8, 5), fontsize=9)
        # Mean crosshair
        ax.axvline(estimation['pair3_offset_mean']*1e3, color='red', ls='--', alpha=0.5)
        ax.axhline(estimation['pair3_Woffset_mean'], color='red', ls='--', alpha=0.5)
        ax.plot(estimation['pair3_offset_mean']*1e3, estimation['pair3_Woffset_mean'],
                'r+', ms=15, mew=2, zorder=4, label='mean')

    ax.set_xlabel(f'{offset_label} [mm]')
    ax.set_ylabel(f'{moment_label} = W·{offset_label} [N·m]')
    ax.set_title(f'3 Same-Trial Pairs\n'
                 f'mean: {offset_label}={estimation["pair3_offset_mean"]*1e3:+.2f}mm, '
                 f'W·off={estimation["pair3_Woffset_mean"]:+.4f}N·m')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 9 combinations
    ax = axes1[1]
    off_9 = np.array(estimation['comb9_offset']) * 1e3
    Woff_9 = np.array(estimation['comb9_Woffset'])
    labels_9 = estimation['comb9_labels']
    valid_9 = ~np.isnan(off_9) & ~np.isnan(Woff_9)

    if np.any(valid_9):
        ax.scatter(off_9[valid_9], Woff_9[valid_9], c='tab:green', s=60, zorder=3, edgecolors='k', alpha=0.8)
        for i in np.where(valid_9)[0]:
            ax.annotate(labels_9[i], (off_9[i], Woff_9[i]),
                        textcoords='offset points', xytext=(6, 4), fontsize=7)
        ax.axvline(estimation['comb9_offset_mean']*1e3, color='red', ls='--', alpha=0.5)
        ax.axhline(estimation['comb9_Woffset_mean'], color='red', ls='--', alpha=0.5)
        ax.plot(estimation['comb9_offset_mean']*1e3, estimation['comb9_Woffset_mean'],
                'r+', ms=15, mew=2, zorder=4, label='mean')

    ax.set_xlabel(f'{offset_label} [mm]')
    ax.set_ylabel(f'{moment_label} = W·{offset_label} [N·m]')
    ax.set_title(f'9 All Combinations\n'
                 f'mean: {offset_label}={estimation["comb9_offset_mean"]*1e3:+.2f}mm, '
                 f'W·off={estimation["comb9_Woffset_mean"]:+.4f}N·m')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig1.suptitle(f'CoM Offset vs Feedforward Moment (axis={axis})', fontsize=13)
    fig1.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_dir / f"estimation_scatter_{axis}.png", dpi=600, bbox_inches='tight')
        print(f"  Estimation scatter → {save_dir / f'estimation_scatter_{axis}.png'}")

    # ── Figure 2: Bar charts — mass, offset, W·offset per combination ──
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: 3 pairs
    _plot_bar_row(axes2[0], labels_3, estimation['pair3_mass'],
                  estimation['pair3_offset'], estimation['pair3_Woffset'],
                  estimation['pair3_ff_onset'],
                  offset_unit_label, moment_unit_label, '3 Same-Trial Pairs')

    # Bottom row: 9 combinations
    _plot_bar_row(axes2[1], labels_9, estimation['comb9_mass'],
                  estimation['comb9_offset'], estimation['comb9_Woffset'],
                  estimation['comb9_ff_onset'],
                  offset_unit_label, moment_unit_label, '9 All Combinations')

    fig2.suptitle(f'Estimation Bar Charts (axis={axis})', fontsize=13)
    fig2.tight_layout()

    if save_dir:
        fig2.savefig(save_dir / f"estimation_bars_{axis}.png", dpi=600, bbox_inches='tight')
        print(f"  Estimation bars    → {save_dir / f'estimation_bars_{axis}.png'}")

    # ── Figure 3: Summary — box plot, 3-pair vs 9-comb ──
    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 6))
    colors_group = ['tab:blue', 'tab:green']

    plot_configs = [
        ('Mass', 'Mass [kg]',
         estimation['pair3_mass'], estimation['comb9_mass']),
        (f'Offset ({offset_label})', offset_unit_label,
         [o * 1e3 for o in estimation['pair3_offset']],
         [o * 1e3 for o in estimation['comb9_offset']]),
        (f'W·offset ({moment_label})', moment_unit_label,
         estimation['pair3_Woffset'], estimation['comb9_Woffset']),
        (f'0.5·(Mp+Mn) ({moment_label})', moment_unit_label,
         estimation['pair3_ff_onset'], estimation['comb9_ff_onset']),
    ]

    for col, (title, ylabel, vals_3, vals_9) in enumerate(plot_configs):
        ax = axes3[col]

        # Filter NaN
        data_3 = [v for v in vals_3 if not np.isnan(v)]
        data_9 = [v for v in vals_9 if not np.isnan(v)]

        box_data = []
        box_labels = []
        box_colors = []
        if data_3:
            box_data.append(data_3)
            box_labels.append(f'3 pairs\n(N={len(data_3)})')
            box_colors.append(colors_group[0])
        if data_9:
            box_data.append(data_9)
            box_labels.append(f'9 combs\n(N={len(data_9)})')
            box_colors.append(colors_group[1])

        if not box_data:
            ax.set_title(title + '\nNo data')
            continue

        bp = ax.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='k', markersize=8),
            medianprops=dict(color='orange', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.7),
            widths=0.5,
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Annotate mean ± std
        for i, data in enumerate(box_data):
            arr = np.array(data)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0
            ax.text(i + 1.3, mean, f'μ={mean:+.4f}\nσ={std:.4f}',
                    va='center', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    fig3.suptitle(f'Estimation Summary: 3 Pairs vs 9 Combinations (axis={axis})\n'
                  f'◆ mean, ─ median, box=IQR, whisker=1.5×IQR', fontsize=13)
    fig3.tight_layout()

    if save_dir:
        fig3.savefig(save_dir / f"estimation_summary_{axis}.png", dpi=600, bbox_inches='tight')
        print(f"  Estimation summary → {save_dir / f'estimation_summary_{axis}.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def _plot_bar_row(axes, labels, masses, offsets, Woffsets, ff_onsets,
                  offset_unit_label, moment_unit_label, row_title):
    """Helper: plot one row of bar charts (mass, offset, moments)."""
    n = len(labels)
    x = np.arange(n)
    colors_bar = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    masses = np.array(masses)
    offsets = np.array(offsets) * 1e3  # mm
    Woffsets = np.array(Woffsets)
    ff_onsets = np.array(ff_onsets)

    # Col 0: Mass
    ax = axes[0]
    valid = ~np.isnan(masses)
    if np.any(valid):
        bars = ax.bar(x[valid], masses[valid], color=colors_bar[valid], edgecolor='k', alpha=0.8)
        ax.axhline(np.nanmean(masses), color='red', ls='--', lw=1.5,
                    label=f'mean={np.nanmean(masses):.4f}')
        for i in np.where(valid)[0]:
            ax.text(i, masses[i], f'{masses[i]:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mass [kg]')
    ax.set_title(f'{row_title} — Mass')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Col 1: Offset
    ax = axes[1]
    valid = ~np.isnan(offsets)
    if np.any(valid):
        colors_off = ['tab:blue' if v >= 0 else 'tab:red' for v in offsets[valid]]
        ax.bar(x[valid], offsets[valid], color=colors_off, edgecolor='k', alpha=0.8)
        ax.axhline(np.nanmean(offsets), color='red', ls='--', lw=1.5,
                    label=f'mean={np.nanmean(offsets):+.3f}')
        for i in np.where(valid)[0]:
            ax.text(i, offsets[i], f'{offsets[i]:+.2f}', ha='center',
                    va='bottom' if offsets[i] >= 0 else 'top', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(offset_unit_label)
    ax.set_title(f'{row_title} — Offset')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', lw=0.5)

    # Col 2: W·offset and 0.5*(Mp+Mn)
    ax = axes[2]
    valid_W = ~np.isnan(Woffsets)
    width = 0.35
    if np.any(valid_W):
        ax.bar(x[valid_W] - width/2, Woffsets[valid_W], width,
               color='tab:blue', edgecolor='k', alpha=0.8, label='W·offset')
    ax.bar(x + width/2, ff_onsets, width,
           color='tab:orange', edgecolor='k', alpha=0.8, label='0.5(Mp+Mn)')
    if np.any(valid_W):
        ax.axhline(np.nanmean(Woffsets), color='tab:blue', ls='--', lw=1,
                    label=f'W·off mean={np.nanmean(Woffsets):+.4f}')
    ax.axhline(np.mean(ff_onsets), color='tab:orange', ls='--', lw=1,
                label=f'ff mean={np.mean(ff_onsets):+.4f}')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(moment_unit_label)
    ax.set_title(f'{row_title} — Feedforward Moment')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', lw=0.5)


# ═════════════════════════════════════════════════════════════
#  Axis Detection
# ═════════════════════════════════════════════════════════════

def detect_axis(data_dir: Path, bags: list[BagData]) -> str:
    """Auto-detect axis from directory name or bag names."""
    dir_name = data_dir.name.lower()
    if 'mx' in dir_name:
        return 'x'
    if 'my' in dir_name:
        return 'y'

    if bags:
        bag_name = bags[0].name.lower()
        if 'mx' in bag_name:
            return 'x'
        if 'my' in bag_name:
            return 'y'

    raise ValueError(
        f"Cannot auto-detect axis from '{data_dir.name}' or bag names. "
        f"Use --axis x or --axis y."
    )


# ═════════════════════════════════════════════════════════════
#  Argument Parser
# ═════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Moment Excitation Analysis — Critical Values + Pivot Estimation"
    )
    parser.add_argument(
        'data_dir', type=str,
        help='Path to the data directory (e.g. DataSet/exp/Mx)',
    )
    parser.add_argument(
        '--axis', type=str, default=None, choices=['x', 'y'],
        help='Angular velocity axis: x (roll) or y (pitch). '
             'Auto-detected from Mx/My if omitted.',
    )
    parser.add_argument(
        '--mass', type=float, default=None,
        help='Known mass [kg]. If provided, uses it instead of estimating.',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory. Defaults to data_dir.',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip showing plots.',
    )
    parser.add_argument(
        '--save-fig', action='store_true',
        help='Save figures as PNG.',
    )
    parser.add_argument(
        '--window-margin', type=int, default=0,
        help='Manual GLR window margin. 0 = auto.',
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    dataset_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir

    # ── 1. Load bags ──────────────────────────────────────
    bags: list[BagData] = load_excitation_dataset(dataset_dir)
    print(f"Loaded {len(bags)} bags: {[b.name for b in bags]}\n")

    # ── 2. Detect axis ────────────────────────────────────
    axis = args.axis if args.axis else detect_axis(dataset_dir, bags)
    axis_name = 'roll' if axis == 'x' else 'pitch'
    offset_label = 'y_off' if axis == 'x' else 'x_off'

    print(f'Axis        : {axis} ({axis_name})')
    print(f'Data dir    : {dataset_dir}')
    print(f'Output dir  : {output_dir}')
    if args.mass:
        print(f'Known mass  : {args.mass} kg')
    print()

    # ── 3. Extract critical values ────────────────────────
    extractor = CriticalValueExtractor(window_margin=args.window_margin)
    print("── Critical Value Extraction (GLR) ──")
    critical_results = extractor.extract_batch(bags, axis=axis)

    # ── 4. Export CSV ─────────────────────────────────────
    print("\n── CSV Export ──")
    extractor.save_batch_csv(critical_results, output_dir=output_dir)

    # ── 5. Pivot estimation (mocap circle fit) ────────────
    print("\n── Pivot Estimation (Mocap Circle Fit, cz=0) ──")
    pivot_results = []
    for bag, crit in zip(bags, critical_results):
        piv = estimate_pivot_from_mocap(
            bag, onset_time=crit.onset_time, axis=axis, cz=0.0,
        )
        pivot_results.append(piv)
        status = f"|cx|={piv['pivot_abs']:.1f}mm  R={piv['R']:.1f}mm  res={piv['residual']:.2f}mm  N={piv['N']}" \
            if not np.isnan(piv['pivot_abs']) else "FAILED"
        print(f"  {bag.name}: {status}")

    # ── 6. Mass & CoM offset ──────────────────────────────
    print("\n── Mass & CoM Offset Estimation ──")
    est = compute_mass_and_offset(
        critical_results, pivot_results, axis=axis, known_mass=args.mass,
    )

    # ── 6b. Save estimation CSV ───────────────────────────
    print("\n── Estimation CSV Export ──")
    save_estimation_csv(
        critical_results, pivot_results, est,
        axis=axis, output_dir=output_dir, known_mass=args.mass,
    )

    # ── 7. Summary ────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  Summary  ({dataset_dir.name},  axis={axis})")
    print(f"{'='*75}")

    # Critical values table
    print(f"\n  ── Critical Values ──")
    print(f"  {'Bag':<25} {'f_col[N]':>10} {'M[N·m]':>12} {'ω[rad/s]':>12}")
    print("  " + "-" * 60)
    for r in critical_results:
        print(f"  {r.bag_name:<25} {r.onset_thrust:>10.4f} {r.onset_moment:>+12.6f} "
              f"{r.onset_omega:>12.6f}")

    # Pivot table
    print(f"\n  ── Pivot Distances ──")
    print(f"  {'Bag':<25} {'|pivot|[mm]':>12} {'R[mm]':>10} {'res[mm]':>10}")
    print("  " + "-" * 55)
    pos_pivots = []
    neg_pivots = []
    for r, p in zip(critical_results, pivot_results):
        pval = f"{p['pivot_abs']:.1f}" if not np.isnan(p['pivot_abs']) else "N/A"
        rval = f"{p['R']:.1f}" if not np.isnan(p['R']) else "N/A"
        resval = f"{p['residual']:.2f}" if not np.isnan(p['residual']) else "N/A"
        print(f"  {r.bag_name:<25} {pval:>12} {rval:>10} {resval:>10}")
        if 'pos' in r.bag_name.lower() and not np.isnan(p['pivot_abs']):
            pos_pivots.append(p['pivot_abs'])
        elif 'neg' in r.bag_name.lower() and not np.isnan(p['pivot_abs']):
            neg_pivots.append(p['pivot_abs'])

    if pos_pivots:
        print(f"\n  pos pivot avg: {np.mean(pos_pivots):.1f} ± {np.std(pos_pivots, ddof=1):.1f} mm")
    if neg_pivots:
        print(f"  neg pivot avg: {np.mean(neg_pivots):.1f} ± {np.std(neg_pivots, ddof=1):.1f} mm")

    # Mass & offset
    print(f"\n  ── Mass & Offset (3 same-trial pairs) ──")
    for i, label in enumerate(est['pair3_labels']):
        m = est['pair3_mass'][i]
        off = est['pair3_offset'][i]
        Woff = est['pair3_Woffset'][i]
        m_s = f"{m:.4f}" if not np.isnan(m) else "N/A"
        off_s = f"{off*1e3:+.3f}" if not np.isnan(off) else "N/A"
        Woff_s = f"{Woff:+.6f}" if not np.isnan(Woff) else "N/A"
        print(f"    {label}: mass={m_s}kg  {offset_label}={off_s}mm  W·off={Woff_s}N·m")
    print(f"    ───────────────────────────────────────────")
    print(f"    mean: mass={est['pair3_mass_mean']:.4f}±{est['pair3_mass_std']:.4f}kg  "
          f"{offset_label}={est['pair3_offset_mean']*1e3:+.3f}±{est['pair3_offset_std']*1e3:.3f}mm  "
          f"W·off={est['pair3_Woffset_mean']:+.6f}±{est['pair3_Woffset_std']:.6f}N·m"
          + (f"  (input mass: {args.mass})" if args.mass else ""))

    print(f"\n  ── Mass & Offset (9 all combinations) ──")
    for i, label in enumerate(est['comb9_labels']):
        m = est['comb9_mass'][i]
        off = est['comb9_offset'][i]
        Woff = est['comb9_Woffset'][i]
        m_s = f"{m:.4f}" if not np.isnan(m) else "N/A"
        off_s = f"{off*1e3:+.3f}" if not np.isnan(off) else "N/A"
        Woff_s = f"{Woff:+.6f}" if not np.isnan(Woff) else "N/A"
        print(f"    {label}: mass={m_s}kg  {offset_label}={off_s}mm  W·off={Woff_s}N·m")
    print(f"    ───────────────────────────────────────────")
    print(f"    mean: mass={est['comb9_mass_mean']:.4f}±{est['comb9_mass_std']:.4f}kg  "
          f"{offset_label}={est['comb9_offset_mean']*1e3:+.3f}±{est['comb9_offset_std']*1e3:.3f}mm  "
          f"W·off={est['comb9_Woffset_mean']:+.6f}±{est['comb9_Woffset_std']:.6f}N·m")

    # Feedforward
    print(f"\n  ── Feedforward Moment ──")
    print(f"  0.5*(Mp+Mn) 3 pairs : {est['pair3_ff_mean']:+.6f} ± {est['pair3_ff_std']:.6f} N·m")
    print(f"  0.5*(Mp+Mn) 9 combs : {est['comb9_ff_mean']:+.6f} ± {est['comb9_ff_std']:.6f} N·m")
    print(f"  W·offset    3 pairs : {est['pair3_Woffset_mean']:+.6f} ± {est['pair3_Woffset_std']:.6f} N·m")
    print(f"  W·offset    9 combs : {est['comb9_Woffset_mean']:+.6f} ± {est['comb9_Woffset_std']:.6f} N·m")

    print(f"\n{'='*75}")

    # ── 8. Plots ──────────────────────────────────────────
    save_dir = output_dir if args.save_fig else None
    show = not args.no_plot

    if show or save_dir:
        # Critical value plots
        CriticalValueExtractor.plot_results(
            critical_results,
            suptitle=f"{dataset_dir.name}",
            save_dir=save_dir,
            show=show,
        )
        # Pivot circle fit plots
        plot_pivot_fits(
            bags, critical_results, pivot_results, axis=axis,
            save_dir=save_dir, show=show,
        )
        # Estimation scatter + bar charts
        plot_estimation_results(
            est, axis=axis,
            save_dir=save_dir, show=show,
        )


if __name__ == "__main__":
    main()