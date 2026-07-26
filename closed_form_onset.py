#!/usr/bin/env python3
"""
Closed-form tip-over onset detection — standalone, minimal implementation
=========================================================================
Everything needed for the closed-form (``cosh``) critical-moment
identification, and nothing else: no quadratic model, no Huber option, no
moment floor, no legacy branches. Read top to bottom.

Derivation
----------
Ramping the commanded moment tilts the vehicle about its landing-gear contact
line. Past the balance point gravity feeds back POSITIVELY, so about the pivot P

    J_P·φ̈ = (M − M_crit) + W·z_CoM·φ ,      W·z_CoM > 0

is UNSTABLE. Writing d = W·z_CoM/J_P > 0 and using a constant moment ramp
(M − M_crit = Ṁ·τ, τ = t − t_crit) gives

    φ̈ − d·φ = Ṁ·τ / J_P ,

whose eigenvalues ±√d are real, so the solution is hyperbolic (not oscillatory).
At the onset the net moment vanishes, hence

    ω(t_crit) = 0    and    α(t_crit) = 0 ,

and the particular + homogeneous solution collapses to a single term:

    ω(τ) = C₁·(cosh(C₂·τ) − 1) + C ,     C₂ = √d ,   C = pre-onset baseline

Why nothing is free
-------------------
C₁ = Ṁ/(J_P·d), and since J_P·d = W·z_CoM,

    C₁ = Ṁ / (W·z_CoM) = K·Ṁ ,           K = 1/(W·z_CoM)

so the amplitude is fixed by the MEASURED ramp rate times a rig constant. With
C₂ and K pinned per rig and ω continuous at the onset (C = pre-onset median),
no shape parameter is free: the only unknown per run is the onset index, found
by an exhaustive sweep. No optimiser, no initial guess, no seed model.

(The time-quadratic model ω ∝ τ² is the d = 0 limit of the same solution — it
drops the very gravity term that creates the instability — and its truncation
error over the window actually used is 14–75%. It is not used here.)

Usage
-----
    python closed_form_onset.py DataSet/exp/case_05/My --axis y
    python closed_form_onset.py DataSet/exp/case_05/My --axis y --csv out.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import math_tools
from utils.extractor import BagData, load_excitation_dataset

# Rotor thrust coefficient and arm length of the test vehicle.
C_T = 1.3175e-7
ARM_LENGTH = 0.265


# ═════════════════════════════════════════════════════════════
#  1.  Signals
# ═════════════════════════════════════════════════════════════

def signals(bag: BagData, axis: str) -> dict:
    """Time-aligned body rate ω, collective thrust and axis moment.

    ω comes from the EKF odometry (gyro-derived, no mocap involved). The rotor
    RPM is mapped to thrust/moment and interpolated onto the ω timebase; t = 0
    is the start of the odometry stream.
    """
    k = 0 if axis == 'x' else 1
    t0 = bag.odom.t[0]

    t = bag.odom.t - t0
    omega = bag.odom.angular_vel[:, k]

    t_rpm = bag.rpm.t - t0
    thrust = math_tools.collective_thrust_vectorized(C_T, bag.rpm.rpm)
    moments = math_tools.rpm_to_moments_vectorized(
        C_T, bag.rpm.rpm, arm_length=ARM_LENGTH)

    return dict(t=t,
                omega=omega,
                f_col=np.interp(t, t_rpm, thrust),
                moment=np.interp(t, t_rpm, moments[:, k]))


def excitation_window(moment: np.ndarray, threshold: float = 0.01) -> slice:
    """Slice from the first |M| > threshold up to the peak |M|."""
    end = int(np.argmax(np.abs(moment)))
    above = np.where(np.abs(moment) > threshold)[0]
    start = int(above[0]) if len(above) and above[0] < end else 0
    return slice(start, end + 1)


def ramp_rate(t: np.ndarray, moment: np.ndarray) -> float:
    """Measured moment ramp rate Ṁ [N·m/s] — the signed slope of M(t).

    Fitted over the WHOLE window, not just after the onset: the commanded ramp
    is linear throughout, so the slope is the same either way, and this keeps Ṁ
    independent of the onset we are about to estimate (no circularity).
    """
    return float(np.polyfit(t, moment, 1)[0])


# ═════════════════════════════════════════════════════════════
#  2.  The onset — exhaustive sweep, zero free parameters
# ═════════════════════════════════════════════════════════════

def _model(tau: np.ndarray, c1: float, c2: float, c: float) -> np.ndarray:
    """ω(τ) = C₁(cosh(C₂τ) − 1) + C.  Clipped to avoid cosh overflow."""
    return c1 * (np.cosh(np.clip(c2 * tau, 0.0, 30.0)) - 1.0) + c


def find_onset(t, omega, c2: float, k: float, m_dot: float,
               min_pre: int = 8, min_post: int = 8, stride: int = 1) -> int:
    """Index of the onset that best splits ω into  [flat baseline | cosh rise].

    For every candidate split j the two segments are fully determined:
        C₁ = K·Ṁ                (physics — not fitted)
        C  = median(ω[:j])      (continuity — not fitted)
    so the cost is a plain arithmetic evaluation and the whole window can be
    swept. Cost = squared residual of both segments.
    """
    n = len(t)
    c1 = k * m_dot
    best_cost, best_j = np.inf, min_pre

    for j in range(min_pre, n - min_post, stride):
        tau = t[j:] - t[j]
        baseline = float(np.median(omega[:j]))
        post = omega[j:] - _model(tau, c1, c2, baseline)
        pre = omega[:j] - baseline
        cost = float(post @ post + pre @ pre)
        if cost < best_cost:
            best_cost, best_j = cost, j

    return best_j


# ═════════════════════════════════════════════════════════════
#  3.  Rig constants (C₂, K)
# ═════════════════════════════════════════════════════════════

def _consistency(runs, c2: float, k: float, stride: int) -> float:
    """Scatter of M_crit across ramp rates, summed over tip directions.

    M_crit is a STATIC threshold, so it must not depend on how fast the moment
    was ramped. That physical requirement is the selection criterion.

    Total residual cannot be used instead: with the onset swept freely, a
    smaller amplitude with an earlier onset trades off against a larger
    amplitude with a later one, leaving the residual degenerate along a ridge
    in (C₂, K). Ramp-rate invariance breaks that degeneracy with physics.
    """
    groups: dict[str, list[float]] = {}
    for r in runs:
        j = find_onset(r['t'], r['omega'], c2, k, r['m_dot'], stride=stride)
        groups.setdefault(r['side'], []).append(float(r['moment'][j]))

    score = 0.0
    for values in groups.values():
        if len(values) < 2:
            continue
        mean = abs(float(np.mean(values)))
        score += float(np.std(values)) / mean if mean > 1e-9 else np.inf
    return score


def rig_constants(runs, c2_range=(3.0, 8.0), k_range=(0.05, 0.70),
                  stride: int = 3) -> tuple[float, float]:
    """Estimate (C₂, K) once for the rig: coarse scan, then local refinement.

    The objective is piecewise CONSTANT (the onset is an integer index), so
    gradient and simplex methods stall on it; a scan is the honest tool. It is
    also flat along a ridge, so a full global optimiser buys nothing — the
    refinement pass only removes the coarse quantisation.
    """
    def scan(c2_values, k_values):
        best = (np.inf, c2_values[0], k_values[0])
        for c2 in c2_values:
            for k in k_values:
                score = _consistency(runs, float(c2), float(k), stride)
                if score < best[0]:
                    best = (score, float(c2), float(k))
        return best

    c2_coarse = np.arange(c2_range[0], c2_range[1] + 1e-9, 0.5)
    k_coarse = np.arange(k_range[0], k_range[1] + 1e-9, 0.05)
    _, c2, k = scan(c2_coarse, k_coarse)

    c2_fine = np.clip(np.linspace(c2 - 0.5, c2 + 0.5, 9), *c2_range)
    k_fine = np.clip(np.linspace(k - 0.05, k + 0.05, 9), *k_range)
    _, c2, k = scan(c2_fine, k_fine)
    return c2, k


# ═════════════════════════════════════════════════════════════
#  4.  Pipeline
# ═════════════════════════════════════════════════════════════

@dataclass
class Onset:
    """Identified onset for one run."""
    name: str
    side: str            # 'neg' | 'pos'  tip direction
    onset_time: float    # t_crit [s]
    critical_moment: float   # M_crit [N·m]   ← the deliverable
    thrust: float        # collective thrust at the onset [N]
    ramp_rate: float     # measured Ṁ [N·m/s]
    amplitude: float     # C₁ = K·Ṁ [rad/s]
    rmse: float          # fit residual over the window [rad/s]


def _runs(bags, axis: str) -> list[dict]:
    """Per-bag windowed signals + measured ramp rate."""
    out = []
    for bag in bags:
        s = signals(bag, axis)
        w = excitation_window(s['moment'])
        t, omega, moment = s['t'][w], s['omega'][w], s['moment'][w]
        if len(t) < 24:
            print(f"  [skip] {bag.name}: window too short ({len(t)} samples)")
            continue
        out.append(dict(name=bag.name,
                        side='neg' if bag.name.lower().startswith('neg') else 'pos',
                        t=t, omega=omega, moment=moment,
                        f_col=s['f_col'][w],
                        m_dot=ramp_rate(t, moment)))
    return out


def identify(bags, axis: str, c2: float | None = None,
             k: float | None = None) -> tuple[list[Onset], float, float]:
    """Full closed-form identification. Returns (onsets, C₂, K)."""
    runs = _runs(bags, axis)
    if not runs:
        raise SystemExit("no usable runs")

    if c2 is None or k is None:
        c2, k = rig_constants(runs)

    results = []
    for r in runs:
        j = find_onset(r['t'], r['omega'], c2, k, r['m_dot'])
        c1 = k * r['m_dot']
        baseline = float(np.median(r['omega'][:j]))
        pred = np.full(len(r['t']), baseline)
        pred[j:] = _model(r['t'][j:] - r['t'][j], c1, c2, baseline)
        results.append(Onset(
            name=r['name'], side=r['side'],
            onset_time=float(r['t'][j]),
            critical_moment=float(r['moment'][j]),
            thrust=float(r['f_col'][j]),
            ramp_rate=r['m_dot'], amplitude=c1,
            rmse=float(np.sqrt(np.mean((r['omega'] - pred) ** 2)))))
    return results, c2, k


# ═════════════════════════════════════════════════════════════
#  5.  CLI
# ═════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Closed-form (cosh) tip-over onset identification.")
    p.add_argument('data_dir')
    p.add_argument('--axis', choices=['x', 'y'], required=True,
                   help="x = roll (Mx), y = pitch (My)")
    p.add_argument('--c2', type=float, default=None,
                   help="pin C₂ [rad/s] instead of estimating it")
    p.add_argument('--k', type=float, default=None,
                   help="pin K = 1/(W·z_CoM) instead of estimating it")
    p.add_argument('--csv', type=str, default=None, help="write results here")
    args = p.parse_args()

    bags = load_excitation_dataset(Path(args.data_dir))
    onsets, c2, k = identify(bags, args.axis, c2=args.c2, k=args.k)

    wz = 1.0 / k
    print(f"\nRig constants : C₂ = {c2:.3f} rad/s,  K = {k:.4f}")
    print(f"                → W·z_CoM = {wz:.2f} N·m,  "
          f"J_P = {wz / c2 ** 2:.3f} kg·m²  (sanity check only)\n")

    print(f"{'run':14} {'t_crit[s]':>10} {'M_crit[N·m]':>12} "
          f"{'Mdot':>7} {'C1':>8} {'RMSE':>8}")
    print("-" * 64)
    for o in sorted(onsets, key=lambda x: (x.side, x.name)):
        print(f"{o.name:14} {o.onset_time:10.3f} {o.critical_moment:+12.4f} "
              f"{o.ramp_rate:+7.3f} {o.amplitude:+8.4f} {o.rmse:8.4f}")
    print("-" * 64)

    for side in ('neg', 'pos'):
        vals = [o.critical_moment for o in onsets if o.side == side]
        if vals:
            print(f"  {side}: M_crit = {np.mean(vals):+.4f} ± {np.std(vals):.4f} "
                  f"N·m  ({len(vals)} runs, spread "
                  f"{100 * np.std(vals) / abs(np.mean(vals)):.1f}%)")

    neg = [o.critical_moment for o in onsets if o.side == 'neg']
    pos = [o.critical_moment for o in onsets if o.side == 'pos']
    if neg and pos:
        # pivot-free moment offset: the common pivot lever cancels in the mean,
        # and any landing-gear asymmetry is retained (a CoM→moment conversion
        # would drop it).
        print(f"\n  Moment offset  M_ff = ½(M_pos + M_neg) = "
              f"{0.5 * (np.mean(pos) + np.mean(neg)):+.4f} N·m")

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run', 'side', 't_crit_s', 'M_crit_Nm', 'thrust_N',
                        'ramp_rate_Nm_s', 'C1_rad_s', 'rmse_rad_s'])
            for o in onsets:
                w.writerow([o.name, o.side, f"{o.onset_time:.4f}",
                            f"{o.critical_moment:.6f}", f"{o.thrust:.4f}",
                            f"{o.ramp_rate:.6f}", f"{o.amplitude:.6f}",
                            f"{o.rmse:.6f}"])
        print(f"\nCSV → {args.csv}")


if __name__ == '__main__':
    main()
