#!/usr/bin/env python3
"""
Moment Excitation Analysis — Onset Detection + Pivot Estimation

Onset detection (default) via the closed-form solution of the linearised,
UNSTABLE tip-over dynamics (φ̈ − dφ = G, d = W·z_CoM/J_P > 0):

    ω(τ) = C₁·(cosh(C₂·τ) − 1) + C,   τ = t − t0,  C₂ = √d

Exact over the excitation segment; monotonic; quadratic (C₂τ)²/2 near onset,
exponential runaway far from it. The onset t0 is swept over the window.
The time-quadratic fit is only the small-angle limit of this form (its error
grows with tilt) and is kept as a comparison baseline (--model piecewise).

Usage
-----
python critical_value_getter_piecewise.py DataSet/exp/Mx
python critical_value_getter_piecewise.py DataSet/exp/My --mass 3.066 --save-fig

# 95% confidence intervals for the identified quantities:
python critical_value_getter_piecewise.py DataSet/exp/My --ci

# Use raw IMU angular velocity (/mavros/imu/data_raw) instead of odom:
python critical_value_getter_piecewise.py DataSet/exp/My --omega-source imu --lpf-cutoff 15

# Comparison baseline only — time-quadratic (small-angle limit):
python critical_value_getter_piecewise.py DataSet/exp/My --model piecewise
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
#  Bag Name → Plot Title
# ═════════════════════════════════════════════════════════════

def bag_name_to_title(bag_name: str) -> str:
    r"""
    Convert bag name to formatted plot title.

    Examples:
        pos_Mx_01 → r'$\dot{M}_x = 0.1$ Nm/s'
        neg_Mx_01 → r'$\dot{M}_x = -0.1$ Nm/s'
        neg_My_03 → r'$\dot{M}_y = -0.3$ Nm/s'
        My_pos_02 → r'$\dot{M}_y = 0.2$ Nm/s'
    """
    name = bag_name.lower()

    # Detect axis
    if 'mx' in name:
        ax = 'x'
    elif 'my' in name:
        ax = 'y'
    else:
        return bag_name

    # Detect direction
    if 'pos' in name:
        sign = ''
    elif 'neg' in name:
        sign = '-'
    else:
        return bag_name

    # Detect trial number → ramp rate
    ramp_map = {'01': '0.1', '02': '0.2', '03': '0.3'}
    ramp = None
    for key, val in ramp_map.items():
        if key in name:
            ramp = val
            break
    if ramp is None:
        return bag_name

    return r'$\dot{M}_' + ax + ' = ' + sign + ramp + r'$ Nm/s'


# ═════════════════════════════════════════════════════════════
#  Low-pass filter (for noisy IMU angular velocity)
# ═════════════════════════════════════════════════════════════

def lowpass_filter(
    t: np.ndarray,
    x: np.ndarray,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Sampling rate is estimated from the median time step of `t`. Useful for
    the raw IMU angular velocity (/mavros/imu/data_raw), which carries heavy
    propeller vibration well above the tip-over dynamics.

    Parameters
    ----------
    t         : (N,) time array [s]
    x         : (N,) signal to filter
    cutoff_hz : cutoff frequency [Hz]
    order     : Butterworth order (applied twice by filtfilt)

    Returns
    -------
    (N,) filtered signal (zero phase lag).
    """
    from scipy.signal import butter, filtfilt  # lazy: optional dependency

    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff {cutoff_hz} Hz must be below Nyquist {nyq:.1f} Hz "
            f"(sample rate {fs:.1f} Hz)."
        )
    b, a = butter(order, cutoff_hz, btype='low', fs=fs)
    return filtfilt(b, a, x)


# ═════════════════════════════════════════════════════════════
#  Robust (Huber) helpers
# ═════════════════════════════════════════════════════════════

def _mad_scale(r: np.ndarray) -> float:
    """Robust noise scale via MAD = Median Absolute Deviation
    (normalized by 1.4826 so it matches the standard deviation for Gaussian
    data)."""
    med = np.median(r)
    return 1.4826 * np.median(np.abs(r - med)) + 1e-12


def _huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    """Huber IRLS (Iteratively Reweighted Least Squares) weights:
    1 for |r|<=delta, delta/|r| beyond (outliers)."""
    a = np.abs(r)
    w = np.ones_like(a)
    m = a > delta
    w[m] = delta / a[m]
    return w


def _huber_cost(r: np.ndarray, delta: float) -> float:
    """Huber loss: quadratic within delta, linear (outlier-robust) beyond."""
    a = np.abs(r)
    return float(np.sum(np.where(a <= delta, 0.5 * r ** 2,
                                 delta * (a - 0.5 * delta))))


def _fit_segments_robust(t, omega, j, delta, n_irls=5, sides='pre'):
    """
    Fit (c, α) for split index j and return the total cost used to compare
    candidate onsets. Robust segments use Huber IRLS
    (Iteratively Reweighted Least Squares); n_irls = number of reweighting
    iterations.

      left  : ω = c            (should be flat → deviations are vibration)
      right : ω = α·(t-t0)²+c  (real tip-over rise)

    sides
    -----
    'pre'  : Huber IRLS on the LEFT segment only (reject pre-onset vibration
             outliers), ordinary least squares on the RIGHT segment so the
             genuine tip-over dynamics are not down-weighted.
    'both' : Huber IRLS on both segments.

    Costs are kept in consistent (½·squared) scale so the mixed objective is
    comparable across candidate split points.
    """
    left = omega[:j]
    right = omega[j:]
    dt2 = (t[j:] - t[j]) ** 2

    # Left constant: robust Huber IRLS (Iteratively Reweighted Least Squares)
    # — down-weight vibration outliers
    c = np.median(left)
    for _ in range(n_irls):
        w = _huber_weights(left - c, delta)
        c = np.sum(w * left) / np.sum(w)
    left_cost = _huber_cost(left - c, delta)

    # Right α given c
    den0 = np.sum(dt2 * dt2)
    alpha = np.sum(dt2 * (right - c)) / den0 if den0 > 1e-30 else 0.0
    if sides == 'both':
        for _ in range(n_irls):
            w = _huber_weights(right - (alpha * dt2 + c), delta)
            den = np.sum(w * dt2 * dt2)
            if den < 1e-30:
                break
            alpha = np.sum(w * dt2 * (right - c)) / den
        right_cost = _huber_cost(right - (alpha * dt2 + c), delta)
    else:  # 'pre': keep ordinary least squares on the rise
        right_cost = 0.5 * np.sum((right - (alpha * dt2 + c)) ** 2)

    return c, alpha, left_cost + right_cost


# ═════════════════════════════════════════════════════════════
#  Piecewise Onset Detection
# ═════════════════════════════════════════════════════════════

def piecewise_onset_fit(
    t: np.ndarray,
    omega: np.ndarray,
    min_seg: int = 5,
    robust: bool = False,
    huber_k: float = 1.345,
    n_irls: int = 5,
    robust_sides: str = 'pre',
) -> dict:
    """
    Fit piecewise model to angular velocity:
        ω(t) = c              for t < t0  (ground contact, ω̇ = 0)
        ω(t) = α·(t-t0)² + c  for t ≥ t0  (tip-over, ω̇(t0) = 0)

    Physical basis: at onset (N=0), angular acceleration starts from zero.
    ω̇ ∝ (t - t0) → ω ∝ (t - t0)²

    Sweep t0 over all candidates in [min_seg, N-min_seg],
    solve (c, α) at each candidate, pick t0* = argmin total_residual.

    Parameters
    ----------
    t       : (N,) time array
    omega   : (N,) angular velocity
    min_seg : minimum segment length
    robust  : if True, robustify the fit so pre-onset vibration outliers are
              down-weighted (Huber IRLS = Iteratively Reweighted Least
              Squares). Onset = argmin total cost.
    huber_k : Huber threshold in units of the robust noise scale
              (MAD = Median Absolute Deviation based); 1.345 gives 95%
              Gaussian efficiency. Residuals beyond huber_k·σ are treated as
              outliers and down-weighted.
    n_irls  : number of IRLS (Iteratively Reweighted Least Squares) iterations
              per segment fit.
    robust_sides : 'pre'  → Huber on the pre-onset (flat) segment only, plain
                            LS on the rise so tip-over dynamics are preserved
                            (recommended). 'both' → Huber on both segments.

    Returns
    -------
    dict with: onset_idx, c, alpha, total_residual, omega_pred, rmse
               (robust mode also adds huber_delta)
    """
    N = len(t)
    best_res = np.inf
    best_idx = N // 2
    best_params = None

    for j in range(min_seg, N - min_seg):
        # Left: ω = c
        left = omega[:j]
        c = np.mean(left)
        res_left = np.sum((left - c) ** 2)

        # Right: ω - c = α·dt²  (single parameter LS)
        right = omega[j:]
        dt = t[j:] - t[j]
        if len(dt) < 2:
            continue
        dt2 = dt ** 2
        y = right - c
        denom = np.sum(dt2 ** 2)
        if denom < 1e-30:
            continue
        alpha = np.sum(dt2 * y) / denom
        pred_right = alpha * dt2 + c
        res_right = np.sum((right - pred_right) ** 2)

        total_res = res_left + res_right
        if total_res < best_res:
            best_res = total_res
            best_idx = j
            best_params = (c, alpha)

    huber_delta = None
    if robust:
        # Global noise scale from the (outlier-robust) MAD = Median Absolute
        # Deviation of the L2 residuals, so the Huber threshold and cost are
        # comparable across candidates.
        c0, a0 = best_params
        pred0 = np.full_like(omega, c0)
        aft0 = t >= t[best_idx]
        pred0[aft0] = a0 * (t[aft0] - t[best_idx]) ** 2 + c0
        huber_delta = huber_k * _mad_scale(omega - pred0)

        best_cost = np.inf
        for j in range(min_seg, N - min_seg):
            if len(t[j:]) < 2:
                continue
            c_j, a_j, cost = _fit_segments_robust(
                t, omega, j, huber_delta, n_irls, sides=robust_sides)
            if cost < best_cost:
                best_cost = cost
                best_idx = j
                best_params = (c_j, a_j)
        best_res = best_cost

    # Build full prediction
    c, alpha = best_params
    omega_pred = np.full_like(omega, c)
    t0 = t[best_idx]
    after = t >= t0
    dt_after = t[after] - t0
    omega_pred[after] = alpha * dt_after ** 2 + c

    rmse = np.sqrt(np.mean((omega - omega_pred) ** 2))

    return dict(
        onset_idx=best_idx,
        c=c, alpha=alpha,
        total_residual=best_res,
        omega_pred=omega_pred,
        rmse=rmse,
        huber_delta=huber_delta,
    )




# Pre-onset baseline statistic for the onset sweep. 'median' (default) is
# robust to the heavy-tailed gyro noise and to the cosh rise leaking into the
# pre-segment when a candidate onset overshoots; 'mean' (LS-optimal constant)
# and 'zero' (the ideal-sensor value implied by the onset conditions) are kept
# only for the ablation (analysis/baseline_stat_ablation.py).
BASELINE_STAT = 'median'


def _baseline_of(x: np.ndarray) -> float:
    if BASELINE_STAT == 'mean':
        return float(np.mean(x))
    if BASELINE_STAT == 'zero':
        return 0.0
    return float(np.median(x))


def cosh_onset_fit(t, omega, moment, onset_guess,
                   sweep_back_s=0.10, sweep_ahead_s=0.30, step_s=0.01,
                   c2_bounds=(3.0, 8.0), moment_floor=0.30, c2_fixed=None,
                   moment_floor_abs=None, ramp_gain=None, ramp_rate=None):
    """
    Onset detection with the closed-form tip-over solution.

    Linearising the dynamics (sinφ≈φ, cosφ≈1) gives φ̈ − dφ = G(t). Because
    the tip-over past the balance point is UNSTABLE (d > 0, positive
    feedback), the eigenvalues are real (±√d) and the exact solution is
    hyperbolic. With the physical onset conditions ω(t_crit)=0 and α(t_crit)=0
    (critical = boundary of static equilibrium) it collapses to

        ω(τ) = C₁·(cosh(C₂·τ) − 1) + C,   τ = t − t_crit,  C₂ = √d

    which is monotonic (no spurious oscillation), reduces to the PLS quadratic
    (C₂τ)²/2 for small τ, and grows exponentially for large τ. A constant
    moment rate Ṁ enters only through the amplitude C₁ = a·Ṁ/d, so no explicit
    polynomial term is needed (it is already the leading term of cosh−1).

    Robustness at high moment-ramp rate (short post-onset window)
    ------------------------------------------------------------
    Three constraints keep the joint-residual sweep well-posed when the
    excitation is fast (few post-onset samples), where the plain fit
    otherwise degenerates and places the onset far too early:

      * C₂ = √(W·z_CoM/J) is a *physical, ramp-independent rig constant*. The
        preferred use is to estimate it ONCE for the rig (``estimate_shared_c2``)
        and pass it as ``c2_fixed`` — then only (C₁, C) are fit per bag, which is
        well conditioned even on short windows and keeps C₂ consistent across
        trials (per-bag C₂ otherwise scatters and saturates the bounds). If
        ``c2_fixed`` is None, C₂ is fit per bag but bounded to ``c2_bounds``
        (default [3, 8] rad/s); leaving it wide open lets a huge C₂ mimic
        "flat-then-spike", fitting a late rise from an arbitrarily early split.
      * a tip-over cannot begin below the static tip-over threshold, so the
        onset search is floored in moment. ``moment_floor_abs`` (absolute, N·m)
        is preferred: the critical moment is a physical constant with a known
        lower bound per axis (e.g. |M_y|≳0.4, |M_x|≳0.7 N·m for this rig), and
        an absolute floor is peak-independent — it cannot be under-cut on a
        fast ramp the way a fraction-of-peak floor can. If ``moment_floor_abs``
        is None, fall back to the relative floor ``moment_floor``·max|M|. Either
        way this rejects the pre-excitation ω transient (which sits at near-zero
        moment) that would otherwise capture the onset.

    The sweep is forward-biased (small ``sweep_back_s``) so it refines the
    reliable seed rather than escaping backward into that transient.

    Returns the same core keys as piecewise_onset_fit; 'alpha' carries C₂
    (the instability rate √d) and 'c' the baseline for CSV compatibility.
    """
    from scipy.optimize import least_squares  # lazy: optional dependency

    N = len(t)
    dt = float(np.median(np.diff(t)))
    full_sweep = onset_guess is None
    if full_sweep:
        # no seed supplied: only used below for the baseline/sign heuristics
        onset_guess = N // 2
    lo = max(1, onset_guess - int(round(sweep_back_s / dt)))
    hi = min(N - 8, onset_guess + int(round(sweep_ahead_s / dt)))
    step = max(1, int(round(step_s / dt)))

    # a tip-over cannot begin below the static threshold: floor the onset search
    # in moment (absolute floor preferred; peak-fraction fallback). Rejects the
    # pre-onset ω transient, which sits at near-zero moment.
    if len(moment) == N:
        if moment_floor_abs is not None:
            above = np.where(np.abs(moment) >= float(moment_floor_abs))[0]
        elif moment_floor > 0.0:
            peak = float(np.max(np.abs(moment)))
            above = np.where(np.abs(moment) >= moment_floor * peak)[0]
        else:
            above = np.array([], dtype=int)
        if len(above):
            lo = max(lo, int(above[0]))

    # tip-over direction (sign of ω at the window tail vs baseline) to init C₁
    base0 = float(np.median(omega[:max(1, onset_guess)]))
    sgn = 1.0 if float(np.mean(omega[int(0.85 * N):])) >= base0 else -1.0

    # ── Fully constrained mode: no free shape parameter ──────────────
    # At the onset the net moment vanishes (M = M_crit ⇒ α = 0), and the
    # linearised dynamics J_P·α = Ṁ·τ + W·z_CoM·φ integrate to
    #     ω(τ) = [Ṁ/(J_P·d)]·(cosh(C₂τ) − 1),   d = W·z_CoM/J_P = C₂²
    # so the amplitude is NOT free: J_P·d = W·z_CoM gives
    #     C₁ = Ṁ / (W·z_CoM) = ramp_gain · Ṁ .
    # With C₂ (rig constant) pinned, Ṁ measured, and ω continuous at the onset
    # (C = pre-onset baseline), every parameter is determined and only the onset
    # index is searched. This removes the amplitude↔onset trade-off that makes
    # the free fit drift early and SNR-dependent on weakly excited runs.
    if (ramp_gain is not None and ramp_rate is not None
            and c2_fixed is not None):
        c2_val = float(c2_fixed)
        C1_fix = float(ramp_gain) * float(ramp_rate)
        # With no free shape parameter the cost at each candidate onset is a
        # plain arithmetic evaluation — no optimiser, no initial guess — so the
        # WHOLE window is swept exhaustively and no seed is needed. This keeps
        # the pipeline free of the quadratic (small-angle) model entirely.
        if full_sweep:
            lo, hi = 8, max(9, N - 8)
        best = (np.inf, lo, None)
        cost_of = {}
        for j in range(lo, max(lo + 1, hi), step):
            tau = t[j:] - t[j]
            gfun = np.cosh(np.clip(c2_val * tau, 0, 30)) - 1
            C0 = _baseline_of(omega[:j]) if j > 0 else 0.0
            post = np.sum((omega[j:] - (C1_fix * gfun + C0)) ** 2)
            pre = np.sum((omega[:j] - C0) ** 2) if j > 0 else 0.0
            cost = float(post + pre)
            cost_of[j] = cost
            if cost < best[0]:
                best = (cost, j, (C1_fix, c2_val, C0))
        cost, j_star, params = best
        # Sub-sample refinement.  The cost is a smooth function of the
        # CONTINUOUS onset -- the linearised form is exactly quadratic in
        # it -- and the sweep only samples that function on the data grid,
        # so the grid minimum is short of the true one by up to half a
        # step.  A parabola through the three costs around j_star has its
        # vertex at the sub-sample offset below.  This is not a fitted
        # curve but the correct local model, and it turns an onset error
        # of order Ts into one of order Ts^2 times the local curvature.
        frac = 0.0
        a = cost_of.get(j_star - step)
        c_ = cost_of.get(j_star + step)
        if a is not None and c_ is not None:
            den = a - 2.0 * cost + c_
            if den > 0.0:
                frac = float(np.clip(0.5 * (a - c_) / den, -0.5, 0.5)) * step
        C1, C2, C = params
        tau = t[j_star:] - t[j_star]
        omega_pred = np.full(N, float(C))
        omega_pred[j_star:] = C1 * (np.cosh(np.clip(C2 * tau, 0, 30)) - 1) + C
        return dict(
            onset_idx=j_star,
            onset_frac=float(frac),
            onset_t=float(t[j_star] + frac * dt),
            c=float(C), alpha=float(C2),
            total_residual=float(cost),
            omega_pred=omega_pred,
            rmse=float(np.sqrt(np.mean((omega - omega_pred) ** 2))),
            huber_delta=None,
            params=tuple(float(x) for x in params),
            model='cosh',
        )

    if c2_fixed is not None:
        # C₂ pinned to the shared rig constant: fit only (C₁, C) per bag.
        c2_val = float(c2_fixed)

        def model(p, tau):
            C1, C = p
            return C1 * (np.cosh(np.clip(c2_val * tau, 0, 30)) - 1) + C

        p0 = lambda C0: [sgn * 1e-3, C0]
        bounds = ([-5.0, -2.0], [5.0, 2.0])
        expand = lambda x: (float(x[0]), c2_val, float(x[1]))
    else:
        # C₂ fit per bag but tightly bounded to the physical band.
        c2_0 = float(np.clip(4.9, c2_bounds[0], c2_bounds[1]))

        def model(p, tau):
            C1, C2, C = p
            return C1 * (np.cosh(np.clip(C2 * tau, 0, 30)) - 1) + C

        p0 = lambda C0: [sgn * 1e-3, c2_0, C0]
        bounds = ([-5.0, c2_bounds[0], -2.0], [5.0, c2_bounds[1], 2.0])
        expand = lambda x: (float(x[0]), float(x[1]), float(x[2]))

    def _sweep(candidates, best):
        for j in candidates:
            tau = t[j:] - t[j]
            y = omega[j:]
            C0 = _baseline_of(omega[:j]) if j > 0 else 0.0
            r = least_squares(lambda p: model(p, tau) - y,
                              p0(C0), method='trf', bounds=bounds,
                              max_nfev=300)
            pre = np.sum((omega[:j] - C0) ** 2) if j > 0 else 0.0
            cost = float(np.sum(r.fun ** 2) + pre)
            if cost < best[0]:
                best = (cost, j, r.x)
        return best

    best = (np.inf, onset_guess, None)
    if full_sweep:
        # No seed of any kind: sweep the whole window with the cosh
        # model alone, coarse first, then refine around the coarse
        # minimum at full resolution.  The cost is smooth in the onset
        # (near the optimum it is locally quadratic), so a coarse pass
        # of ~40 candidates brackets the basin and the fine pass
        # recovers the seeded sweep's resolution -- with no quadratic
        # (small-angle) model anywhere in the path.
        lo, hi = 8, max(9, N - 8)
        coarse = max(step, (hi - lo) // 40)
        best = _sweep(range(lo, hi, coarse), best)
        j0 = best[1]
        best = _sweep(range(max(lo, j0 - coarse),
                            min(hi, j0 + coarse + 1), step), best)
    else:
        best = _sweep(range(lo, max(lo + 1, hi), step), best)

    cost, j_star, x_star = best
    params = expand(x_star)
    C1, C2, C = params
    tau = t[j_star:] - t[j_star]
    omega_pred = np.full(N, float(C))
    omega_pred[j_star:] = C1 * (np.cosh(np.clip(C2 * tau, 0, 30)) - 1) + C
    rmse = float(np.sqrt(np.mean((omega - omega_pred) ** 2)))

    return dict(
        onset_idx=j_star,
        c=float(C), alpha=float(C2),
        total_residual=float(cost),
        omega_pred=omega_pred,
        rmse=rmse,
        huber_delta=None,
        params=tuple(float(x) for x in params),
        model='cosh',
    )


def estimate_shared_c2(bags, axis, min_post=45, c2_bounds=(3.0, 8.0), **kwargs):
    """
    Estimate the shared instability rate C₂ = √(W·z_CoM/J) for a rig.

    C₂ is a physical constant of the test rig (weight, CoM height, inertia), so
    it is the SAME for every bag of one configuration — it does not depend on
    the excitation ramp rate or tip direction. Fitting it independently per bag
    lets it scatter (and saturate the bounds) on short/fast windows, which is
    unphysical. Instead we fit it per bag ONLY on the reliable long-window bags
    (≥ ``min_post`` post-onset samples) and take the robust median; the caller
    then pins every bag to this single value (``c2_fixed``).

    Returns the median C₂ (float), clipped to ``c2_bounds``. Falls back to the
    band centre if no bag has a long enough window.
    """
    vals = []
    for bag in bags:
        try:
            crit, pw = extract_piecewise(bag, axis, model='cosh',
                                         c2_bounds=c2_bounds, **kwargs)
        except Exception:
            continue
        i0, i1 = detect_excitation_window(crit.moment)
        if (i1 - crit.onset_idx) >= min_post and pw.get('alpha') is not None:
            vals.append(float(pw['alpha']))
    if not vals:
        return float(0.5 * (c2_bounds[0] + c2_bounds[1]))
    return float(np.clip(np.median(vals), c2_bounds[0], c2_bounds[1]))


def estimate_ramp_gain(bags, axis, c2, k_grid=None, **kwargs):
    """
    Estimate the shared ramp gain K = 1/(W·z_CoM) [rad/(s·N·m)] for a rig.

    The onset conditions (net moment zero ⇒ α = 0) fix the closed-form
    amplitude to C₁ = Ṁ/(W·z_CoM) = K·Ṁ, so K — like C₂ — is a property of the
    rig, not of the individual run. It is estimated once by choosing the K that
    minimises the total two-segment residual over the whole dataset (each bag
    contributes its own measured ramp rate Ṁ), then pinned for every bag.

    K and C₂ are not independent: together they imply W·z_CoM = 1/K and
    J_P = W·z_CoM/C₂², which is a useful physical sanity check on the fit.
    """
    if k_grid is None:
        k_grid = np.arange(0.04, 0.60, 0.01)
    prepared = []
    for bag in bags:
        try:
            crit, _ = extract_piecewise(bag, axis, model='piecewise', **kwargs)
        except Exception:
            continue
        i0, i1 = detect_excitation_window(crit.moment)
        win = slice(i0, i1 + 1)
        t, om, M = crit.t[win], crit.omega[win], crit.moment[win]
        if len(t) < 12:
            continue
        guess = piecewise_onset_fit(t, om)['onset_idx']
        m_dot = float(np.polyfit(t, M, 1)[0])
        prepared.append((t, om, guess, m_dot))
    if not prepared:
        return None

    best = (np.inf, float(k_grid[0]))
    for K in k_grid:
        total = 0.0
        for t, om, guess, m_dot in prepared:
            pw = cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=guess,
                                c2_fixed=c2, moment_floor=0.0,
                                ramp_gain=float(K), ramp_rate=m_dot)
            total += pw['total_residual']
        if total < best[0]:
            best = (total, float(K))
    return best[1]


def estimate_rig_constants(bags, axis, c2_grid=None, k_grid=None, stride=3,
                           method='grid', c2_range=(3.0, 8.0),
                           k_range=(0.05, 0.70), seed=0, **kwargs):
    """
    Estimate (C₂, K) for a rig using ONLY the closed-form model.

    Both are rig constants: C₂ = √(W·z_CoM/J_P) and K = 1/(W·z_CoM). They are
    selected on a 2-D grid, and the selection criterion is *physical
    self-consistency* rather than total residual:

        M_crit is a STATIC tip-over threshold, so it must not depend on how
        fast the moment was ramped. We therefore pick the (C₂, K) that make the
        identified M_crit most repeatable across ramp rates, scored as the sum
        over tip directions of the coefficient of variation std|M|/|mean M|.

    Total-residual minimisation cannot be used here: with the onset swept over
    the whole window, a smaller amplitude with an earlier onset trades off
    against a larger amplitude with a later one, so the residual is degenerate
    along a ridge in (C₂, K). The ramp-rate invariance of M_crit breaks that
    degeneracy using physics rather than an auxiliary (small-angle) model.

    Returns (c2, k). Note the two constants are individually only loosely
    determined (the ridge is flat); the identified onset is nevertheless robust
    along it, so W·z_CoM = 1/K and J_P = 1/(K·C₂²) should be read as
    order-of-magnitude sanity checks, not precision measurements.
    """
    if c2_grid is None:
        c2_grid = np.arange(3.0, 8.01, 0.5)
    if k_grid is None:
        k_grid = np.arange(0.10, 0.52, 0.04)

    prepared = []
    for bag in bags:
        try:
            crit = prepare_signals(bag, axis, **kwargs)
        except Exception:
            continue
        i0, i1 = detect_excitation_window(crit['moment'])
        win = slice(i0, i1 + 1)
        t, om, M = crit['t'][win], crit['omega'][win], crit['moment'][win]
        if len(t) < 24:
            continue
        side = 'neg' if bag.name.lower().startswith('neg') else 'pos'
        prepared.append((side, t - t[0], om, M, float(np.polyfit(t, M, 1)[0])))
    if not prepared:
        return None, None

    def onset_moment(t, om, M, c2, k, m_dot, strd):
        pw = cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                            c2_fixed=float(c2), moment_floor=0.0,
                            ramp_gain=float(k), ramp_rate=m_dot,
                            step_s=strd * float(np.median(np.diff(t))))
        return float(M[pw['onset_idx']])

    def score_of(c2, k, strd):
        groups = {}
        for side, t, om, M, m_dot in prepared:
            groups.setdefault(side, []).append(
                onset_moment(t, om, M, c2, k, m_dot, strd))
        score = 0.0
        for vals in groups.values():
            if len(vals) < 2:
                continue
            mu = abs(float(np.mean(vals)))
            score += float(np.std(vals)) / mu if mu > 1e-9 else np.inf
        return score

    if method == 'de':
        # The objective is piecewise CONSTANT in (C₂, K): the onset is an
        # integer index, so small parameter changes leave it (and therefore
        # M_crit) unchanged. Gradient-based and simplex methods stall on such a
        # staircase, so use a population-based derivative-free global search.
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            lambda p: score_of(float(p[0]), float(p[1]), stride),
            bounds=[tuple(c2_range), tuple(k_range)],
            popsize=10, maxiter=25, tol=0.0, mutation=(0.3, 0.9),
            recombination=0.8, polish=False, seed=seed, init='sobol')
        return float(res.x[0]), float(res.x[1])

    def scan(c2s, ks):
        best = (np.inf, None, None)
        for c2 in c2s:
            for k in ks:
                score = score_of(float(c2), float(k), stride)
                if score < best[0]:
                    best = (score, float(c2), float(k))
        return best

    # coarse scan, then one refinement pass around the winner so the reported
    # constants are not quantised to the coarse step. The objective is flat
    # along a (C₂, K) ridge, so this converges immediately — a full global
    # optimiser (method='de') costs ~4x more for no measurable gain.
    _, c2_b, k_b = scan(c2_grid, k_grid)
    dc2 = float(np.diff(c2_grid)[0]) if len(c2_grid) > 1 else 0.5
    dk = float(np.diff(k_grid)[0]) if len(k_grid) > 1 else 0.04
    c2_fine = np.clip(np.linspace(c2_b - dc2, c2_b + dc2, 9), *c2_range)
    k_fine = np.clip(np.linspace(k_b - dk, k_b + dk, 9), *k_range)
    _, c2_b, k_b = scan(c2_fine, k_fine)
    return c2_b, k_b


# Allocator feasibility limits per axis [N.m]: past these commanded moments
# the thrust allocation saturates, the executed ramp is no longer linear and
# the closed-form (cosh-family) model no longer applies — the window is
# truncated where |M| first exceeds the cap.
MOMENT_CAP = {'x': 2.37, 'y': 2.74}

# Slope-error gate floor [N·m]: the ramp-tracking check is evaluated on the
# sub-segment |M| >= floor only. The floors are the per-axis lower bounds of
# |M_crit| over the ±20 mm CoM-offset design box ((W-f)·l_p − W·0.020 with
# margin), so the gate scores exactly the segment where an onset can occur —
# early spin-up imperfections at near-zero moment cannot reject a run whose
# ramp is within spec where it matters. Identification's Ṁ estimate and the
# linearity check remain full-window.
SLOPE_GATE_FLOOR = {'x': 0.7, 'y': 0.4}


def detect_excitation_window(
    moment: np.ndarray,
    threshold: float = 0.01,
    moment_cap: Optional[float] = None,
) -> tuple[int, int]:
    """
    Find excitation window: [start of the ramp leading to the peak, max|M|].

    The start is the LAST sub-threshold sample before the peak (walking
    backwards from max|M|), not the first supra-threshold sample. The two are
    identical when |M| grows monotonically, but the backward definition is
    robust to small |M| blips before the actual ramp — e.g. the rotor spin-up
    transient during arming, which on one run crossed the threshold ~1.6 s
    early and pulled a flat idle stretch into the window, corrupting the
    measured ramp rate by −31%.

    ``moment_cap`` truncates the window where |M| first exceeds the allocator
    feasibility limit (see ``MOMENT_CAP``): beyond it the allocation
    saturates, the ramp is no longer linear and the cosh family does not
    apply. On the reference dataset the cap never binds (peak |M| ≤ 1.7 N·m).
    """
    idx_end = int(np.argmax(np.abs(moment)))
    if moment_cap is not None:
        over = np.where(np.abs(moment) > float(moment_cap))[0]
        if len(over) > 0 and over[0] <= idx_end:
            idx_end = max(1, int(over[0]) - 1)
    below = np.where(np.abs(moment[:idx_end]) <= threshold)[0]
    if len(below) > 0:
        return int(min(below[-1] + 1, idx_end)), idx_end
    return 0, idx_end


def prepare_signals(
    bag: BagData,
    axis: str,
    C_T: float = 1.3175e-7,
    arm_length: float = 0.265,
    omega_source: str = 'odom',
    lpf_cutoff: Optional[float] = None,
    lpf_order: int = 4,
) -> dict:
    """
    Time-aligned signals for one bag: ω, collective thrust and axis moment.

    Pure signal preparation — no onset model of any kind is fitted here, so
    callers that must stay free of a particular onset model (e.g. the rig
    constant estimation) can obtain the raw traces without one. The global time
    reference is odom.t[0] regardless of ``omega_source``.
    """
    t0_ref = bag.odom.t[0]
    axis_idx = 0 if axis == 'x' else 1

    if omega_source == 'imu':
        if bag.imu is None:
            raise ValueError(
                f"{bag.name}: --omega-source imu requested but "
                f"/mavros/imu/data_raw is not present in this bag."
            )
        t = bag.imu.t - t0_ref
        omega = bag.imu.angular_vel[:, axis_idx]
    else:
        t = bag.odom.t - t0_ref
        omega = bag.odom.angular_vel[:, axis_idx]

    if lpf_cutoff is not None:
        omega = lowpass_filter(t, omega, lpf_cutoff, order=lpf_order)

    t_rpm = bag.rpm.t - t0_ref
    f_col_raw = math_tools.collective_thrust_vectorized(C_T, bag.rpm.rpm)
    moments_raw = math_tools.rpm_to_moments_vectorized(
        C_T, bag.rpm.rpm, arm_length=arm_length,
    )
    return dict(
        t=t,
        omega=omega,
        f_col=np.interp(t, t_rpm, f_col_raw),
        moment=np.interp(t, t_rpm, moments_raw[:, axis_idx]),
    )


def extract_piecewise(
    bag: BagData,
    axis: str,
    C_T: float = 1.3175e-7,
    arm_length: float = 0.265,
    threshold: float = 0.01,
    omega_source: str = 'odom',
    lpf_cutoff: Optional[float] = None,
    lpf_order: int = 4,
    robust: bool = False,
    huber_k: float = 1.345,
    robust_sides: str = 'pre',
    model: str = 'cosh',
    cosh_c2: Optional[float] = None,
    c2_bounds: tuple = (3.0, 8.0),
    moment_floor_abs: Optional[float] = None,
    ramp_gain: Optional[float] = None,
) -> CriticalValueResult:
    """
    Extract critical values using onset detection.

    Pipeline:
      1. Prepare signals (ω, f_col, moment)
      2. Excitation window [|M|>0.01, max|M|]
      3. Onset fit on ω in window (time-quadratic PLS or cosh closed-form)
      4. onset = argmin total residual

    Parameters
    ----------
    model        : 'cosh'      → closed-form unstable tip-over solution
                                 ω(τ) = C₁(cosh(C₂τ)−1) + C (default;
                                 exact solution of the linearised dynamics,
                                 C₂=√d instability rate). Reported method.
                   'piecewise' → time-quadratic onset. This is only the
                                 small-angle (Taylor) limit of the cosh form
                                 and its error grows with tilt; kept as a
                                 comparison baseline / sweep seed, not for
                                 identification.
    omega_source : 'odom' → ω from /mavros/local_position/odom (default)
                   'imu'  → ω from /mavros/imu/data_raw
    lpf_cutoff   : if set, apply a zero-phase Butterworth low-pass filter at
                   this cutoff [Hz] to ω before window detection and the
                   piecewise fit. Recommended for the raw IMU source, which is
                   dominated by propeller vibration (e.g. 15.0). None = off.
    lpf_order    : Butterworth order for the low-pass filter.
    robust       : if True, robustify the piecewise fit so pre-onset vibration
                   outliers are down-weighted (Huber IRLS = Iteratively
                   Reweighted Least Squares).
    huber_k      : Huber threshold in units of the robust noise scale.
    robust_sides : 'pre' (Huber on pre-onset only, recommended) or 'both'.

    The global time reference (t0) stays odom.t[0] regardless of source, so
    onset_time and the downstream mocap pivot estimation remain consistent.
    """
    sig = prepare_signals(bag, axis, C_T=C_T, arm_length=arm_length,
                          omega_source=omega_source, lpf_cutoff=lpf_cutoff,
                          lpf_order=lpf_order)
    t, omega, f_col, moment = (sig['t'], sig['omega'],
                               sig['f_col'], sig['moment'])

    # Excitation window, truncated at the allocator feasibility limit —
    # past it the allocation saturates and the cosh family does not apply
    idx_start, idx_end = detect_excitation_window(
        moment, threshold, moment_cap=MOMENT_CAP.get(axis))
    win = slice(idx_start, idx_end + 1)

    # Onset fit: PLS quadratic or cosh closed-form
    if model == 'cosh':
        # No moment floor by default: the plain closed-form cosh onset, so the
        # fast-ramp behaviour (where the quasi-static assumption breaks down) is
        # reported honestly rather than masked. A floor can still be re-enabled
        # by passing moment_floor_abs explicitly.
        # measured moment ramp rate Ṁ over the excitation window (linear ramp);
        # with the shared gain K it fixes the amplitude C₁ = K·Ṁ (no free param)
        m_dot = float(np.polyfit(t[win], moment[win], 1)[0])
        # No seed in either mode: the whole window is swept (the free
        # fit coarse-to-fine, the constrained fit exhaustively), so the
        # quadratic (small-angle) model is not used anywhere in the
        # cosh path.
        pw = cosh_onset_fit(t[win], omega[win], moment[win], onset_guess=None,
                            c2_bounds=c2_bounds, c2_fixed=cosh_c2,
                            moment_floor=0.0, moment_floor_abs=moment_floor_abs,
                            ramp_gain=ramp_gain, ramp_rate=m_dot)
    else:
        pw = piecewise_onset_fit(t[win], omega[win], robust=robust,
                                 huber_k=huber_k, robust_sides=robust_sides)
    onset_idx = idx_start + pw['onset_idx']

    # Score: use negative residual as "score" (for compatibility)
    # Higher = better fit at this point
    score_values = np.array([-pw['total_residual']])
    score_t = np.array([t[onset_idx]])

    return CriticalValueResult(
        bag_name=bag.name,
        axis=axis,
        t=t,
        omega=omega,
        f_col=f_col,
        moment=moment,
        score_t=score_t,
        score_values=score_values,
        onset_idx=onset_idx,
        onset_time=float(t[onset_idx]),
        onset_score=float(-pw['total_residual']),
        onset_thrust=float(f_col[onset_idx]),
        onset_moment=float(moment[onset_idx]),
        onset_omega=float(omega[onset_idx]),
    ), pw


def extract_piecewise_batch(
    bags: list[BagData],
    axis: str,
    ramp_gate_pct: Optional[float] = 3.0,
    n_full_min: Optional[int] = 38,
    lin_rmse_max: Optional[float] = 0.030,
    **kwargs,
) -> tuple[list[CriticalValueResult], list[dict]]:
    """Run piecewise extraction on every bag.

    Run-level ramp-quality gates, both evaluated on the measured moment trace
    alone (onset-free, hence no circularity), applied BEFORE the rig constants
    are estimated so a poorly executed ramp cannot contaminate the
    dataset-coupled amplitude constraint C₁ = K·Ṁ:

    * ``ramp_gate_pct`` (default 3%): full-window ramp-rate error
      |Ṁ_meas − Ṁ_cmd|/Ṁ_cmd. Conservative by construction — a 3% Ṁ error
      perturbs C₁ by 3%, ~7× below the ±20% level the sensitivity analysis
      shows to be harmless. Runs without a parseable commanded rate are never
      gated on this criterion.
    * ``n_full_min`` (default 38): realized excitation-window sample count
      N_full. A necessary condition independent of where the onset falls: no
      window shorter than 38 samples can furnish the 30 pre-onset (baseline +
      sweep minimum) plus 8 post-onset samples the onset search requires. This
      is the hardware counterpart of the a priori screening bound
      N̂_pre = M_crit^min/(Ṁ·T_s) ≥ 30 — planned vs realized sample support.
    * ``lin_rmse_max`` (default 0.030 N·m): linearity RMSE of M(t) about its
      own best-fit line — a FAULT detector for qualitative ramp failures
      (aborted/stepped/double ramps), not a fine-quality criterion. The
      threshold sits at ~10× the execution noise floor (~3 mN·m, rate
      independent — hence an absolute rather than normalized bound), above the
      maximum observed on healthy runs (22 mN·m) and far below the one
      observed genuine fault (162 mN·m).

    None disables any gate.

    For the cosh model, C₂ is a rig constant: it is estimated once from the
    reliable long-window bags and pinned for every bag (unless the caller
    already passed ``cosh_c2``), so the instability rate stays consistent.
    """
    if (ramp_gate_pct is not None or n_full_min is not None
            or lin_rmse_max is not None):
        sig_kwargs = {k: kwargs[k] for k in ('omega_source', 'lpf_cutoff',
                      'lpf_order', 'C_T', 'arm_length') if k in kwargs}
        kept = []
        for bag in bags:
            sig = prepare_signals(bag, axis, **sig_kwargs)
            i0, i1 = detect_excitation_window(
                sig['moment'], moment_cap=MOMENT_CAP.get(axis))
            n_full = i1 - i0 + 1
            if n_full_min is not None and n_full < n_full_min:
                print(f"  [gate] {bag.name}: N_full={n_full} < {n_full_min} "
                      f"— window too short for onset identification, excluded")
                continue
            if n_full >= 12 and (ramp_gate_pct is not None
                                 or lin_rmse_max is not None):
                win = slice(i0, i1 + 1)
                t_w, m_w = sig['t'][win], sig['moment'][win]
                a, b = np.polyfit(t_w, m_w, 1)
                if lin_rmse_max is not None:
                    lin = float(np.std(m_w - (a * t_w + b)))
                    if lin > lin_rmse_max:
                        print(f"  [gate] {bag.name}: linearity RMSE "
                              f"{lin * 1e3:.1f} mN·m > "
                              f"{lin_rmse_max * 1e3:.0f} — ramp execution "
                              f"fault, excluded")
                        continue
                cmd = commanded_ramp_rate(bag.name)
                if ramp_gate_pct is not None and cmd:
                    floor = SLOPE_GATE_FLOOR.get(axis, 0.0)
                    mask = np.abs(m_w) >= floor
                    a_gate = (np.polyfit(t_w[mask], m_w[mask], 1)[0]
                              if int(mask.sum()) >= 10 else a)
                    eps = (abs(float(a_gate)) - cmd) / cmd * 100.0
                    if abs(eps) > ramp_gate_pct:
                        print(f"  [gate] {bag.name}: ramp-rate error "
                              f"{eps:+.2f}% > {ramp_gate_pct:.0f}% — excluded")
                        continue
            kept.append(bag)
        bags = kept

    if kwargs.get('model', 'cosh') == 'cosh':
        sig_kwargs = {k: kwargs[k] for k in ('omega_source', 'lpf_cutoff',
                      'lpf_order', 'C_T', 'arm_length') if k in kwargs}
        if kwargs.get('cosh_c2') is None or kwargs.get('ramp_gain') is None:
            c2, k_gain = estimate_rig_constants(bags, axis, **sig_kwargs)
            kwargs.setdefault('cosh_c2', None)
            kwargs['cosh_c2'] = kwargs['cosh_c2'] or c2
            kwargs['ramp_gain'] = kwargs.get('ramp_gain') or k_gain
            if kwargs['cosh_c2'] and kwargs['ramp_gain']:
                wz = 1.0 / kwargs['ramp_gain']
                print(f"  Rig constants (closed-form only): "
                      f"C₂={kwargs['cosh_c2']:.3f} rad/s, "
                      f"K={kwargs['ramp_gain']:.3f} "
                      f"→ W·z_CoM={wz:.2f} N·m, "
                      f"J_P={wz / kwargs['cosh_c2'] ** 2:.3f} kg·m²")

    results = []
    pw_fits = []
    for bag in bags:
        print(f"  Processing {bag.name} (axis={axis}, piecewise) ...")
        crit, pw = extract_piecewise(bag, axis, **kwargs)
        print(
            f"    onset t={crit.onset_time:.4f}s  "
            f"f_col={crit.onset_thrust:.3f}N  "
            f"M_{axis}={crit.onset_moment:+.6f}N·m  "
            f"ω_{axis}={crit.onset_omega:.6f}rad/s  "
            f"RMSE={pw['rmse']:.6f}"
        )
        results.append(crit)
        pw_fits.append(pw)
    return results, pw_fits


# ═════════════════════════════════════════════════════════════
#  Ramp-tracking quality  (commanded vs actual moment ramp rate)
# ═════════════════════════════════════════════════════════════

def commanded_ramp_rate(bag_name: str,
                        table: Optional[dict] = None) -> Optional[float]:
    """
    Commanded moment ramp rate [N·m/s] parsed from the bag name.

    The trailing token encodes the set rate in centi-units: '045'→0.45,
    '065'→0.65, '090'→0.90, '120'→1.20. The legacy slow-run tokens
    '01'/'02'/'03' map to 0.10/0.20/0.30 N·m/s. ``table`` overrides both.
    """
    tok = bag_name.split('_')[-1]
    if table is not None and tok in table:
        return float(table[tok])
    if tok in _SLOW_RATE_TOKENS:
        return _SLOW_RATE_TOKENS[tok]
    if tok.isdigit() and len(tok) == 3:
        return int(tok) / 100.0
    return None


_SLOW_RATE_TOKENS = {'01': 0.10, '02': 0.20, '03': 0.30}


def assess_ramp_quality(crit: CriticalValueResult,
                        commanded_rate: float) -> Optional[dict]:
    """
    Post-onset moment-ramp tracking quality.

    Over the post-onset segment [onset, peak|M|] the applied moment is a linear
    ramp. This fits that segment and compares it with the commanded rate:

      * ``actual_rate``    — |slope| of the least-squares line through M(t)
      * ``slope_error_pct``— (actual − commanded)/commanded, in %
      * ``linearity_rmse`` — RMSE of M(t) about its own best-fit line
                             (how straight the applied ramp is)
      * ``tracking_rmse``  — RMSE of M(t) about the *commanded* ramp line
                             M(t_crit)+Ṁ_cmd·τ (how well it tracks the command)

    Low slope error + low RMSE ⇒ the ramp excitation was executed as commanded,
    which supports the constant-Ṁ assumption behind the closed-form onset model
    (C₁ = a·Ṁ/d). Returns None if the segment is too short.
    """
    i0, i1 = detect_excitation_window(crit.moment)
    j = crit.onset_idx
    t = crit.t[j:i1 + 1]
    M = crit.moment[j:i1 + 1]
    if len(t) < 3 or not commanded_rate:
        return None
    sgn = 1.0 if crit.onset_moment >= 0 else -1.0
    tau = t - t[0]
    slope, intercept = np.polyfit(tau, M, 1)
    actual_rate = abs(float(slope))
    lin_rmse = float(np.sqrt(np.mean((M - (slope * tau + intercept)) ** 2)))
    m_cmd = M[0] + sgn * commanded_rate * tau
    track_rmse = float(np.sqrt(np.mean((M - m_cmd) ** 2)))
    return dict(
        commanded_rate=float(commanded_rate),
        actual_rate=actual_rate,
        slope_error_pct=(actual_rate - commanded_rate) / commanded_rate * 100.0,
        linearity_rmse=lin_rmse,
        tracking_rmse=track_rmse,
        n_post=int(len(t)),
    )


# ═════════════════════════════════════════════════════════════
#  Pivot Estimation via Mocap Circle Fit
# ═════════════════════════════════════════════════════════════

def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def fit_circle_cz_fixed(xy: np.ndarray, z: np.ndarray, cz: float = 0.0):
    rhs = xy ** 2 + (z - cz) ** 2
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    beta, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    cx = beta[0]
    R = np.sqrt(max(beta[1] + cx ** 2, 0))
    residuals = np.sqrt((xy - cx) ** 2 + (z - cz) ** 2) - R
    return cx, R, float(np.std(residuals))


def _tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Total tilt angle [deg] from a (N,4) wxyz quaternion array."""
    w, x, y, z = quat.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(np.sqrt(roll ** 2 + pitch ** 2))


def _tilt_departure(t: np.ndarray, tilt: np.ndarray,
                    thresh_deg: float = 1.0, persist: int = 10) -> Optional[float]:
    """Time at which the tilt persistently departs from its initial baseline."""
    n0 = max(10, int(0.1 * len(t)))
    base = float(np.median(tilt[:n0]))
    dev = np.abs(tilt - base) > thresh_deg
    run = 0
    for i in range(len(dev)):
        run = run + 1 if dev[i] else 0
        if run >= persist:
            return float(t[i - persist + 1])
    return None


def align_mocap_time(bag: BagData, desync_threshold_s: float = 100.0) -> np.ndarray:
    """Mocap pose time re-expressed on the odometry clock (relative to t0).

    When the mocap bridge loses network time sync its header stamps carry a
    large CONSTANT clock offset (observed: ~4.4e6 s on the affected runs) while
    remaining monotonic at the correct rate. The offset is recovered by
    matching the tilt DEPARTURE time — the moment the tilt leaves its resting
    baseline, which both streams observe. Departure matching stays robust even
    on runs whose total excursion is small (~3°), where a plain least-squares
    correlation of the tilt traces locks onto the flat baseline instead.

    Runs whose apparent offset is below ``desync_threshold_s`` are returned
    unchanged, so well-synchronised data is never touched.
    """
    t0 = bag.odom.t[0]
    t_mc = bag.pose.t - t0
    t_od = bag.odom.t - t0

    if abs(t_mc[0] - t_od[0]) < desync_threshold_s:
        return t_mc

    tilt_mc = _tilt_deg(bag.pose.quaternion)
    tilt_od = _tilt_deg(bag.odom.quaternion)

    dep_mc = _tilt_departure(t_mc, tilt_mc)
    dep_od = _tilt_departure(t_od, tilt_od)
    if dep_mc is None or dep_od is None:
        # last resort: assume both topics started recording together
        return t_mc - (t_mc[0] - t_od[0])
    delta = dep_mc - dep_od

    # fine refinement: correlate the tilt traces over ±0.3 s around the match
    dt = 0.01
    grid = np.arange(t_od[0], t_od[-1], dt)
    ref = np.interp(grid, t_od, tilt_od)

    def cost(d):
        cand = np.interp(grid, t_mc - d, tilt_mc, left=np.nan, right=np.nan)
        ok = ~np.isnan(cand)
        if ok.sum() < 50:
            return np.inf
        r = cand[ok] - ref[ok]
        return float(r @ r) / ok.sum()

    fine = np.arange(delta - 0.3, delta + 0.3, 0.005)
    delta = float(fine[int(np.argmin([cost(d) for d in fine]))])
    return t_mc - delta


def estimate_pivot_from_mocap(
    bag: BagData,
    onset_time: float,
    axis: str,
    cz: float = 0.0,
    max_window_s: float = 4.0,
    apex_patience: int = 12,
    tilt_cap_deg: float = 10.0,
    motion_min_mm: float = 5.0,
) -> dict:
    t_mc = align_mocap_time(bag)
    px = bag.pose.position[:, 0]
    py = bag.pose.position[:, 1]
    pz = bag.pose.position[:, 2]

    q0 = bag.pose.quaternion[0]
    yaw0 = quat_to_yaw(q0[0], q0[1], q0[2], q0[3])

    idle_mask = t_mc < onset_time * 0.5
    if np.sum(idle_mask) < 10:
        idle_mask = np.arange(len(t_mc)) < 50
    px0 = np.mean(px[idle_mask])
    py0 = np.mean(py[idle_mask])

    c = np.cos(-yaw0); s = np.sin(-yaw0)
    dx_b = (px - px0) * c - (py - py0) * s
    dy_b = (px - px0) * s + (py - py0) * c

    d_horiz = dx_b if axis == 'y' else dy_b

    mc_onset_idx = int(np.searchsorted(t_mc, onset_time))
    if mc_onset_idx >= len(t_mc) - 5:
        return dict(pivot_abs=np.nan, R=np.nan, residual=np.nan,
                    N=0, xy_fit=None, z_fit=None, cx=np.nan)

    # Bound the fit window to the clean pivoting phase. Three guards:
    #   * horizon: at most ``max_window_s`` after the onset;
    #   * tilt cap: stop once the tilt exceeds ``tilt_cap_deg`` — on fast runs
    #     the vehicle tips far past the small-rotation regime (observed up to
    #     ~57°), where gear compliance/slip pulls the marker off the rigid
    #     pivot circle and inflates the residual to ~10 mm;
    #   * apex: a persistent decrease of |d| ends the window (post-excitation
    #     sliding/handling is not on the circle). Decreases only count once the
    #     motion is real (|d| above ``motion_min_mm``), so a noise peak in the
    #     flat pre-motion stretch cannot truncate the window.
    horizon = int(np.searchsorted(t_mc, onset_time + max_window_s))
    tilt = _tilt_deg(bag.pose.quaternion)
    a = np.abs(d_horiz[mc_onset_idx:max(horizon, mc_onset_idx + 6)])
    tw = tilt[mc_onset_idx:max(horizon, mc_onset_idx + 6)]
    motion_min = motion_min_mm * 1e-3
    apex, run, peak = len(a) - 1, 0, -np.inf
    for i in range(len(a)):
        if tw[i] > tilt_cap_deg:
            apex = i
            break
        if a[i] >= peak:
            peak, apex, run = a[i], i, 0
        elif peak > motion_min:
            run += 1
            if run >= apex_patience:
                break
    sl = slice(mc_onset_idx, mc_onset_idx + apex + 1)

    xy_fit = d_horiz[sl] * 1e3
    z_fit = pz[sl] * 1e3

    if len(xy_fit) < 5:
        return dict(pivot_abs=np.nan, R=np.nan, residual=np.nan,
                    N=0, xy_fit=None, z_fit=None, cx=np.nan)

    cx, R, res = fit_circle_cz_fixed(xy_fit, z_fit, cz)

    return dict(pivot_abs=abs(cx), cx=cx, R=R, residual=res,
                N=len(xy_fit), xy_fit=xy_fit, z_fit=z_fit)


# ═════════════════════════════════════════════════════════════
#  Mass & CoM Estimation
# ═════════════════════════════════════════════════════════════

def _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass=None):
    G = 9.81
    if pp is None or pn is None or pp < 1e-6 or pn < 1e-6:
        return np.nan, np.nan, np.nan

    mg = known_mass * G if known_mass else (fn * pn - Mn + fp * pp + Mp) / (pn + pp)
    m = mg / G

    if axis == 'y':
        off_n = pn * (fn / mg - Mn / (mg * pn) - 1)
        off_p = pp * (1 - fp / mg - Mp / (mg * pp))
    else:
        off_n = pn * (1 - fn / mg + Mn / (mg * pn))
        off_p = pp * (fp / mg + Mp / (mg * pp) - 1)

    offset = 0.5 * (off_n + off_p)
    return m, offset, mg * offset


def compute_mass_and_offset(
    critical_results: list[CriticalValueResult],
    pivot_results: list[dict],
    axis: str,
    known_mass: Optional[float] = None,
) -> dict:
    pos_crits = [r for r in critical_results if 'pos' in r.bag_name.lower()]
    neg_crits = [r for r in critical_results if 'neg' in r.bag_name.lower()]
    pos_pivots = [p for r, p in zip(critical_results, pivot_results)
                  if 'pos' in r.bag_name.lower()]
    neg_pivots = [p for r, p in zip(critical_results, pivot_results)
                  if 'neg' in r.bag_name.lower()]

    def _build(pos_list, neg_list, pp_list, pn_list):
        mass, offset, Woff, ff = [], [], [], []
        labels = []
        for i, pc in enumerate(pos_list):
            for j, nc in enumerate(neg_list):
                fp, fn = pc.onset_thrust, nc.onset_thrust
                Mp, Mn = pc.onset_moment, nc.onset_moment
                pp = pp_list[i]['pivot_abs'] * 1e-3 if not np.isnan(pp_list[i]['pivot_abs']) else None
                pn = pn_list[j]['pivot_abs'] * 1e-3 if not np.isnan(pn_list[j]['pivot_abs']) else None
                # W*x_off = -0.5*(Mpy+Mny), W*y_off = +0.5*(Mpx+Mnx)
                sign = -1.0 if axis == 'y' else 1.0
                ff.append(sign * 0.5 * (Mp + Mn))
                m, o, w = _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass)
                mass.append(m); offset.append(o); Woff.append(w)
                labels.append(f"p{i+1}-n{j+1}")
        return mass, offset, Woff, ff, labels

    # 3 same-trial pairs
    n_pairs = min(len(pos_crits), len(neg_crits))
    p3_m, p3_o, p3_w, p3_ff, p3_l = [], [], [], [], []
    for i in range(n_pairs):
        fp, fn = pos_crits[i].onset_thrust, neg_crits[i].onset_thrust
        Mp, Mn = pos_crits[i].onset_moment, neg_crits[i].onset_moment
        pp = pos_pivots[i]['pivot_abs'] * 1e-3 if not np.isnan(pos_pivots[i]['pivot_abs']) else None
        pn = neg_pivots[i]['pivot_abs'] * 1e-3 if not np.isnan(neg_pivots[i]['pivot_abs']) else None
        # W*x_off = -0.5*(Mpy+Mny), W*y_off = +0.5*(Mpx+Mnx)
        sign = -1.0 if axis == 'y' else 1.0
        p3_ff.append(sign * 0.5 * (Mp + Mn))
        m, o, w = _solve_single_pair(fp, fn, Mp, Mn, pp, pn, axis, known_mass)
        p3_m.append(m); p3_o.append(o); p3_w.append(w); p3_l.append(f"p{i+1}-n{i+1}")

    # 9 all combinations
    c9_m, c9_o, c9_w, c9_ff, c9_l = _build(pos_crits, neg_crits, pos_pivots, neg_pivots)

    def _s(arr):
        a = np.array(arr, dtype=float); v = a[~np.isnan(a)]
        if len(v) == 0: return np.nan, 0.0
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    return dict(
        pair3_mass=p3_m, pair3_offset=p3_o, pair3_Woffset=p3_w,
        pair3_ff_onset=p3_ff, pair3_labels=p3_l,
        pair3_mass_mean=_s(p3_m)[0], pair3_mass_std=_s(p3_m)[1],
        pair3_offset_mean=_s(p3_o)[0], pair3_offset_std=_s(p3_o)[1],
        pair3_Woffset_mean=_s(p3_w)[0], pair3_Woffset_std=_s(p3_w)[1],
        pair3_ff_mean=_s(p3_ff)[0], pair3_ff_std=_s(p3_ff)[1],
        comb9_mass=c9_m, comb9_offset=c9_o, comb9_Woffset=c9_w,
        comb9_ff_onset=c9_ff, comb9_labels=c9_l,
        comb9_mass_mean=_s(c9_m)[0], comb9_mass_std=_s(c9_m)[1],
        comb9_offset_mean=_s(c9_o)[0], comb9_offset_std=_s(c9_o)[1],
        comb9_Woffset_mean=_s(c9_w)[0], comb9_Woffset_std=_s(c9_w)[1],
        comb9_ff_mean=_s(c9_ff)[0], comb9_ff_std=_s(c9_ff)[1],
    )


# ═════════════════════════════════════════════════════════════
#  95% Confidence intervals
# ═════════════════════════════════════════════════════════════

def compute_confidence_intervals(
    critical_results, pivot_results, axis,
    known_mass=None, n_boot=10000, seed=0, alpha=0.05,
):
    """
    95% confidence intervals for the identified quantities.

    The N_pos × N_neg combinations are NOT independent samples — they are
    built from N_pos + N_neg measurements (each trial is reused across
    combinations), so treating them as n = N_pos·N_neg overstates the
    precision (pseudo-replication). Two honest interval estimates are given:

      * moment offset M_ff = 0.5(Mp+Mn): analytic propagation from the
        positive/negative critical-moment means, with a Welch–Satterthwaite
        t multiplier — the small-sample-robust, defensible interval.

      * all quantities: bootstrap over TRIALS — the positive and negative
        trials are resampled with replacement (respecting the pos/neg
        structure), the full estimation is recomputed, and the 2.5/97.5
        percentiles are taken. This propagates pseudo-replication correctly,
        but for a very small number of trials the percentile interval tends
        to be optimistic (undercovers); prefer the analytic t interval then.

    Returns a dict with point estimates and (lo, hi) CIs; offsets in metres.
    """
    from scipy import stats  # lazy: optional dependency

    pos = [(r, p) for r, p in zip(critical_results, pivot_results)
           if 'pos' in r.bag_name.lower()]
    neg = [(r, p) for r, p in zip(critical_results, pivot_results)
           if 'neg' in r.bag_name.lower()]
    n_p, n_n = len(pos), len(neg)
    Mp = np.array([r.onset_moment for r, _ in pos])
    Mn = np.array([r.onset_moment for r, _ in neg])
    sign = -1.0 if axis == 'y' else 1.0

    # ── Analytic CI for the feedforward moment offset (linear in Mp, Mn) ──
    ff_mean = sign * 0.5 * (Mp.mean() + Mn.mean())
    sp = Mp.std(ddof=1) if n_p > 1 else 0.0
    sn = Mn.std(ddof=1) if n_n > 1 else 0.0
    var = 0.25 * (sp ** 2 / n_p + sn ** 2 / n_n)
    se = float(np.sqrt(var))
    # Welch–Satterthwaite effective degrees of freedom
    num = (sp ** 2 / n_p + sn ** 2 / n_n) ** 2
    den = 0.0
    if n_p > 1:
        den += (sp ** 2 / n_p) ** 2 / (n_p - 1)
    if n_n > 1:
        den += (sn ** 2 / n_n) ** 2 / (n_n - 1)
    df = num / den if den > 0 else max(n_p + n_n - 2, 1)
    t = float(stats.t.ppf(1 - alpha / 2, df))
    ff_ci_analytic = (ff_mean - t * se, ff_mean + t * se)

    # ── Bootstrap over trials (resample pos & neg independently) ──
    rng = np.random.default_rng(seed)
    pc, pp = [r for r, _ in pos], [p for _, p in pos]
    nc, npv = [r for r, _ in neg], [p for _, p in neg]
    keys = ['comb9_ff_mean', 'comb9_offset_mean',
            'comb9_Woffset_mean', 'comb9_mass_mean']
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        pi = rng.integers(0, n_p, n_p)
        ni = rng.integers(0, n_n, n_n)
        crits = [pc[i] for i in pi] + [nc[i] for i in ni]
        pivs = [pp[i] for i in pi] + [npv[i] for i in ni]
        e = compute_mass_and_offset(crits, pivs, axis, known_mass)
        for k in keys:
            boot[k].append(e[k])

    def _pct(vals):
        a = np.array(vals, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(a, 100 * alpha / 2)),
                float(np.percentile(a, 100 * (1 - alpha / 2))))

    return dict(
        n_pos=n_p, n_neg=n_n, n_boot=n_boot, df=df,
        ff_mean=ff_mean, ff_se=se,
        ff_ci_analytic=ff_ci_analytic,
        ff_ci_boot=_pct(boot['comb9_ff_mean']),
        offset_ci_boot=_pct(boot['comb9_offset_mean']),   # metres
        Woffset_ci_boot=_pct(boot['comb9_Woffset_mean']),
        mass_ci_boot=_pct(boot['comb9_mass_mean']),
    )


# ═════════════════════════════════════════════════════════════
#  CSV Export
# ═════════════════════════════════════════════════════════════

def save_estimation_csv(
    critical_results, pivot_results, estimation, axis, output_dir,
    known_mass=None,
) -> Path:
    import csv
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    offset_label = 'x_off' if axis == 'y' else 'y_off'

    # 1. Summary
    p = output_dir / f"com_estimation_summary_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag_name','direction','f_crit_N','M_crit_Nm',
                     'pivot_mm','pivot_R_mm','pivot_rmse_mm','pivot_N_pts'])
        for crit, piv in zip(critical_results, pivot_results):
            d = 'pos' if 'pos' in crit.bag_name.lower() else 'neg'
            w.writerow([crit.bag_name, d,
                f"{crit.onset_thrust:.6f}", f"{crit.onset_moment:.8f}",
                f"{piv['pivot_abs']:.2f}" if not np.isnan(piv['pivot_abs']) else '',
                f"{piv['R']:.2f}" if not np.isnan(piv['R']) else '',
                f"{piv['residual']:.4f}" if not np.isnan(piv['residual']) else '',
                piv['N']])
    print(f"  Summary     → {p}")

    # 2. 3 pairs
    p2 = output_dir / f"com_estimation_pairs_{axis}.csv"
    with open(p2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair','mass_kg',f'{offset_label}_mm','W_offset_Nm','ff_onset_Nm'])
        for i, lb in enumerate(estimation['pair3_labels']):
            m,o,wo,ff = estimation['pair3_mass'][i], estimation['pair3_offset'][i], \
                        estimation['pair3_Woffset'][i], estimation['pair3_ff_onset'][i]
            w.writerow([lb, f"{m:.6f}" if not np.isnan(m) else '',
                f"{o*1e3:.4f}" if not np.isnan(o) else '',
                f"{wo:.8f}" if not np.isnan(wo) else '', f"{ff:.8f}"])
    print(f"  3 pairs     → {p2}")

    # 3. 9 combs
    p3 = output_dir / f"com_estimation_combs_{axis}.csv"
    with open(p3, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['combination','mass_kg',f'{offset_label}_mm','W_offset_Nm','ff_onset_Nm'])
        for i, lb in enumerate(estimation['comb9_labels']):
            m,o,wo,ff = estimation['comb9_mass'][i], estimation['comb9_offset'][i], \
                        estimation['comb9_Woffset'][i], estimation['comb9_ff_onset'][i]
            w.writerow([lb, f"{m:.6f}" if not np.isnan(m) else '',
                f"{o*1e3:.4f}" if not np.isnan(o) else '',
                f"{wo:.8f}" if not np.isnan(wo) else '', f"{ff:.8f}"])
    print(f"  9 combs     → {p3}")

    # 4. Aggregated
    p4 = output_dir / f"com_estimation_result_{axis}.csv"
    pp = [pv['pivot_abs'] for r, pv in zip(critical_results, pivot_results)
          if 'pos' in r.bag_name.lower() and not np.isnan(pv['pivot_abs'])]
    pn = [pv['pivot_abs'] for r, pv in zip(critical_results, pivot_results)
          if 'neg' in r.bag_name.lower() and not np.isnan(pv['pivot_abs'])]
    with open(p4, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter','value','std','unit','note'])
        if pp: w.writerow(['pivot_pos', f"{np.mean(pp):.2f}",
            f"{np.std(pp,ddof=1):.2f}" if len(pp)>1 else '0.00', 'mm', f'N={len(pp)}'])
        if pn: w.writerow(['pivot_neg', f"{np.mean(pn):.2f}",
            f"{np.std(pn,ddof=1):.2f}" if len(pn)>1 else '0.00', 'mm', f'N={len(pn)}'])
        for prefix, tag in [('pair3','3pair'), ('comb9','9comb')]:
            e = estimation
            w.writerow([f'mass_{tag}', f"{e[f'{prefix}_mass_mean']:.6f}",
                f"{e[f'{prefix}_mass_std']:.6f}", 'kg',
                f"known={known_mass}" if known_mass else 'estimated'])
            w.writerow([f'{offset_label}_{tag}', f"{e[f'{prefix}_offset_mean']*1e3:.4f}",
                f"{e[f'{prefix}_offset_std']*1e3:.4f}", 'mm', tag])
            w.writerow([f'W_offset_{tag}', f"{e[f'{prefix}_Woffset_mean']:.8f}",
                f"{e[f'{prefix}_Woffset_std']:.8f}", 'Nm', tag])
            w.writerow([f'ff_onset_{tag}', f"{e[f'{prefix}_ff_mean']:.8f}",
                f"{e[f'{prefix}_ff_std']:.8f}", 'Nm', f'0.5*(Mp+Mn) {tag}'])
    print(f"  Result      → {p4}")
    return p4


def save_piecewise_rmse_csv(
    critical_results: list,
    pw_fits: list[dict],
    axis: str,
    output_dir: Path,
) -> Path:
    """Save piecewise fit RMSE values to CSV."""
    import csv
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"piecewise_rmse_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag_name', 'onset_time_s', 'M_crit_Nm', 'alpha', 'c', 'rmse_rad_s'])
        for crit, pw in zip(critical_results, pw_fits):
            w.writerow([
                crit.bag_name,
                f"{crit.onset_time:.6f}",
                f"{crit.onset_moment:.8f}",
                f"{pw['alpha']:.8f}",
                f"{pw['c']:.8f}",
                f"{pw['rmse']:.8f}",
            ])
    print(f"  PW RMSE     → {p}")
    return p


def save_pivot_csv(
    critical_results: list,
    pivot_results: list[dict],
    axis: str,
    output_dir: Path,
) -> Path:
    """Save pivot circle-fit parameters (cx, R, residual) to CSV."""
    import csv
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"pivot_params_{axis}.csv"
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bag_name', 'direction', 'cx_mm', 'pivot_abs_mm',
                    'R_mm', 'res_mm', 'N_pts'])
        for crit, piv in zip(critical_results, pivot_results):
            d = 'pos' if 'pos' in crit.bag_name.lower() else 'neg'
            w.writerow([
                crit.bag_name, d,
                f"{piv['cx']:.4f}" if not np.isnan(piv['cx']) else '',
                f"{piv['pivot_abs']:.4f}" if not np.isnan(piv['pivot_abs']) else '',
                f"{piv['R']:.4f}" if not np.isnan(piv['R']) else '',
                f"{piv['residual']:.6f}" if not np.isnan(piv['residual']) else '',
                piv['N'],
            ])
    print(f"  Pivot params → {p}")
    return p


# ═════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════

def plot_piecewise_fits(
    bags, critical_results, pw_fits, axis,
    save_dir=None, show=True,
):
    """Plot ω actual vs onset-model fit for all bags."""
    n = len(bags)
    cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows), squeeze=False)
    _models = set(pw.get('model', 'piecewise') for pw in pw_fits)
    if 'cosh' in _models:
        _model_name, _sub = 'Hyperbolic (cosh)', 'fit'
    else:
        _model_name, _sub = 'Piecewise', 'pred'
    omega_label = r'$\omega_{x,act}$' if axis == 'x' else r'$\omega_{y,act}$'
    omega_pred_label = (r'$\omega_{x,' + _sub + r'}$' if axis == 'x'
                        else r'$\omega_{y,' + _sub + r'}$')

    for idx, (bag, crit, pw) in enumerate(zip(bags, critical_results, pw_fits)):
        r, c = divmod(idx, cols); ax = axes[r][c]
        t = crit.t
        # Window
        moment = crit.moment
        idx_start, idx_end = detect_excitation_window(moment)
        win = slice(idx_start, idx_end + 1)
        t_w = t[win]; omega_w = crit.omega[win]

        # Prediction in window. Use the stored prediction when it is
        # window-aligned (covers the cosh model, whose ω_pred is not a
        # time-quadratic); otherwise rebuild the quadratic from (c, α).
        t0 = t_w[pw['onset_idx']]
        stored = pw.get('omega_pred')
        if stored is not None and len(stored) == len(omega_w):
            pred = np.asarray(stored)
        else:
            cc, alpha = pw['c'], pw['alpha']
            pred = np.full_like(omega_w, cc)
            after = t_w >= t0
            dt = t_w[after] - t0
            pred[after] = alpha * dt**2 + cc

        # Onset moment label based on bag name
        _name = bag.name.lower()
        if 'pos' in _name:
            _msign = '+'
        elif 'neg' in _name:
            _msign = '-'
        else:
            _msign = ''
        _maxis = 'x' if axis == 'x' else 'y'
        _onset_label = r'$M_{' + _maxis + ',' + _msign + r'}$' + f' = {crit.onset_moment:+.4f} Nm'

        ax.plot(t_w, omega_w, 'k-', lw=0.8, alpha=0.8, label=f'{omega_label}')
        ax.plot(t_w, pred, 'b-', lw=2, alpha=0.5, label=omega_pred_label)
        ax.axvline(t0, color='red', ls='--', lw=1, alpha=0.7)
        ax.plot(t0, crit.onset_omega, 'r.', ms=4, zorder=5)

        # Residual band. In robust mode the fit minimises the Huber-processed
        # residual, so show that instead of the raw one: pre-onset residuals
        # are clipped to ±δ (Huber influence ψ), the rise is kept as-is.
        huber_delta = pw.get('huber_delta')
        resid = omega_w - pred
        if huber_delta is not None:
            resid_h = resid.copy()
            pre = np.arange(len(resid)) < pw['onset_idx']
            resid_h[pre] = np.clip(resid[pre], -huber_delta, huber_delta)
            ax.fill_between(t_w, pred, pred + resid_h, alpha=0.12, color='red',
                            label=r'Huber residual ($|r|_{pre}\leq\delta$)')
            outl = pre & (np.abs(resid) > huber_delta)
            if np.any(outl):
                ax.plot(t_w[outl], omega_w[outl], 'x', color='darkorange',
                        ms=5, mew=1.2, zorder=6, label='down-weighted outlier')
        else:
            ax.fill_between(t_w, omega_w, pred, alpha=0.1, color='red')

        ax2 = ax.twinx()
        ax2.plot(t_w, moment[win], 'tab:green', lw=1.5, alpha=0.7)
        ax2.plot(t0, crit.onset_moment, 'r.', ms=6, zorder=5, label=_onset_label)
        _M_label = r'$M_x$' if axis == 'x' else r'$M_y$'
        ax2.set_ylabel(_M_label + ' [N·m]', color='tab:green', fontsize=12)

        ax.set_ylabel(f'{omega_label} [rad/s]', fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_title(bag_name_to_title(bag.name), fontsize=13)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        leg_loc = 'lower left' if 'neg' in bag.name.lower() else 'upper left'
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc=leg_loc)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols); axes[r][c].set_visible(False)

    _ax = 'x' if axis == 'x' else 'y'
    fig.suptitle(_model_name + r' Onset Fit: $\omega_{' + _ax + r',act}$ vs $\omega_{'
                 + _ax + ',' + _sub + r'}$', fontsize=14)
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / f"piecewise_fit_{axis}.png", dpi=300, bbox_inches='tight')
        print(f"  PW fit plot → {Path(save_dir) / f'piecewise_fit_{axis}.png'}")
    if show: plt.show()
    else: plt.close(fig)


def plot_pivot_fits(bags, critical_results, pivot_results, axis, save_dir=None, show=True):
    n = len(bags); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False)
    hl = r'$\Delta x_{body}$' if axis == 'y' else r'$\Delta y_{body}$'

    for idx, (bag, crit, piv) in enumerate(zip(bags, critical_results, pivot_results)):
        r, c = divmod(idx, cols); ax = axes[r][c]
        if piv['xy_fit'] is None:
            ax.set_title(bag_name_to_title(bag.name) + '\nInsufficient data', fontsize=13)
            continue
        xy, z, cx, R, res = piv['xy_fit'], piv['z_fit'], piv['cx'], piv['R'], piv['residual']
        n_pts = len(xy)

        # Data points (color gradient = time)
        colors = plt.cm.viridis(np.linspace(0, 1, n_pts))
        ax.scatter(xy, z, c=colors, s=10, zorder=3, label='data')

        # Identified circle — ARC only (angular span of data + small margin)
        ang = np.arctan2(z - 0.0, xy - cx)
        a_min, a_max = np.min(ang), np.max(ang)
        margin = 0.1 * (a_max - a_min)
        theta = np.linspace(a_min - margin, a_max + margin, 200)
        ax.plot(cx + R*np.cos(theta), R*np.sin(theta), 'r-', lw=1.5, alpha=0.6,
                label='identified circle')

        # Rotation direction arrows ALONG the identified arc (follow time order)
        ang_start, ang_end = ang[0], ang[-1]
        arc_ang = np.linspace(ang_start, ang_end, min(6, n_pts))
        for k in range(len(arc_ang) - 1):
            x0 = cx + R*np.cos(arc_ang[k]);   z0 = R*np.sin(arc_ang[k])
            x1 = cx + R*np.cos(arc_ang[k+1]); z1 = R*np.sin(arc_ang[k+1])
            if np.sqrt((x1-x0)**2 + (z1-z0)**2) < 0.01:
                continue
            ax.annotate('', xy=(x1, z1), xytext=(x0, z0),
                arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2, mutation_scale=15), zorder=4)

        ax.set_aspect('equal'); ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(bag_name_to_title(bag.name), fontsize=13)
        ax.set_xlabel(f'{hl} [mm]', fontsize=12)
        ax.set_ylabel('z [mm]', fontsize=12)

    for idx in range(n, rows*cols): r, c = divmod(idx, cols); axes[r][c].set_visible(False)

    # Single shared legend outside the figure (all subplots share same labels)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc='upper right',
               bbox_to_anchor=(1.0, 0.98), framealpha=0.9)

    fig.suptitle('Mocap Circle Fit (cz=0, onset→max|d|)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    if save_dir:
        fig.savefig(Path(save_dir) / f"pivot_circle_fit_{axis}.png", dpi=300, bbox_inches='tight')
        print(f"  Pivot plot  → {Path(save_dir) / f'pivot_circle_fit_{axis}.png'}")
    if show: plt.show()
    else: plt.close(fig)


def plot_estimation_results(estimation, axis, save_dir=None, show=True):
    offset_label = '$x_{off}$' if axis == 'y' else '$y_{off}$'
    moment_label = '$M_y^{off}$' if axis == 'y' else '$M_x^{off}$'
    offset_unit = 'x_off [mm]' if axis == 'y' else 'y_off [mm]'
    _Woff_tex = r'$W \cdot x_{off}$' if axis == 'y' else r'$W \cdot y_{off}$'
    moment_unit = _Woff_tex + ' [N·m]'

    # Scatter
    fig1, ax12 = plt.subplots(1, 2, figsize=(14, 6))
    for i, (prefix, color, title) in enumerate([
        ('pair3', 'tab:blue', '3 Same-Trial Pairs'),
        ('comb9', 'tab:green', '9 All Combinations'),
    ]):
        ax = ax12[i]
        off = np.array(estimation[f'{prefix}_offset'])*1e3
        Woff = np.array(estimation[f'{prefix}_Woffset'])
        labels = estimation[f'{prefix}_labels']
        valid = ~np.isnan(off) & ~np.isnan(Woff)
        if np.any(valid):
            ax.scatter(off[valid], Woff[valid], c=color, s=80, zorder=3, edgecolors='k')
            for j in np.where(valid)[0]:
                ax.annotate(labels[j], (off[j], Woff[j]), textcoords='offset points',
                            xytext=(6, 4), fontsize=8)
            ax.axvline(estimation[f'{prefix}_offset_mean']*1e3, color='red', ls='--', alpha=0.5)
            ax.axhline(estimation[f'{prefix}_Woffset_mean'], color='red', ls='--', alpha=0.5)
        ax.set_xlabel(f'{offset_label} [mm]'); ax.set_ylabel(f'{moment_label} [N·m]')
        ax.set_title(title); ax.grid(True, alpha=0.3)

    fig1.suptitle(f'Offset vs Feedforward (axis={axis})', fontsize=13)
    fig1.tight_layout()
    if save_dir:
        fig1.savefig(Path(save_dir)/f"estimation_scatter_{axis}.png", dpi=600, bbox_inches='tight')

    # Box plot summary
    fig3, ax3 = plt.subplots(1, 4, figsize=(18, 6))

    # Dynamic labels based on axis
    off_name = 'x_{off}' if axis == 'y' else 'y_{off}'
    Woff_label_pivot = r'$W \cdot ' + off_name + r'$ (Pivot-Based)'
    Woff_label_free  = r'$W \cdot ' + off_name + r'$ (Pivot-Free)'

    configs = [
        ('Estimated Mass', 'Mass [kg]',
         estimation['pair3_mass'], estimation['comb9_mass']),
        (f'CoM Offset ({offset_label})', offset_unit,
         [o*1e3 for o in estimation['pair3_offset']], [o*1e3 for o in estimation['comb9_offset']]),
        (Woff_label_pivot, moment_unit,
         estimation['pair3_Woffset'], estimation['comb9_Woffset']),
        (Woff_label_free, moment_unit,
         estimation['pair3_ff_onset'], estimation['comb9_ff_onset']),
    ]
    colors_g = ['tab:blue', 'tab:green']
    for col, (title, ylabel, v3, v9) in enumerate(configs):
        ax = ax3[col]
        d3 = [v for v in v3 if not np.isnan(v)]
        d9 = [v for v in v9 if not np.isnan(v)]
        data = []; lbls = []; cols_b = []
        if d3: data.append(d3); lbls.append(f'3 pairs\n(N={len(d3)})'); cols_b.append(colors_g[0])
        if d9: data.append(d9); lbls.append(f'9 combs\n(N={len(d9)})'); cols_b.append(colors_g[1])
        if not data: continue
        bp = ax.boxplot(data, labels=lbls, patch_artist=True, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='k', markersize=8),
            medianprops=dict(color='orange', linewidth=2), widths=0.5)
        for patch, clr in zip(bp['boxes'], cols_b): patch.set_facecolor(clr); patch.set_alpha(0.4)
        for j, dd in enumerate(data):
            a = np.array(dd); m = np.mean(a); s = np.std(a, ddof=1) if len(a)>1 else 0
            ax.text(j+1.3, m, f'μ={m:+.4f}\nσ={s:.4f}', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')

    fig3.suptitle(f'CoM Estimation: Pivot-Based vs Pivot-Free (axis={axis}) — ◆mean, ━median', fontsize=13)
    fig3.tight_layout()
    if save_dir:
        fig3.savefig(Path(save_dir)/f"estimation_summary_{axis}.png", dpi=600, bbox_inches='tight')

    if show: plt.show()
    else: plt.close(fig1); plt.close(fig3)


# ═════════════════════════════════════════════════════════════
#  Axis Detection
# ═════════════════════════════════════════════════════════════

def detect_axis(data_dir: Path, bags: list[BagData]) -> str:
    for src in [data_dir.name.lower()] + [b.name.lower() for b in bags]:
        if 'mx' in src: return 'x'
        if 'my' in src: return 'y'
    raise ValueError(f"Cannot detect axis from '{data_dir.name}'. Use --axis.")


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Moment Excitation — Piecewise Onset + Pivot")
    p.add_argument('data_dir', type=str)
    p.add_argument('--axis', type=str, default=None, choices=['x','y'])
    p.add_argument(
        '--omega-source', type=str, default='odom', choices=['odom', 'imu'],
        help="Angular velocity source for onset detection: "
             "'odom' (/mavros/local_position/odom, default) or "
             "'imu' (/mavros/imu/data_raw).",
    )
    p.add_argument(
        '--lpf-cutoff', type=float, default=None,
        help="Butterworth low-pass cutoff [Hz] applied to ω before onset "
             "detection (e.g. 15). Recommended with --omega-source imu to "
             "suppress propeller vibration. Off if omitted.",
    )
    p.add_argument(
        '--lpf-order', type=int, default=4,
        help="Butterworth order for --lpf-cutoff (default 4).",
    )
    p.add_argument(
        '--model', type=str, default='cosh',
        choices=['cosh', 'piecewise'],
        help="Onset model: 'cosh' (closed-form unstable tip-over "
             "ω=C1(cosh(C2τ)-1)+C, default/reported) or 'piecewise' "
             "(time-quadratic — small-angle limit only, comparison baseline).",
    )
    p.add_argument(
        '--robust', action='store_true',
        help="Robustify the piecewise onset fit (Huber IRLS = Iteratively "
             "Reweighted Least Squares) to down-weight pre-onset vibration "
             "outliers (default: ordinary least squares).",
    )
    p.add_argument(
        '--robust-sides', type=str, default='pre', choices=['pre', 'both'],
        help="Where to apply Huber: 'pre' (pre-onset flat segment only, "
             "keeps the rise as plain LS; recommended) or 'both'.",
    )
    p.add_argument(
        '--huber-k', type=float, default=1.345,
        help="Huber threshold in robust noise-scale units (default 1.345, "
             "= 95%% Gaussian efficiency). Only used with --robust.",
    )
    p.add_argument('--mass', type=float, default=None)
    p.add_argument(
        '--ci', action='store_true',
        help="Report 95%% confidence intervals (analytic propagation for the "
             "moment offset + bootstrap over trials for CoM offset / mass).",
    )
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--save-fig', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir

    # 1. Load
    bags = load_excitation_dataset(dataset_dir)
    print(f"Loaded {len(bags)} bags: {[b.name for b in bags]}\n")

    # 2. Axis
    axis = args.axis if args.axis else detect_axis(dataset_dir, bags)
    offset_label = 'y_off' if axis == 'x' else 'x_off'
    omega_topic = ('/mavros/imu/data_raw' if args.omega_source == 'imu'
                   else '/mavros/local_position/odom')
    print(f'Axis        : {axis} ({"roll" if axis=="x" else "pitch"})')
    _mdl = {'cosh': 'Closed-form cosh: ω=C1(cosh(C2τ)-1)+C',
            'piecewise': 'Piecewise quadratic fit'}[args.model]
    print(f'Detection   : {_mdl}')
    print(f'ω source    : {args.omega_source} ({omega_topic})')
    if args.lpf_cutoff is not None:
        print(f'LPF         : Butterworth {args.lpf_cutoff:g} Hz (order {args.lpf_order})')
    if args.robust:
        print(f'Fit         : robust Huber IRLS [Iteratively Reweighted Least Squares] '
              f'(k={args.huber_k:g}, sides={args.robust_sides})')
    if args.mass: print(f'Known mass  : {args.mass} kg')
    print()

    # 3. Piecewise onset
    print("── Piecewise Onset Detection ──")
    critical_results, pw_fits = extract_piecewise_batch(
        bags, axis=axis, omega_source=args.omega_source,
        lpf_cutoff=args.lpf_cutoff, lpf_order=args.lpf_order,
        robust=args.robust, huber_k=args.huber_k, robust_sides=args.robust_sides,
        model=args.model,
    )

    # 4. CSV (critical values)
    print("\n── Critical Value CSV ──")
    ext = CriticalValueExtractor()
    ext.save_batch_csv(critical_results, output_dir=output_dir)

    # 5. Pivot
    print("\n── Pivot Estimation (Mocap Circle Fit, cz=0) ──")
    pivot_results = []
    for bag, crit in zip(bags, critical_results):
        piv = estimate_pivot_from_mocap(bag, crit.onset_time, axis=axis)
        pivot_results.append(piv)
        s = f"|cx|={piv['pivot_abs']:.1f}mm R={piv['R']:.1f}mm res={piv['residual']:.2f}mm N={piv['N']}" \
            if not np.isnan(piv['pivot_abs']) else "FAILED"
        print(f"  {bag.name}: {s}")

    # 6. Mass & CoM
    print("\n── Mass & CoM Offset ──")
    est = compute_mass_and_offset(critical_results, pivot_results, axis=axis, known_mass=args.mass)

    # 6a. 95% confidence intervals
    if args.ci:
        print("\n── 95% Confidence Intervals ──")
        ci = compute_confidence_intervals(
            critical_results, pivot_results, axis=axis, known_mass=args.mass)
        off_lbl = 'x_off' if axis == 'y' else 'y_off'
        print(f"  [n = {ci['n_pos']} pos + {ci['n_neg']} neg trials — "
              f"the {ci['n_pos']*ci['n_neg']} combinations are NOT independent]")
        print(f"  Moment offset M_ff = {ci['ff_mean']:+.5f} N·m")
        print(f"    analytic (Welch t, df={ci['df']:.1f}) : "
              f"[{ci['ff_ci_analytic'][0]:+.5f}, {ci['ff_ci_analytic'][1]:+.5f}]  (defensible)")
        print(f"    bootstrap (B={ci['n_boot']})            : "
              f"[{ci['ff_ci_boot'][0]:+.5f}, {ci['ff_ci_boot'][1]:+.5f}]")
        o = ci['offset_ci_boot']
        print(f"  CoM {off_lbl}  bootstrap 95% CI : "
              f"[{o[0]*1e3:+.2f}, {o[1]*1e3:+.2f}] mm")
        w = ci['Woffset_ci_boot']
        print(f"  W·offset   bootstrap 95% CI : [{w[0]:+.5f}, {w[1]:+.5f}] N·m")
        m = ci['mass_ci_boot']
        print(f"  mass       bootstrap 95% CI : [{m[0]:.3f}, {m[1]:.3f}] kg")
        print("  (small n: the analytic t interval is conservative; bootstrap "
              "may undercover — report the analytic one.)")

    # 6b. CSV
    print("\n── Estimation CSV ──")
    save_estimation_csv(critical_results, pivot_results, est, axis=axis,
                        output_dir=output_dir, known_mass=args.mass)
    save_piecewise_rmse_csv(critical_results, pw_fits, axis=axis,
                            output_dir=output_dir)
    save_pivot_csv(critical_results, pivot_results, axis=axis,
                   output_dir=output_dir)

    # 7. Summary
    print(f"\n{'='*75}")
    print(f"  Summary ({dataset_dir.name}, axis={axis}, Piecewise onset)")
    print(f"{'='*75}")

    print(f"\n  ── Critical Values ──")
    print(f"  {'Bag':<25} {'f_col[N]':>10} {'M[N·m]':>12} {'ω[rad/s]':>12}")
    print("  "+"-"*60)
    for r in critical_results:
        print(f"  {r.bag_name:<25} {r.onset_thrust:>10.4f} {r.onset_moment:>+12.6f} {r.onset_omega:>12.6f}")

    print(f"\n  ── Pivot ──")
    for r, p in zip(critical_results, pivot_results):
        pv = f"{p['pivot_abs']:.1f}" if not np.isnan(p['pivot_abs']) else "N/A"
        print(f"  {r.bag_name:<25} {pv:>10} mm")

    pp = [p['pivot_abs'] for r, p in zip(critical_results, pivot_results) if 'pos' in r.bag_name.lower() and not np.isnan(p['pivot_abs'])]
    pn = [p['pivot_abs'] for r, p in zip(critical_results, pivot_results) if 'neg' in r.bag_name.lower() and not np.isnan(p['pivot_abs'])]
    if pp: print(f"\n  pp avg: {np.mean(pp):.1f} ± {np.std(pp,ddof=1):.1f} mm")
    if pn: print(f"  pn avg: {np.mean(pn):.1f} ± {np.std(pn,ddof=1):.1f} mm")

    for tag, prefix in [("3 pairs", "pair3"), ("9 combs", "comb9")]:
        print(f"\n  ── {tag} ──")
        for i, lb in enumerate(est[f'{prefix}_labels']):
            m,o,w = est[f'{prefix}_mass'][i], est[f'{prefix}_offset'][i], est[f'{prefix}_Woffset'][i]
            ms = f"{m:.4f}" if not np.isnan(m) else "N/A"
            os = f"{o*1e3:+.3f}" if not np.isnan(o) else "N/A"
            ws = f"{w:+.6f}" if not np.isnan(w) else "N/A"
            print(f"    {lb}: m={ms}kg {offset_label}={os}mm W·off={ws}Nm")
        print(f"    mean: m={est[f'{prefix}_mass_mean']:.4f}±{est[f'{prefix}_mass_std']:.4f}  "
              f"{offset_label}={est[f'{prefix}_offset_mean']*1e3:+.3f}±{est[f'{prefix}_offset_std']*1e3:.3f}mm  "
              f"W·off={est[f'{prefix}_Woffset_mean']:+.6f}±{est[f'{prefix}_Woffset_std']:.6f}")

    print(f"\n  ── Feedforward ──")
    print(f"  0.5*(Mp+Mn) 3p: {est['pair3_ff_mean']:+.6f} ± {est['pair3_ff_std']:.6f} N·m")
    print(f"  0.5*(Mp+Mn) 9c: {est['comb9_ff_mean']:+.6f} ± {est['comb9_ff_std']:.6f} N·m")
    print(f"  W·offset    3p: {est['pair3_Woffset_mean']:+.6f} ± {est['pair3_Woffset_std']:.6f} N·m")
    print(f"  W·offset    9c: {est['comb9_Woffset_mean']:+.6f} ± {est['comb9_Woffset_std']:.6f} N·m")
    print(f"\n{'='*75}")

    # 8. Plots
    save_dir = output_dir if args.save_fig else None
    show = not args.no_plot
    if show or save_dir:
        plot_piecewise_fits(bags, critical_results, pw_fits, axis, save_dir, show)
        plot_pivot_fits(bags, critical_results, pivot_results, axis, save_dir, show)
        plot_estimation_results(est, axis, save_dir, show)


if __name__ == "__main__":
    main()