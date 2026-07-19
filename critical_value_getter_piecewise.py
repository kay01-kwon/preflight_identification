#!/usr/bin/env python3
"""
Moment Excitation Analysis — Piecewise Onset Detection + Pivot Estimation

Onset detection via piecewise quadratic fit on ω:
    ω(t) = c                            for t < t0  (ground contact)
    ω(t) = a·(t-t0)² + b·(t-t0) + c    for t ≥ t0  (tip-over)

t0* = argmin Σ residuals²  (sweep all candidate t0 in excitation window)

Usage
-----
python critical_value_getter_piecewise.py DataSet/exp/Mx
python critical_value_getter_piecewise.py DataSet/exp/My --mass 3.066 --save-fig

# Use raw IMU angular velocity (/mavros/imu/data_raw) instead of odom:
python critical_value_getter_piecewise.py DataSet/exp/My --omega-source imu

# Raw IMU + 15 Hz low-pass filter to suppress propeller vibration:
python critical_value_getter_piecewise.py DataSet/exp/My --omega-source imu --lpf-cutoff 15

# Robust (Huber) onset fit to reject pre-onset spikes/outliers:
python critical_value_getter_piecewise.py DataSet/exp/My --omega-source imu --lpf-cutoff 15 --robust

# Closed-form unstable tip-over model ω=C1(cosh(C2τ)-1)+C on the odom rate:
python critical_value_getter_piecewise.py DataSet/exp/My --model cosh
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




def cosh_onset_fit(t, omega, moment, onset_guess,
                   sweep_back_s=0.4, sweep_ahead_s=0.3, step_s=0.02):
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

    The onset t_crit is swept (tight around onset_guess); the critical moment
    is M at the best onset. Fit is 3-parameter (C₁, C₂, C), well conditioned.

    Returns the same core keys as piecewise_onset_fit; 'alpha' carries C₂
    (the instability rate √d) and 'c' the baseline for CSV compatibility.
    """
    from scipy.optimize import least_squares  # lazy: optional dependency

    N = len(t)
    dt = float(np.median(np.diff(t)))
    lo = max(1, onset_guess - int(round(sweep_back_s / dt)))
    hi = min(N - 20, onset_guess + int(round(sweep_ahead_s / dt)))
    step = max(1, int(round(step_s / dt)))

    # tip-over direction (sign of ω at the window tail vs baseline) to init C₁
    base0 = float(np.median(omega[:max(1, onset_guess)]))
    sgn = 1.0 if float(np.mean(omega[int(0.85 * N):])) >= base0 else -1.0

    def model(p, tau):
        C1, C2, C = p
        return C1 * (np.cosh(np.clip(C2 * tau, 0, 30)) - 1) + C

    best = (np.inf, onset_guess, np.array([sgn * 1e-3, 3.0, base0]))
    for j in range(lo, max(lo + 1, hi), step):
        tau = t[j:] - t[j]
        y = omega[j:]
        C0 = float(np.median(omega[:j])) if j > 0 else 0.0
        r = least_squares(lambda p: model(p, tau) - y,
                          [sgn * 1e-3, 3.0, C0], method='trf',
                          bounds=([-5.0, 0.05, -2.0], [5.0, 30.0, 2.0]),
                          max_nfev=300)
        pre = np.sum((omega[:j] - C0) ** 2) if j > 0 else 0.0
        cost = float(np.sum(r.fun ** 2) + pre)
        if cost < best[0]:
            best = (cost, j, r.x)

    cost, j_star, params = best
    C1, C2, C = params
    omega_pred = np.full(N, float(C))
    omega_pred[j_star:] = model(params, t[j_star:] - t[j_star])
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


def detect_excitation_window(
    moment: np.ndarray,
    threshold: float = 0.01,
) -> tuple[int, int]:
    """
    Find excitation window: [first |M|>threshold, max|M|].
    """
    idx_end = int(np.argmax(np.abs(moment)))
    above = np.where(np.abs(moment) > threshold)[0]
    if len(above) > 0 and above[0] < idx_end:
        return int(above[0]), idx_end
    return 0, idx_end


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
    model: str = 'piecewise',
) -> CriticalValueResult:
    """
    Extract critical values using piecewise onset detection.

    Pipeline:
      1. Prepare signals (ω, f_col, moment)
      2. Excitation window [|M|>0.01, max|M|]
      3. Onset fit on ω in window (time-quadratic PLS or cosh closed-form)
      4. onset = argmin total residual

    Parameters
    ----------
    model        : 'piecewise' → time-quadratic PLS onset (default)
                   'cosh'      → closed-form unstable tip-over solution
                                 ω(τ) = C₁(cosh(C₂τ)−1) + C (monotonic;
                                 quadratic→exponential; C₂=√d instability rate)
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

    # Optional low-pass filter (suppress IMU vibration before onset fit)
    if lpf_cutoff is not None:
        omega = lowpass_filter(t, omega, lpf_cutoff, order=lpf_order)

    t_rpm = bag.rpm.t - t0_ref

    f_col_raw = math_tools.collective_thrust_vectorized(C_T, bag.rpm.rpm)
    moments_raw = math_tools.rpm_to_moments_vectorized(
        C_T, bag.rpm.rpm, arm_length=arm_length,
    )
    moment_raw = moments_raw[:, axis_idx]

    f_col = np.interp(t, t_rpm, f_col_raw)
    moment = np.interp(t, t_rpm, moment_raw)

    # Excitation window
    idx_start, idx_end = detect_excitation_window(moment, threshold)
    win = slice(idx_start, idx_end + 1)

    # Onset fit: PLS quadratic or cosh closed-form
    if model == 'cosh':
        guess = piecewise_onset_fit(t[win], omega[win])['onset_idx']
        pw = cosh_onset_fit(t[win], omega[win], moment[win], onset_guess=guess)
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
    **kwargs,
) -> tuple[list[CriticalValueResult], list[dict]]:
    """Run piecewise extraction on every bag."""
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


def estimate_pivot_from_mocap(
    bag: BagData,
    onset_time: float,
    axis: str,
    cz: float = 0.0,
) -> dict:
    t0 = bag.odom.t[0]
    t_mc = bag.pose.t - t0
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

    mc_max_idx = mc_onset_idx + np.argmax(np.abs(d_horiz[mc_onset_idx:]))
    sl = slice(mc_onset_idx, mc_max_idx + 1)

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
        fig.savefig(Path(save_dir) / f"piecewise_fit_{axis}.png", dpi=600, bbox_inches='tight')
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
        fig.savefig(Path(save_dir) / f"pivot_circle_fit_{axis}.png", dpi=600, bbox_inches='tight')
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
        '--model', type=str, default='piecewise',
        choices=['piecewise', 'cosh'],
        help="Onset model: 'piecewise' (time-quadratic PLS, default) or "
             "'cosh' (closed-form unstable tip-over ω=C1(cosh(C2τ)-1)+C).",
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