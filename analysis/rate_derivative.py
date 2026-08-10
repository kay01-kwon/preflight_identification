#!/usr/bin/env python3
"""Angular acceleration from the rate gyro, valid at both window edges.

A Savitzky-Golay differentiator fits a local polynomial over its window.
Inside the series that is what one wants; at the FIRST and LAST
(w-1)/2 samples there is no full window, and scipy's default
``mode='interp'`` fits one polynomial to the terminal w samples and
evaluates it out to the boundary.  The derivative of an extrapolated
quadratic is not a measurement, and on a near-zero, noisy rate it can
be very wrong.

That matters here because the natural coding pattern is to slice the
excitation window first and differentiate afterwards:

    sl  = slice(onset, end)
    om  = omega[sl]
    omd = savgol_filter(om, 9, 2, deriv=1, delta=dt)      # WRONG

The onset is then the left boundary, and the onset is exactly where the
physics says omega = omega_dot = 0 -- so the whole inversion gets
anchored to a fabricated value.  Measured on
case_03/Mx/neg_Mx_045 the artefact is 3.69 rad/s^2 at the onset against
0.05 from a wide window, i.e. J_P * 3.69 = 1.23 N.m of spurious offset
carried into every subsequent sample.  The same happens, less
visibly, at the window end.

The fix costs nothing: the bags carry hundreds of samples on both sides
of the excitation window (499 before and 488 after, in that run), so
differentiate the FULL trace and slice the RESULT.

    omd = omega_dot(omega_full, dt, w)[sl]                # RIGHT

Both edges then sit deep inside real data and no extrapolation occurs.
"""
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def omega_dot(omega_full, dt, window=9, poly=2):
    """d omega / dt over a full trace, to be sliced afterwards.

    Parameters
    ----------
    omega_full : (N,) array
        The rate signal over the WHOLE bag, not the excitation window.
    dt : float
        Sample interval.
    window, poly : int
        Savitzky-Golay length (odd) and polynomial order.

    Returns
    -------
    (N,) array of the same length, aligned with ``omega_full``.
    """
    omega_full = np.asarray(omega_full, dtype=float)
    n = len(omega_full)
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 else n - 1
    if w <= poly:
        return np.gradient(omega_full, dt)
    return savgol_filter(omega_full, w, poly, deriv=1, delta=dt)


def edge_margin(n_total, onset, end, window):
    """Samples of real data outside [onset, end] on each side.

    Below (window-1)/2 on either side the corresponding edge of the
    derivative is still extrapolated; callers that care should check.
    """
    half = (int(window) - 1) // 2
    return dict(before=onset, after=n_total - 1 - end, needed=half,
                ok=onset >= half and (n_total - 1 - end) >= half)


def omega_dot_poly(tau, omega, order=5, anchored=True):
    """omega_dot from a polynomial fit in time, differentiated analytically.

    An alternative to local differentiation that uses the onset
    conditions instead of a filter window.  At the onset the physics
    fixes omega(0) = omega_dot(0) = 0, so the polynomial has no constant
    and no linear term:

        omega(tau) = sum_{k=2}^{order} a_k tau^k
        omega_dot  = sum_{k=2}^{order} k a_k tau^(k-1)

    which is the same structure the cosh solution has --
    omega = C1 (cosh C2 tau - 1) = C1 (C2^2 tau^2/2 + C2^4 tau^4/24 + ...)
    -- so the basis is not an arbitrary choice.  Three properties follow:
    the derivative is exact rather than differenced, so no window is
    involved and neither edge is extrapolated; omega_dot(0) = 0 holds by
    construction rather than by luck; and the fit averages over the
    whole window instead of 9 samples, which is where the noise
    reduction comes from.

    The cost is that a polynomial of this order cannot represent the
    real ~10 Hz content in the rate, so this deliberately estimates the
    SMOOTH tipping motion and discards the oscillation.  Whether that is
    the right thing to do depends on the question being asked; it is not
    free, because the other terms of the balance still carry the
    oscillation.

    Parameters
    ----------
    tau : (N,) array, time from the onset (tau[0] = 0).
    omega : (N,) array, rate in the tipping-positive sense.
    order : highest power retained.
    anchored : impose omega(0) = omega_dot(0) = 0.  With False the fit
        is an ordinary polynomial of the same order, for comparison.

    Returns
    -------
    (phi_fit, omega_fit, omega_dot_fit) : each (N,)
        phi_fit is the integral of omega_fit with phi(0) = 0, so all
        three kinematic quantities come from ONE polynomial and all
        three onset conditions hold at once.  Using the smoothed
        omega_dot with the raw measured attitude would be inconsistent:
        the oscillation would be removed from the inertia term and left
        in the gravity term, where nothing cancels it.
    """
    tau = np.asarray(tau, dtype=float)
    omega = np.asarray(omega, dtype=float)
    ks = np.arange(2, order + 1) if anchored else np.arange(0, order + 1)
    A = np.vstack([tau ** k for k in ks]).T
    coef, *_ = np.linalg.lstsq(A, omega, rcond=None)
    fit = A @ coef
    dks = ks[ks >= 1]
    D = np.vstack([tau ** (k - 1) for k in dks]).T
    dot = D @ (coef[ks >= 1] * dks)
    I = np.vstack([tau ** (k + 1) for k in ks]).T
    phi = I @ (coef / (ks + 1))
    return phi, fit, dot


def kinematics_from_phi(tau, phi, order=6):
    """phi, omega, omega_dot from a polynomial fit to the ATTITUDE.

    The companion of omega_dot_poly, fitted to the angle instead of the
    rate, and the preferable one when the two disagree.

    The onset conditions are conditions on the ANGLE --
    phi(0) = phi'(0) = phi''(0) = 0 -- so imposing them deletes the k =
    0, 1, 2 terms and the series starts at cubic:

        phi(tau)  = sum_{k=3}^{order} b_k tau^k
        omega     = sum_{k=3}^{order} k b_k tau^(k-1)
        omega_dot = sum_{k=3}^{order} k(k-1) b_k tau^(k-2)

    which is again the cosh structure: phi = integral of
    C1(cosh C2 tau - 1) starts at tau^3.

    Why fit the angle rather than the rate.  On this vehicle the
    reported body rate is about 0.890 of the rate implied by the
    attitude, and motion capture corroborates the ATTITUDE (the odom and
    mocap attitudes agree to 0.999 in roll).  Fitting the angle
    therefore uses the corroborated signal and needs no scale factor at
    all, where fitting the rate needs an empirical 0.890 divisor.  The
    discrepancy is not a kinematic identity -- it survives the exact
    quaternion route, omega = 2 vec(conj(q) (x) qdot), and the measured
    tipping axis sits only 2.4 deg off the body axis, so neither the
    Euler detour nor an axis misalignment accounts for it.

    The cost is one extra differentiation of a noisier signal; whether
    that is worth avoiding the empirical factor is an experiment, not a
    matter of principle.
    """
    tau = np.asarray(tau, dtype=float)
    phi = np.asarray(phi, dtype=float)
    ks = np.arange(3, order + 1)
    A = np.vstack([tau ** k for k in ks]).T
    coef, *_ = np.linalg.lstsq(A, phi, rcond=None)
    fit = A @ coef
    om = np.vstack([k * tau ** (k - 1) for k in ks]).T @ coef
    omd = np.vstack([k * (k - 1) * tau ** (k - 2) for k in ks]).T @ coef
    return fit, om, omd


def omega_dot_butter(omega_full, dt, fc, order=2):
    """Central difference over a full trace, then a zero-phase low pass.

    The alternative to fitting a polynomial over the window: differentiate
    with the plain centred difference, which has no free parameter, and
    take the noise out afterwards with a Butterworth filter run forwards
    and backwards so no phase is introduced.  The one choice left is the
    cutoff, and that is set by physics rather than taste -- the bags
    carry a real ~10 Hz structural component and blade passing well
    above it, so the cutoff belongs between them.

    Filter the FULL trace and slice the result, for exactly the reason
    in this module's docstring: filtfilt pads and reflects at the ends,
    and the onset must not sit in that transient.

    Unlike omega_dot_poly this does NOT force omega_dot(0) = 0.  That is
    a difference in kind, not degree: the polynomial imposes the static
    balance at the onset as a constraint, this measures whatever the
    gyro says there.
    """
    od = np.gradient(np.asarray(omega_full, float), dt)
    wn = fc * 2.0 * dt                      # fc / (fs/2), fs = 1/dt
    if not 0 < wn < 1:
        raise ValueError(f"cutoff {fc} Hz outside (0, {0.5 / dt}) Hz")
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, od)


def butter_lowpass(x_full, dt, fc, order=2):
    """Zero-phase low pass on a full trace, no differentiation."""
    wn = fc * 2.0 * dt
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, np.asarray(x_full, float))


def integrate_from_onset(tau, omega):
    """phi(tau) with phi(0) = 0, trapezoidal.

    Used so that phi, omega and omega_dot describe one motion when the
    derivative comes from a filter rather than from a fitted polynomial
    -- the consistency that omega_dot_poly's 'polyk' variant provides by
    construction.
    """
    c = np.concatenate([[0.0],
                        np.cumsum(0.5 * (omega[1:] + omega[:-1])
                                  * np.diff(tau))])
    return c - c[0]
