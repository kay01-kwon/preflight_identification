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
from scipy.signal import savgol_filter


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
