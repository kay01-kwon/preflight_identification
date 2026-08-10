"""Raw 200 Hz IMU gyro against the 100 Hz odometry angular velocity.

The pipeline takes omega from odom, the flight controller's fused
estimate, although the bags also carry /mavros/imu/data_raw at twice the
rate.  The obvious worry is that odom is doubly filtered -- by the EKF
and then by the Savitzky-Golay derivative -- and that the raw gyro would
give a truer omega_dot.

It does not.  At a MATCHED time window (9 samples at 100 Hz against 19 at
200 Hz, both about 90 ms) the raw gyro puts the peak omega_dot 32-145%
above the fitted cosh and drives the ground-effect slope to -139, -177
and +40 mN.m/deg, against -19, -12 and -15 from odom.

The spectrum says why.  On pos_Mx_01 the raw gyro carries 98.8% of its
AC power in 50-100 Hz, peaking at 90.8 Hz -- the blade-passing frequency
of six rotors near 5000 rpm -- at an RMS of 716 mrad/s.  The odom
angular velocity carries 98.8% of its power in 2-20 Hz at 176 mrad/s,
the EKF having already rejected the vibration.  A 90 ms Savitzky-Golay
window does not remove 90 Hz content well enough to differentiate
through it.

Using the raw gyro would therefore need a notch at blade passing first.
The odom choice is what makes the derivative usable at all; it is not an
oversight.  The cost is that omega carries the EKF's own lag and
smoothing, which cannot be characterised from the bags.
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from analysis.rate_derivative import omega_dot
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
from error_budget import ge_moment
from analysis.pnls_constants import PNLS_CONSTANTS

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
G, Z = 9.81, 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}; LP = {'x': 0.140, 'y': 0.110}
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF_MM = {('case_02','Mx'):-14.29}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}

case, axname, axis = 'case_02', 'Mx', 'x'
mass = MASS[case]; W = mass * G
j_p = J_CAD[axis] + mass * (Z ** 2 + LP[axis] ** 2)
c2f, kf = PNLS_CONSTANTS[(case, axname)]
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(ROOT / case / axname)
    crits, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2f, ramp_gain=kf)
by = {b.name: b for b in bags}

print(f"{'bag':<12}{'src':>6}{'Hz':>6}{'w':>5}{'peak om_dot':>12}"
      f"{'vs cosh':>9}{'GE slope':>10}{'GE level':>10}")
for name in ('pos_Mx_01', 'pos_Mx_045', 'pos_Mx_120'):
    crit = next((c for c in crits if c.bag_name == name), None)
    if crit is None:
        continue
    bag = by[name]
    s = 1.0
    roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
    phi_all = roll if axis == 'x' else pitch
    sig_o = cvp.prepare_signals(bag, axis)
    n = min(len(phi_all), len(sig_o['t']))
    _, i1 = cvp.detect_excitation_window(
        sig_o['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    i1 = min(i1, n - 1)
    t_on, t_end = sig_o['t'][crit.onset_idx], sig_o['t'][i1]
    # the guard only has to hold where the trace is read back, which
    # is between the onset and the window end
    raw = ge_moment(bag, sig_o, axis, n, True,
                    window=slice(crit.onset_idx, i1 + 1))
    piv = cvp.estimate_pivot_from_mocap(bag, crit.onset_time, axis)
    lp = piv['pivot_abs'] * 1e-3 if not np.isnan(piv['pivot_abs']) else LP[axis]
    a = lp + s * OFF_SIGN[axname] * OFF_MM[(case, axname)] * 1e-3
    for src in ('odom', 'imu'):
        sig = cvp.prepare_signals(bag, axis, omega_source=src)
        k0 = int(np.searchsorted(sig['t'], t_on))
        k1 = int(np.searchsorted(sig['t'], t_end))
        tt = sig['t'][k0:k1 + 1]
        om = sig['omega'][k0:k1 + 1]
        dt = float(np.median(np.diff(tt)))
        # the derivative is taken on the WHOLE trace of this source and
        # sliced afterwards; taken on the slice, both edges would be
        # extrapolated (analysis/rate_derivative.py)
        om_src_full = sig['omega']
        tau = tt - tt[0]
        # everything else resampled onto this clock
        phi = np.interp(tt, sig_o['t'][:n], phi_all[:n] - phi_all[crit.onset_idx])
        m = np.interp(tt, sig_o['t'], sig_o['moment'])
        f = np.interp(tt, sig_o['t'], sig_o['f_col'])
        gem = np.interp(tt, sig_o['t'][:n], raw)
        om = om - om[0] + sig_o['omega'][crit.onset_idx]
        wid = 9 if src == 'odom' else 19        # matched to ~90 ms
        omd = omega_dot(om_src_full, dt, wid)[k0:k1 + 1]
        mdot = float(np.polyfit(tau, m, 1)[0])
        omd_fit = kf * mdot * c2f * np.sinh(np.clip(c2f * tau, 0, 30))
        ge = j_p * omd - m - f * lp + W * a * np.cos(phi) - W * Z * np.sin(phi)
        deg = np.rad2deg(phi)
        sd, id_ = np.polyfit(deg, 1e3 * ge, 1)
        print(f"{name if src=='odom' else '':<12}{src:>6}{1/dt:6.0f}{wid:5d}"
              f"{omd.max():12.3f}{100*(omd.max()/omd_fit.max()-1):+8.1f}%"
              f"{sd:10.2f}{id_:10.1f}")
    sm = float(np.polyfit(np.rad2deg(phi), 1e3 * gem, 1)[0])
    print(f"{'':12}{'model':>6}{'':6}{'':5}{'':12}{'':9}{sm:10.2f}")
