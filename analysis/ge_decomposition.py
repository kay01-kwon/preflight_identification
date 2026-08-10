"""Why does the model ground-effect moment look flat while M ramps?

It is not flat -- it is the near-cancellation of two opposite trends.
dM_GE = sum_i gain_i(height) * T_i * arm_i, and during the ramp

  the differential thrust grows, so sum T_i arm_i grows with the applied
  moment; this is the b*M term of the static branch, b = 0.0431;
  the vehicle tilts, so the far rotors -- the long arms -- rise away
  from the ground and lose their gain.

Freezing one at a time on case_02/Mx separates them [mN.m/deg]:

    run           dM_win    full   thrust only   tilt only
    pos_Mx_01     0.087    -3.08        +0.40       -3.44
    pos_Mx_045    0.245    -1.80        +1.66       -3.32
    pos_Mx_120    0.464    -0.55        +3.06       -3.35

The thrust part scales with the moment increment over the window, 5.3x
across these runs against 7.6x in the slope.  The tilt part is pure
geometry and barely moves.  So the faster the ramp, the flatter the net
ground-effect moment -- and the apparent flatness in fig:gedyn is that
cancellation, not an absence of dependence.

This also sharpens the residual.  The largest single component the model
has is the tilt part at -3.4 mN.m/deg, and the measured residual is -44,
thirteen times larger.  And the thrust part varies eightfold with ramp
rate while the residual does not (-48.6 slow, -38.6 mid, -55.6 fast, not
even monotone), so the residual is not a thrust-proportional term --
consistent with C_T having been ruled out separately.
"""
import contextlib, io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from utils import math_tools
import error_budget as eb
from analysis.pnls_constants import PNLS_CONSTANTS

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
case, axname, ax = 'case_02', 'Mx', 'x'
c2f, kf = PNLS_CONSTANTS[(case, axname)]
with contextlib.redirect_stdout(io.StringIO()):
    bags = load_excitation_dataset(ROOT / case / axname)
    crits, _ = cvp.extract_piecewise_batch(bags, ax, cosh_c2=c2f, ramp_gain=kf)
by = {b.name: b for b in bags}

print(f"{'bag':<12}{'dM_win':>8}{'dphi':>7} |{'full':>18}|{'thrust only':>18}"
      f"|{'tilt only':>18}")
print(f"{'':12}{'[N.m]':>8}{'[deg]':>7} |{'slope':>9}{'change':>9}"
      f"|{'slope':>9}{'change':>9}|{'slope':>9}{'change':>9}")
for name in ('pos_Mx_01', 'pos_Mx_045', 'pos_Mx_120'):
    crit = next((c for c in crits if c.bag_name == name), None)
    if crit is None:
        continue
    bag = by[name]
    sig = cvp.prepare_signals(bag, ax)
    roll, pitch = math_tools.quaternion_to_euler_vectorized(bag.odom.quaternion)
    phi_all = roll if ax == 'x' else pitch
    n = min(len(phi_all), len(sig['t']))
    _, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(ax))
    j = crit.onset_idx; i1 = min(i1, n - 1); sl = slice(j, i1 + 1)
    deg = np.rad2deg(phi_all[sl] - phi_all[j])
    dm = sig['moment'][i1] - sig['moment'][j]

    full = eb.ge_moment(bag, sig, ax, n, True, window=sl)[sl]
    # freeze the attitude at the onset: only the thrusts move
    q = bag.odom.quaternion.copy()
    bag_t = bag.__class__(**{**bag.__dict__,
                             'odom': bag.odom.__class__(**{
                                 **bag.odom.__dict__,
                                 'quaternion': np.repeat(q[j:j+1], len(q), 0)})})
    thr_only = eb.ge_moment(bag_t, sig, ax, n, True, window=sl)[sl]
    # freeze the thrusts at the onset: only the attitude moves
    rpm = bag.rpm.rpm.copy()
    k0 = int(np.searchsorted(bag.rpm.t - bag.odom.t[0], sig['t'][j]))
    rpm_f = np.repeat(rpm[k0:k0+1], len(rpm), 0)
    bag_a = bag.__class__(**{**bag.__dict__,
                             'rpm': bag.rpm.__class__(**{**bag.rpm.__dict__,
                                                          'rpm': rpm_f})})
    tilt_only = eb.ge_moment(bag_a, sig, ax, n, True, window=sl)[sl]

    cells = []
    for v in (full, thr_only, tilt_only):
        s_ = np.polyfit(deg, 1e3 * v, 1)[0]
        cells.append(f"{s_:9.2f}{1e3*(v[-1]-v[0]):9.1f}")
    print(f"{name:<12}{dm:8.3f}{deg[-1]:7.2f} |" + '|'.join(cells))
print()
print("'change' is the end-to-onset difference in mN.m over the window.")
