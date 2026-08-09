"""Which term actually kills the attitude-slope measurement?

Build a trajectory whose ground-effect moment is KNOWN, integrate the
exact contact dynamics, sample it at the measured rate with the measured
gyro noise, then run the same Savitzky-Golay derivative and the same
inversion.  Whatever the inversion fails to recover is attributable,
term by term.

Result, against a true slope of -2.0 mN.m/deg:

    Savitzky-Golay derivative, exact constants   +0.5 bias
    + measured gyro noise (200 draws)            +-4.2 scatter, no bias
    J_P off by 4%                                +-7.2
    z_CoM off by 5 mm                            +-2.8
    gravity arm off by 2 mm                      -0.06 (moves the level)

So the differentiation and the noise are not the limit -- the noise
averages down to +-0.4 over the 116 runs -- and J_P dominates the
systematic budget.

CAUTION: these sensitivities describe THIS synthetic trajectory and do
not transfer to the measured runs.  Along the parallel axis they
predict d(slope)/dz = +0.363 mN.m/deg per mm, i.e. +24.0 over the
190-256 mm sweep of ge_dynamics_check.py --z-sweep; the observed change
is +3.7.  Do not invert the measured residual through these numbers to
recover (J_P, z_CoM): a first attempt to do so both flipped the sign
and relied on a sensitivity seven times too large.  They are a budget,
not an estimator.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

W, Z, JP = 31.59, 0.261, 0.3335          # CAD truth
# JP = J_CAD_x + m (z^2 + l_p^2) = 0.051085 + 3.220 (0.261^2 + 0.140^2)
LP, A_ARM = 0.140, 0.150                 # pivot arm, gravity arm
MDOT = 0.65                              # N.m/s
DT, SIG_GYRO = 0.010, 0.00313            # 100 Hz, measured pre-onset std
GE0, GE_SLOPE = 0.165, -2.0e-3           # N.m and N.m/deg  <- the truth
F_THRUST = 0.0                           # folded into the balance below
SAVGOL = 9
RNG = np.random.default_rng(7)


def rhs(t, y):
    phi, om = y
    ge = GE0 + GE_SLOPE * np.rad2deg(phi)
    m = MDOT * t + (W * A_ARM - GE0 - F_THRUST * LP)   # onset-balanced ramp
    return [om, (m + F_THRUST * LP + ge
                 - W * A_ARM * np.cos(phi) + W * Z * np.sin(phi)) / JP]


sol = solve_ivp(rhs, (0, 0.60), [0.0, 0.0], max_step=DT / 4, rtol=1e-10,
                atol=1e-12, dense_output=True)
t = np.arange(0, 0.60, DT)
phi, om = sol.sol(t)
keep = np.rad2deg(phi) < 7.2
t, phi, om = t[keep], phi[keep], om[keep]
deg = np.rad2deg(phi)
m = MDOT * t + (W * A_ARM - GE0 - F_THRUST * LP)


def invert(om_series, jp, z, a_arm, use_exact_dot=False):
    if use_exact_dot:
        omd = np.gradient(om_series, DT)          # noise-free reference
    else:
        omd = savgol_filter(om_series, SAVGOL, 2, deriv=1, delta=DT)
    ge = (jp * omd - m - F_THRUST * LP
          + W * a_arm * np.cos(phi) - W * z * np.sin(phi))
    sel = deg > 0.3
    s, i = np.polyfit(deg[sel], 1e3 * ge[sel], 1)
    return s, i


print(f"truth: level {1e3*GE0:.0f} mN.m, slope {1e3*GE_SLOPE:+.1f} mN.m/deg")
print(f"excursion reached {deg[-1]:.1f} deg in {t[-1]:.2f} s\n")
print(f"{'case':<44}{'slope':>9}{'level':>9}")
print('-' * 62)
s, i = invert(om, JP, Z, A_ARM, use_exact_dot=True)
print(f"{'exact constants, exact derivative':<44}{s:9.2f}{i:9.1f}")
s, i = invert(om, JP, Z, A_ARM)
print(f"{'exact constants, Savitzky-Golay derivative':<44}{s:9.2f}{i:9.1f}")
noisy = om + RNG.normal(0, SIG_GYRO, len(om))
ss = [invert(om + RNG.normal(0, SIG_GYRO, len(om)), JP, Z, A_ARM)
      for _ in range(200)]
sa = np.array([v[0] for v in ss]); ia = np.array([v[1] for v in ss])
print(f"{'+ measured gyro noise (200 draws)':<44}"
      f"{sa.mean():9.2f}{ia.mean():9.1f}   +-{sa.std():.2f} / +-{ia.std():.1f}")
for dj, lab in ((0.04, 'J_P +4%'), (-0.04, 'J_P -4%'),
                (0.23, 'J_P -23% (roll calibration)'),):
    s, i = invert(om, JP * (1 + dj if dj < 0.2 else -dj), Z, A_ARM)
    print(f"{'exact derivative, ' + lab:<44}{s:9.2f}{i:9.1f}")
for dz, lab in ((0.005, 'z_CoM +5 mm'), (-0.005, 'z_CoM -5 mm')):
    s, i = invert(om, JP, Z + dz, A_ARM)
    print(f"{'exact derivative, ' + lab:<44}{s:9.2f}{i:9.1f}")
s, i = invert(om, JP, Z, A_ARM + 0.002)
print(f"{'exact derivative, gravity arm +2 mm':<44}{s:9.2f}{i:9.1f}")
