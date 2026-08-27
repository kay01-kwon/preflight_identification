#!/usr/bin/env python3
"""Synthetic verification of the envelope-witness chain, four checks.

Integrates the exact nonlinear tip-over about the pivot edge,

    J_P phidd = M_crit + Mdot*tau - W*l_arm*cos(phi) + W*z*sin(phi),

so every quantity the theory names is known exactly: the nominal
solution, the deviation e_omega, the forcing rho, the true onset.
Gaussian gyro noise and a bias are added, the deployed four-parameter
family is fitted by least squares over the full window (pre-onset
baseline included), and four claims are checked per ramp rate:

  1  witness inequality   RMS(r_fit) <= RMS(e_omega + n)
  2  envelope             |e_omega| <= E pointwise, and
                          RMS(e_omega) <= rho_bar*K*C2*sqrt(B(x)/x)
  3  onset shift          fitted t* earlier than the true onset and
                          inside the a-priori ceiling of (97a),
                          |t*| <= artanh(rho_bar*C2/Mdot)/C2; the
                          end-matched value of (97b) is printed as a
                          reference (the free fit trades amplitude
                          against onset along the ridge, so it need
                          not land on it exactly)
  4  pure delay           shifting the whole record's clock changes
                          the fitted residual not at all and moves
                          t* by exactly the shift
  5  comparison system    delta_e = e_omega - beta*sinh(C2 tau) with
                          the end-matched beta sits under the
                          Dirichlet cap (manuscript (107)),
                          (W*arm*phi*omega_max/12)*K -- measurable
                          here because the constants are exact
  6  realised shift       the deployed constrained readout's
                          critical-moment shift Mdot*|t* - t_on|
                          sits under the ceiling of (104),
                          (Mdot/C2)*artanh(rho_bar*C2/Mdot)

Usage
-----
  python analysis/witness_verify.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq

RNG = np.random.default_rng(7)
G = 9.81
M_KG, Z, L_ARM = 3.066, 0.272, 0.150          # arm = l_p 0.140 + off 0.010
J_CAD = 0.0505
PHI_MAX = np.deg2rad(5.0)
R_PHI = 1 / 7
SIGMA, BIAS = 2.450e-4, 2.0e-3                # rad/s
FS, T_PRE = 200.0, 1.0

W = M_KG * G
J_P = J_CAD + M_KG * (Z ** 2 + (L_ARM - 0.010) ** 2)
C2 = np.sqrt(W * Z / J_P)
K = 1.0 / (W * Z)


def simulate(mdot):
    """Exact tip-over from the true onset; returns the sampled record."""
    x = brentq(lambda v: np.sinh(v) - v - PHI_MAX * W * Z * C2 / mdot,
               1e-6, 30)
    t_end = x / C2

    def f(t, s):
        phi, om = s
        m_app = W * L_ARM + mdot * t          # ramp through the onset
        return [om, (m_app - W * L_ARM * np.cos(phi)
                     + W * Z * np.sin(phi)) / J_P]

    tau = np.arange(0, t_end, 1 / FS)
    sol = solve_ivp(f, (0, t_end), [0, 0], t_eval=tau, rtol=1e-10,
                    atol=1e-12)
    om_true = sol.y[1]
    om_nom = K * mdot * (np.cosh(C2 * tau) - 1)
    e_om = om_true - om_nom

    t_rec = np.concatenate([np.arange(-T_PRE, 0, 1 / FS), tau])
    om_rec = np.concatenate([np.zeros(int(T_PRE * FS)), om_true])
    y = om_rec + BIAS + RNG.normal(0, SIGMA, len(t_rec))
    return t_rec, y, tau, e_om, x


def fit(t, y):
    """The deployed family: (C1*, C2*, t*, C), full-window LS."""
    def model(p):
        c1, c2, ts, c = p
        h = np.full_like(t, c)
        m = t >= ts
        h[m] += c1 * (np.cosh(c2 * (t[m] - ts)) - 1.0)
        return h

    p0 = [K * 0.5, C2, 0.02, 0.0]
    r = least_squares(lambda p: model(p) - y, p0,
                      x_scale=[1e-3, 1.0, 1e-2, 1e-3], xtol=1e-15,
                      ftol=1e-15)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def main():
    print(f'C2 {C2:.3f} rad/s   J_P {J_P:.4f}   K {K:.5f}')
    hdr = ('rate  | RMS(r)   RMS(e+n)  ok | RMS(e)   cap      ok  '
           'ptwise | t*      ref(97b)  ceil(97a)  ok | '
           'delay-res  d(t*)    ok | RMS(de) cap107  ok | '
           'dM      ceil104  ok')
    print(hdr)
    print('-' * len(hdr))
    for mdot in (0.10, 0.45, 1.20):
        t, y, tau, e_om, x = simulate(mdot)
        n_pre = int(T_PRE * FS)

        # -- check 1: witness inequality --------------------------------
        # witness = nominal + bias; its residual is e_omega + noise
        p, rms_fit = fit(t, y)
        nom_full = np.concatenate([np.zeros(n_pre),
                                   K * mdot * (np.cosh(C2 * tau) - 1)])
        rms_wit = float(np.sqrt(np.mean((y - BIAS - nom_full) ** 2)))
        ok1 = rms_fit <= rms_wit

        # -- check 2: envelope ------------------------------------------
        rho_bar = R_PHI * 0.5 * W * L_ARM * PHI_MAX ** 2
        env_pt = rho_bar * K * C2 * np.sinh(C2 * tau)
        ptwise = bool(np.all(np.abs(e_om[1:]) <= env_pt[1:]))
        B = 0.25 * np.sinh(2 * x) - 0.5 * x
        cap = rho_bar * K * C2 * np.sqrt(B / x)
        rms_e = float(np.sqrt(np.mean(e_om ** 2)))
        ok2 = rms_e <= cap

        # -- check 3: the onset shift, its ceiling, and (97b) -----------
        c1s, c2s, ts, _ = p
        beta = e_om[-1] / np.sinh(x)
        dtc_pred = np.arctanh(-beta / (K * mdot)) / C2   # < 0
        u_adm = rho_bar * C2 / mdot
        dtc_cap = np.arctanh(min(u_adm, 0.999)) / C2
        ok3 = (ts < 0) and (abs(ts) <= dtc_cap)

        # -- check 4: pure delay absorbed -------------------------------
        p_d, rms_d = fit(t + 0.080, y)
        ok4 = (abs(rms_d - rms_fit) < 1e-9
               and abs((p_d[2] - ts) - 0.080) < 1e-4)

        # -- check 5: the Dirichlet cap on delta_e ((107)) --------------
        de = e_om - beta * np.sinh(C2 * tau)
        rms_de = float(np.sqrt(np.mean(de ** 2)))
        om_max = K * mdot * (np.cosh(x) - 1) \
            + rho_bar * np.sinh(x) / (J_P * C2)
        cap107 = (W * L_ARM * PHI_MAX * om_max / 12.0) * K
        ok5 = rms_de <= cap107

        # -- check 6: realised critical-moment shift vs (104) -----------
        # the deployed constrained readout (sub-sample refined), on a
        # record whose true onset is exactly t = 0
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import critical_value_getter_piecewise as cvp
        pw6 = cvp.cosh_onset_fit(t, y, np.zeros_like(t),
                                 onset_guess=None, c2_fixed=C2,
                                 moment_floor=0.0, ramp_gain=K,
                                 ramp_rate=mdot)
        dm = mdot * abs(pw6['onset_t'])
        u6 = rho_bar * C2 / mdot
        ceil104 = (mdot / C2) * np.arctanh(min(u6, 0.999))
        ok6 = dm <= ceil104

        print(f'{mdot:4.2f}  | {np.degrees(rms_fit):7.4f} '
              f'{np.degrees(rms_wit):8.4f} {"Y" if ok1 else "N":>2} '
              f'| {np.degrees(rms_e):7.4f} {np.degrees(cap):7.4f} '
              f'{"Y" if ok2 else "N":>2}   {"Y" if ptwise else "N":>3}  '
              f'| {1e3*ts:+7.2f} {1e3*dtc_pred:+8.2f} {1e3*dtc_cap:8.2f} ms '
              f'{"Y" if ok3 else "N"} '
              f'| {np.degrees(rms_d):8.4f} {1e3*(p_d[2]-ts):+7.2f} ms '
              f'{"Y" if ok4 else "N"} '
              f'| {np.degrees(rms_de):7.4f} {np.degrees(cap107):7.4f} '
              f'{"Y" if ok5 else "N"} '
              f'| {1e3*dm:6.3f} {1e3*ceil104:7.3f} mN·m '
              f'{"Y" if ok6 else "N"}')


if __name__ == '__main__':
    main()
