#!/usr/bin/env python3
"""Two anchors, two rate trends: the rho_dot route vs the rho-level
K route, and why the bound's growth with Mdot is an anchor choice.

The realised delta_e is rate-flat while the deployed model term grows
1.96x across the sweep.  The resolution: the Dirichlet boundary zero
sits directly under the rho_dot peak, so the true response is governed
by the rho LEVEL (int rho_dot = rho(T) ~ phi_end^2, flat in rate), not
the rho_dot peak (~ omega_max, doubling).  A bound anchored on rho(T)
-- the Green-kernel route, |delta_e| <= (1/J_P) int |K| rho_env with
K the two-piece kernel of the Dirichlet representation -- inherits the
flat trend (x1.03 across the sweep) at a looser level (1.16-1.19
deg/s).  Both routes are valid bounds, so min(A, K) is valid: on this
campaign the rho_dot route wins everywhere (crossover near Mdot ~ 2),
coverage 140/140, worst used 0.86.

Usage: python analysis/k_route_trend.py
"""
import numpy as np
sys.path.insert(0, '/home/user/preflight_identification/analysis')
from fit_quality_bound import ARMS, W, BETA_M, rho_bar
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from kernel_free_bound import SHAPE_SAFETY, model_term

HERE = '/home/user/preflight_identification/analysis'
with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
    rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
kqm = float(np.median([d['kq'] for d in rows if d['kq'] is not None]))

def k_route(d):
    """rho-LEVEL shaped bound: sup_tau (1/J_P) int |K(tau,s)| env(s) ds,
    env = rho2(T) e^{2C2(s-T)} + rho1(T) e^{C2(s-T)},
    rho2 = 1.05 * (1/2) W a phi_max^2 (arm; Wz term dropped by sign),
    rho1 = |beta_M| dM_win phi_max."""
    c2, k = d['c2'], d['k']
    x = c2 * d['tau'][-1]; wz = 1.0 / k
    r2 = SHAPE_SAFETY * 0.5 * W * ARMS[d['axis']] * PHI_BOX ** 2
    r1 = abs(BETA_M) * d['dm_win'] * PHI_BOX
    n = 1200
    s = np.linspace(0, x, n)                      # scaled time
    env = r2 * np.exp(2 * (s - x)) + r1 * np.exp(s - x)
    ds = s[1] - s[0]
    best = 0.0
    for tp in np.linspace(0, x, 240):
        Kv = np.where(s <= tp,
                      np.sinh(x - tp) * np.cosh(s) / np.sinh(x),
                      np.sinh(tp) * np.cosh(x - s) / np.sinh(x))
        best = max(best, float(np.sum(np.abs(Kv) * env) * ds))
    return float(np.rad2deg(best * c2 / wz))

for d in rows:
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
    de_sup, dpre = model_term(d, rb)
    d['mA'] = de_sup                       # rho_dot route (sup form)
    d['mK'] = k_route(d)                   # rho-level K route
    d['mmin'] = min(d['mA'], d['mK'])
    kq = d['kq'] if d['kq'] is not None else kqm
    nh = d['rms_n'] * np.sqrt(1.0 + (s_med * kq) ** 2)
    d['cap_min'] = d['mmin'] + dpre + nh

rates = sorted({d['rate'] for d in rows})
print(f"{'Mdot':>6}{'rho_dot rt':>12}{'K (rho) rt':>12}{'min':>8}"
      f"{'used(min)':>11}")
for rt in rates:
    v = [d for d in rows if d['rate'] == rt]
    a = np.mean([d['mA'] for d in v]); kk = np.mean([d['mK'] for d in v])
    m = np.mean([d['mmin'] for d in v])
    u = np.mean([d['rms_min'] / d['cap_min'] for d in v])
    print(f"{rt:6.2f}{a:12.3f}{kk:12.3f}{m:8.3f}{u:11.3f}")
ins = sum(1 for d in rows if d['rms_min'] <= d['cap_min'])
print(f"\ncoverage with min(both routes): {ins}/140, "
      f"worst used {max(d['rms_min']/d['cap_min'] for d in rows):.3f}")
ta = [np.mean([d['mA'] for d in rows if d['rate'] == r]) for r in rates]
tk = [np.mean([d['mK'] for d in rows if d['rate'] == r]) for r in rates]
tm = [np.mean([d['mmin'] for d in rows if d['rate'] == r]) for r in rates]
print(f"trend (fast/slow): rho_dot {ta[-1]/ta[0]:.2f}x, "
      f"K {tk[-1]/tk[0]:.2f}x, min {tm[-1]/tm[0]:.2f}x")
