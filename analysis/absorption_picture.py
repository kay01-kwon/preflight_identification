#!/usr/bin/env python3
"""What the estimator absorbs, drawn -- and a check on who gets weighted.

Two claims in the current drafts disagree with each other, and the
disagreement is sharp enough that only one can survive.

  (107) says the onset weight w(s) is NON-INCREASING, so forcing that
        acts EARLY is weighted MOST.
  the "sinh 0 = 0" paragraph of access_sec6e_onset.tex says a
        perturbation acting at the onset moves the threshold NOT AT ALL,
        because the onset-displacing half of the kernel carries sinh C2 s.

The second reads the split e = [cosh P - sinh Q]/J_P and assigns the
cosh half to the amplitude.  That is right only if the amplitude is FREE
to take it.  With C1 pinned the cosh half cannot go there, and it does
not vanish either -- it leaks into the onset through the projection,
because cosh and sinh are only 3.7 degrees apart on this window.

This script settles it against the actual estimator: the piecewise model
with C1 and C2 pinned, the onset swept exhaustively, the baseline in
closed form.  No linearisation anywhere in the measurement.

Usage: python analysis/absorption_picture.py [out.png]
"""
import sys

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C2, J_P = 5.046, 0.4260
WZ = J_P * C2 ** 2
W, ARM, ZCOM, BETA = 31.59, 0.160, 0.30, 0.03446
PHI_MAX = np.deg2rad(10.0)
RHO_BAR = 0.01204                      # N.m, the box value for roll
PRE = 0.15                             # pre-onset record, s
N = 4001


def window(mdot):
    """Solve sinh x - x = phi_max Wz C2 / Mdot for the window length."""
    rhs = PHI_MAX * WZ * C2 / mdot
    return brentq(lambda x: np.sinh(x) - x - rhs, 1e-6, 20.0)


class Run:
    """One synthetic run with a KNOWN true onset at tau = 0."""

    def __init__(self, mdot):
        self.mdot = mdot
        self.x = window(mdot)
        self.te = self.x / C2
        self.c1 = mdot / WZ
        self.t = np.linspace(-PRE, self.te, N)
        self.tau = np.clip(self.t, 0.0, None)
        self.f = np.where(self.t < 0, 0.0,
                          self.c1 * (np.cosh(C2 * self.tau) - 1.0))
        self.w = np.gradient(self.t)
        self.w[0] *= 0.5
        self.w[-1] *= 0.5
        self.post = self.t >= 0.0

    def n(self, v):                    # L2 over the post-onset window
        return float(np.sqrt(np.sum((v ** 2 * self.w)[self.post])))

    def rms(self, v):
        m = self.post
        return float(np.sqrt(np.sum((v ** 2 * self.w)[m]) / self.w[m].sum()))

    def duhamel(self, rho):
        """e(tau) = (1/J_P) int_0^tau cosh(C2 (tau-s)) rho(s) ds."""
        m = self.post
        tt, rr = self.tau[m], rho[m]
        out = np.zeros(tt.size)
        for i in range(1, tt.size):
            k = np.cosh(np.clip(C2 * (tt[i] - tt[:i + 1]), 0, 40))
            out[i] = np.trapz(k * rr[:i + 1], tt[:i + 1]) / J_P
        full = np.zeros_like(self.t)
        full[m] = out
        return full

    def fit(self, y, dmax=0.14, nd=2801):
        """The real estimator: sweep the onset, baseline in closed form."""
        best = (np.inf, 0.0, None)
        for d in np.linspace(-dmax, dmax, nd):
            shape = np.where(self.t < d, 0.0,
                             self.c1 * (np.cosh(np.clip(
                                 C2 * (self.t - d), 0, 40)) - 1.0))
            c = float(np.sum((y - shape) * self.w) / self.w.sum())
            r = y - shape - c
            v = float(np.sum(r ** 2 * self.w))
            if v < best[0]:
                best = (v, d, shape + c)
        return best[1], best[2]


def weight(run, s):
    """(107): w(s), the normalised onset weight."""
    g = np.linspace(0.0, run.te, 2001)
    num = np.array([np.trapz(np.sinh(C2 * g[g >= si])
                             * np.cosh(C2 * (g[g >= si] - si)),
                             g[g >= si]) for si in np.atleast_1d(s)])
    den = np.trapz(np.sinh(C2 * g) ** 2, g)
    return WZ * num / (J_P * C2 * den)


def profile(run, p):
    """A one-parameter family of rho shapes, window MEAN pinned at rho_bar.

    p >= 0 rises across the window, p < 0 falls.  Both are bounded --
    a negative power of tau would be singular at the onset, and the
    point here is the ARRANGEMENT of rho, not its peak.
    """
    z = np.clip(run.tau / run.te, 0.0, 1.0)
    u = z ** p if p >= 0 else (1.0 - z) ** (-p)
    u = np.where(run.post, u, 0.0)
    m = run.post
    mean = np.trapz(u[m], run.t[m]) / run.te
    return u * (RHO_BAR / mean)


def physical(run):
    """The real channels along the nominal trajectory."""
    phi = run.c1 * (np.sinh(C2 * run.tau) / C2 - run.tau)
    phi = np.where(run.post, phi, 0.0)
    g2 = W * ARM * np.cos(phi) - W * ZCOM * np.sin(phi)
    return np.where(run.post,
                    0.5 * g2 * phi ** 2 + BETA * run.mdot * run.tau * phi, 0.0)


def zoom(ax, run, y, g, d, c_d, c_g, c_f):
    """Inset on the onset region: at full scale the two curves are one."""
    ins = ax.inset_axes([0.09, 0.50, 0.47, 0.42])
    m = (run.t > -0.045) & (run.t < 0.105)
    ins.plot(run.t[m], y[m], '-', color=c_d, lw=3.0, alpha=0.55)
    ins.plot(run.t[m], g[m], '--', color=c_g, lw=1.5)
    ins.axvline(0.0, color=c_f, lw=1.1)
    ins.axvline(d, color=c_g, lw=1.1, ls='--')
    ins.annotate('', xy=(d, 0.004), xytext=(0.0, 0.004),
                 arrowprops=dict(arrowstyle='<->', color='k', lw=1.0))
    ins.text((d + 0.0) / 2, 0.0048, f'{1e3*d:.1f} ms', ha='center',
             fontsize=8)
    ins.set_ylim(-0.002, 0.010)
    ins.tick_params(labelsize=6.5)
    ins.set_title('onset, zoomed', fontsize=7.5, pad=2)
    ins.grid(alpha=0.2, lw=0.3)
    return ins


def measure(run, rho):
    e = run.duhamel(rho)
    d, g = run.fit(run.f + e)
    return dict(e=e, d=d, g=g, r=run.f + e - g,
                dM=run.mdot * d, rms=run.rms(run.f + e - g))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'absorption_picture.png'
    fast, slow = Run(1.20), Run(0.10)
    print(f"\n  window: x = {fast.x:.3f} at Mdot 1.20,"
          f" {slow.x:.3f} at 0.10;  rho_bar = {1e3*RHO_BAR:.2f} mN.m\n")

    # ---- 1. is the weight largest at s = 0, or smallest?
    ss = np.linspace(0, fast.te, 9)
    ws = weight(fast, ss)
    print("  (107) weight w(s), and the threshold error a unit impulse")
    print("  at s actually produces, measured on the real estimator\n")
    print(f"  {'s [s]':>7}{'w(s) [1/s]':>12}{'w(s)*m predicted':>19}"
          f"{'measured':>11}{'ratio':>8}")
    for si, wi in zip(ss[:-1], ws[:-1]):
        sd = 0.004
        rho = np.where(fast.post,
                       np.exp(-0.5 * ((fast.tau - si) / sd) ** 2), 0.0)
        # the mass ACTUALLY delivered: a bump at s = 0 is half cut off by
        # the window, and crediting it with a whole one would fake a
        # factor of two into the first row.
        m_act = float(np.trapz(rho[fast.post], fast.t[fast.post]))
        got = measure(fast, rho)['dM']
        print(f"  {si:7.3f}{wi:12.4f}{-1e3*wi*m_act:19.3f}"
              f"{1e3*got:11.3f}{-got/(wi*m_act):8.3f}")

    # ---- 2. the worst case the bound allows
    print(f"\n  rho profiles: (tau/te)^p, window mean pinned at rho_bar\n")
    print(f"  {'p':>6}{'shape':>14}{'|dM_crit|':>12}{'/ rho_bar':>11}"
          f"{'onset [ms]':>12}{'resid RMS':>12}{'deg/s':>9}")
    rows = []
    for p, name in ((-2.0, 'falls fast'), (-1.0, 'falls'), (0.0, 'flat'),
                    (1.0, 'linear'), (2.0, 'quadratic'), (4.0, 'quartic')):
        rho = profile(fast, p)
        g = measure(fast, rho)
        rows.append((p, name, g))
        print(f"  {p:6.1f}{name:>14}{1e3*abs(g['dM']):12.3f}"
              f"{abs(g['dM'])/RHO_BAR:11.3f}{1e3*g['d']:12.2f}"
              f"{g['rms']:12.5f}{np.rad2deg(g['rms']):9.3f}")
    ph = measure(fast, physical(fast))
    print(f"  {'':6}{'physical':>14}{1e3*abs(ph['dM']):12.3f}"
          f"{abs(ph['dM'])/RHO_BAR:11.3f}{1e3*ph['d']:12.2f}"
          f"{ph['rms']:12.5f}{np.rad2deg(ph['rms']):9.3f}")

    # ---- 3. the two channels, separated
    print(f"\n  the two halves of the kernel, driven separately\n")
    print(f"  {'channel':>12}{'|dM_crit| mN.m':>17}{'resid RMS deg/s':>18}")
    amp = fast.duhamel(np.zeros_like(fast.tau))
    amp = np.where(fast.post, 0.02 * (np.cosh(C2 * fast.tau) - 1.0)
                   * fast.c1, 0.0)     # a pure amplitude error, 2%
    d_amp, g_amp = fast.fit(fast.f + amp)
    print(f"  {'amplitude':>12}{1e3*fast.mdot*d_amp:17.3f}"
          f"{np.rad2deg(fast.rms(fast.f + amp - g_amp)):18.4f}")
    on = np.where(fast.post, -0.010 * (-fast.c1 * C2
                                       * np.sinh(C2 * fast.tau)), 0.0)
    d_on, g_on = fast.fit(fast.f + on)
    print(f"  {'onset':>12}{1e3*fast.mdot*d_on:17.3f}"
          f"{np.rad2deg(fast.rms(fast.f + on - g_on)):18.4f}")

    # ---- the picture
    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.055, right=0.985, top=0.885, bottom=0.085)
    C_D, C_F, C_G, C_R = '0.35', '#2874a6', '#c0392b', '#148f77'

    # (a) cosh and sinh, normalised -- why anything can be absorbed
    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(0, fast.x, 800)
    def unit(v):
        v = v - np.trapz(v, g) / fast.x
        return v / np.sqrt(np.trapz(v ** 2, g))
    ch, sh = unit(np.cosh(g)), unit(np.sinh(g))
    ax.plot(g, ch, '-', color=C_F, lw=2.2, label=r'$\cosh C_2\tau$')
    ax.plot(g, sh, '--', color=C_G, lw=2.0, label=r'$\sinh C_2\tau$')
    lo = ch - np.trapz(ch * sh, g) * sh
    ang = np.rad2deg(np.arcsin(np.sqrt(max(np.trapz(lo ** 2, g), 0))))
    ax.plot(g, 12 * lo, '-', color=C_R, lw=1.4,
            label=f'what is left, ' + r'$\times12$')
    ax.set_title(f'(a) the two shapes are {ang:.2f}$^\\circ$ apart\n'
                 'mean removed, unit norm', fontsize=10.5)
    ax.set_xlabel(r'$C_2\tau$', fontsize=9)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.25, lw=0.4)

    # (b) the weight: early is weighted MOST
    ax = fig.add_subplot(gs[0, 1])
    sg = np.linspace(0, fast.te, 400)
    ax.plot(sg, weight(fast, sg), '-', color=C_F, lw=2.2)
    ax.axhline(1.0 / fast.te, color=C_D, ls=':', lw=1.3)
    ax.text(fast.te * 0.55, 1.0 / fast.te * 1.12, r'mean, $\int w=1$',
            fontsize=8.5, color=C_D)
    ax.set_title('(b) the onset weight $w(s)$ of (107)\n'
                 'largest at $s=0$: early forcing counts MOST',
                 fontsize=10.5)
    ax.set_xlabel('$s$ [s]', fontsize=9)
    ax.set_ylabel('$w(s)$  [1/s]', fontsize=9)
    ax.grid(alpha=0.25, lw=0.4)

    # (c) the worst admissible rho: perfect fit, maximum error
    ax = fig.add_subplot(gs[0, 2])
    flat = [r for r in rows if r[0] == 0.0][0][2]
    ax.plot(fast.t, fast.f + flat['e'], '-', color=C_D, lw=3.2,
            label='data $y$', alpha=0.55)
    ax.plot(fast.t, flat['g'], '--', color=C_G, lw=1.6, label=r'fit $\hat g$')
    ax.axvline(0.0, color=C_F, lw=1.2, ls='-')
    ax.axvline(flat['d'], color=C_G, lw=1.2, ls='--')
    zoom(ax, fast, fast.f + flat['e'], flat['g'], flat['d'], C_D, C_G, C_F)
    ax.set_title(f"(c) $\\rho\\equiv\\bar\\rho$: residual"
                 f" {np.rad2deg(flat['rms']):.3f}$^\\circ$/s,"
                 f"\nthreshold off by {1e3*abs(flat['dM']):.1f} mN.m"
                 f" = the whole budget", fontsize=10.5)
    ax.set_xlabel('$t$ [s]', fontsize=9)
    ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=9)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)

    # (d) amplitude error masquerading as onset
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(fast.t, fast.f + amp, '-', color=C_D, lw=3.2, alpha=0.55,
            label='data: $+2\\%$ amplitude')
    ax.plot(fast.t, g_amp, '--', color=C_G, lw=1.6, label=r'fit $\hat g$')
    ax.axvline(0.0, color=C_F, lw=1.2)
    ax.axvline(d_amp, color=C_G, lw=1.2, ls='--')
    zoom(ax, fast, fast.f + amp, g_amp, d_amp, C_D, C_G, C_F)
    ax.set_title(f'(d) a PURE amplitude error, no onset error\n'
                 f'fit moves the onset {1e3*d_amp:.1f} ms'
                 f' = {1e3*abs(fast.mdot*d_amp):.1f} mN.m',
                 fontsize=10.5)
    ax.set_xlabel('$t$ [s]', fontsize=9)
    ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=9)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)

    # (e) residual against bias, over the rho family
    ax = fig.add_subplot(gs[1, 1])
    for p, name, gg in rows:
        ax.plot(np.rad2deg(gg['rms']), 1e3 * abs(gg['dM']), 'o', ms=9,
                color=C_F if p >= 0 else C_G)
        ax.annotate(name, (np.rad2deg(gg['rms']), 1e3 * abs(gg['dM'])),
                    textcoords='offset points', xytext=(7, 3), fontsize=8.5)
    ax.plot(np.rad2deg(ph['rms']), 1e3 * abs(ph['dM']), 's', ms=9,
            color=C_R)
    ax.annotate('physical', (np.rad2deg(ph['rms']), 1e3 * abs(ph['dM'])),
                textcoords='offset points', xytext=(7, 3), fontsize=8.5)
    ax.axhline(1e3 * RHO_BAR, color='k', ls='--', lw=1.3)
    ax.text(0.98, 1e3 * RHO_BAR * 1.10, r'the bound $\bar\rho$',
            fontsize=8.5, ha='right', transform=ax.get_yaxis_transform())
    ax.set_yscale('log')
    ax.set_title('(e) the residual does not rank the error\n'
                 'same budget, six arrangements', fontsize=10.5)
    ax.set_xlabel(r'residual RMS [$^\circ$/s]', fontsize=9)
    ax.set_ylabel(r'$|\Delta M_{\rm crit}|$ [mN.m]', fontsize=9)
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # (f) the bound's rate dependence against the realised one
    ax = fig.add_subplot(gs[1, 2])
    lab, bnd, pert, res = [], [], [], []
    for md in (0.10, 0.20, 0.45, 0.80, 1.20):
        r = Run(md)
        gg = measure(r, physical(r))
        peak = r.c1 * (np.cosh(r.x) - 1.0)
        lab.append(f'{md:.2f}')
        bnd.append(100 / np.tanh(r.x / 2) * RHO_BAR * C2 / md)
        pert.append(100 * r.rms(gg['e']) / peak)
        res.append(100 * gg['rms'] / peak)
    i = np.arange(len(lab))
    ax.bar(i - 0.26, bnd, 0.26, color='0.62', label='(113) bound on $|e|$')
    ax.bar(i, pert, 0.26, color=C_F, label=r'$|e_\omega|$, realised')
    ax.bar(i + 0.26, res, 0.26, color=C_R, label='residual')
    ax.set_xticks(i)
    ax.set_xticklabels(lab)
    ax.set_yscale('log')
    ax.set_title(f'(f) the bound swings {bnd[0]/bnd[-1]:.0f}$\\times$ across'
                 f' the ramp rates;\nwhat actually happens swings'
                 f' {pert[0]/pert[-1]:.1f}$\\times$', fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('% of peak rate', fontsize=9)
    ax.legend(fontsize=8, loc='center left')
    ax.grid(alpha=0.25, lw=0.4, axis='y', which='both')
    print(f"\n  {'Mdot':>6}{'(113) bound %':>15}{'realised |e| %':>16}"
          f"{'residual %':>12}{'absorbed':>10}")
    for k, md in enumerate(lab):
        print(f"  {md:>6}{bnd[k]:15.2f}{pert[k]:16.4f}{res[k]:12.4f}"
              f"{100*(1-res[k]/pert[k]):9.1f}%")

    fig.suptitle('What the estimator absorbs: a good fit does not mean a '
                 'good onset', fontsize=13.5, y=0.965)
    fig.savefig(out, dpi=145)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
