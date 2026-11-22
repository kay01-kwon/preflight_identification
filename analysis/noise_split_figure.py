#!/usr/bin/env python3
"""The noise estimate of (98)-(99) on one run, drawn for a slide.

  (a) time domain: the quiet stretch of the SAME length that ends where
      the excitation window opens (grey) and the post-onset fit window
      (blue), with the fit residual r and the quiet record q underneath
  (b) frequency domain: amplitude spectra of r and q with the model's
      spectral home C2/2pi and the brick-wall split f_c = 5 Hz; above
      f_c the residual and the quiet record coincide (pure disturbance),
      below f_c the residual carries model error as well, so the low
      band is priced from q only, as the ratio kappa_q

The residual r and the split are the ones of analysis/failing_runs.py
(amplitude channel at the calibrated C2*, rfft brick wall), so the
numbers printed are the ones the campaign constants are built from.

Usage: python analysis/noise_split_figure.py [case] [axis] [run] [out.png]
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from constrained_calibration import ROOT
from failing_runs import FC, amplitude_best, split
from pnls_constants import PNLS_CONSTANTS
from utils.extractor import load_excitation_dataset

C_R, C_Q, C_Y = '#c0392b', '0.45', '#2874a6'


def aspec(v, dt, npad=4096):
    """One-sided amplitude spectrum [deg/s], Hann-tapered and padded for
    display; the numbers use the raw bins (split)."""
    vv = np.degrees(np.asarray(v, float))
    vv = vv - vv.mean()
    w = np.hanning(len(vv))
    f = np.fft.rfftfreq(npad, d=dt)
    a = np.abs(np.fft.rfft(vv * w, n=npad)) * 2.0 / w.sum()
    k = max(3, npad // len(vv)) | 1
    a = np.convolve(a, np.ones(k) / k, mode='same')
    return f[1:], a[1:]


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_05'
    axis_dir = sys.argv[2] if len(sys.argv) > 2 else 'My'
    run = sys.argv[3] if len(sys.argv) > 3 else 'pos_My_065'
    out = sys.argv[4] if len(sys.argv) > 4 else 'docs/exp_noise_split.png'
    axis = 'x' if axis_dir == 'Mx' else 'y'
    c2, k = PNLS_CONSTANTS[(case, axis_dir)]
    deg = np.degrees
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))

    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        bag = next(b for b in bags if b.name == run)
        sig = cvp.prepare_signals(bag, axis)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    w = slice(i0, i1 + 1)
    t, om, mom = sig['t'][w], sig['omega'][w], sig['moment'][w]
    md = float(np.polyfit(t, mom, 1)[0])
    pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t), onset_guess=None,
                            c2_fixed=c2, moment_floor=0.0, ramp_gain=k, ramp_rate=md)
    j = pw['onset_idx']
    tau = t[j:] - t[j]
    dt = float(np.median(np.diff(tau)))
    npost = len(tau)
    _, r = amplitude_best(tau, om[j:], c2)           # fit residual, post-onset
    q_all, tq_all = sig['omega'][:i0], sig['t'][:i0]
    q, tq = q_all[-npost:], tq_all[-npost:]          # quiet stretch, same length
    r_lo, r_hi = split(r, dt)
    q_lo, q_hi = split(q - q.mean(), dt)
    kq = rms(q_lo) / rms(q_hi)
    n_hi = rms(r_hi)
    n_run = n_hi * np.sqrt(1 + kq ** 2)
    fm = c2 / (2 * np.pi)
    t0 = t[j]

    print(f'{case}/{axis_dir} {run}: onset t_c = {t0:.3f} s, window {tau[-1]:.2f} s, '
          f'{npost} samples, dt = {dt * 1e3:.1f} ms, df = {1 / tau[-1]:.2f} Hz')
    print(f'  C2*/2pi = {fm:.2f} Hz,  f_c = {FC:.0f} Hz')
    print(f'  residual r : RMS {deg(rms(r)):.3f}  lo {deg(rms(r_lo)):.3f}  hi {deg(n_hi):.3f} deg/s')
    print(f'  quiet q    : RMS {deg(rms(q - q.mean())):.3f}  lo {deg(rms(q_lo)):.3f}  '
          f'hi {deg(rms(q_hi)):.3f} deg/s  ->  kappa_q = {kq:.2f}')
    print(f'  this run   : RMS(n) = {deg(n_hi):.3f} * sqrt(1 + {kq:.2f}^2) = {deg(n_run):.3f} deg/s')

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.55], hspace=0.42, wspace=0.34)
    A = fig.add_subplot(gs[0, 0])
    B = fig.add_subplot(gs[0, 1])
    Q = fig.add_subplot(gs[0, 2], sharey=B)
    S = fig.add_subplot(gs[1, :])

    # ── (a) the fit window: samples and the calibrated-family fit ──
    A.plot(tau, deg(om[j:]), '.', ms=3, color='0.5', zorder=1, label=r'$y_i$')
    A.plot(tau, deg(pw['omega_pred'][j:]), '-', lw=1.6, color=C_R, zorder=3, label=r'$\hat h$')
    A.set_title('(a) fit window, %d samples' % npost, fontsize=8.5)
    A.set_xlabel(r'$\tau$ [s]', fontsize=8.5)
    A.set_ylabel(r'$\omega$ [deg/s]', fontsize=8.5)
    A.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

    # ── (b) the residual, (c) the quiet stretch: same length, same scale ──
    B.plot(tau, deg(r), '-', lw=1.0, color=C_R)
    B.axhline(0, lw=0.5, color='0.6')
    B.set_title(r'(b) residual $r = y - \hat h$', fontsize=8.5)
    B.set_xlabel(r'$\tau$ [s]', fontsize=8.5)
    B.set_ylabel('[deg/s]', fontsize=8.5)
    B.text(0.97, 0.05, 'RMS %.2f' % deg(rms(r)), transform=B.transAxes, fontsize=7.5,
           ha='right', va='bottom', color=C_R)
    Q.plot(tq - tq[0], deg(q - q.mean()), '-', lw=1.0, color=C_Q)
    Q.axhline(0, lw=0.5, color='0.6')
    Q.set_title('(c) quiet stretch $q$, same length', fontsize=8.5)
    Q.set_xlabel('[s]  (before the excitation window)', fontsize=8.5)
    Q.text(0.97, 0.05, 'RMS %.2f' % deg(rms(q - q.mean())), transform=Q.transAxes, fontsize=7.5,
           ha='right', va='bottom', color='0.3')
    lim = 1.15 * deg(np.abs(r)).max()
    B.set_ylim(-lim, lim)
    Q.tick_params(labelleft=False)

    # ── (d) spectra and the split ─────────────────────────────────
    fr, ar = aspec(r, dt)
    fq, aq = aspec(q, dt)
    S.loglog(fr, ar, color=C_R, lw=1.6, label='residual $r$ (fit window)')
    S.loglog(fq, aq, color=C_Q, lw=1.4, ls='--', label='quiet stretch $q$')
    fmax = fr.max()
    S.axvspan(FC, fmax, color='0.92', lw=0, zorder=0)
    S.axvline(fm, color=C_Y, lw=1.0, ls=':')
    S.axvline(FC, color='k', lw=1.2, ls='-.')
    ymax = max(ar.max(), aq.max()) * 3.2
    ymin = ymax * 1e-3
    S.text(fm * 1.07, ymax * 0.55, r'$C_2^{*}/2\pi$ = %.2f Hz' % fm, fontsize=8.5, color=C_Y)
    S.text(FC * 0.94, ymax * 0.30, r'$f_c$ = 5 Hz', fontsize=8.5, ha='right')
    S.text(0.985, 0.96, r'high band $f > f_c$: $r_{\rm hi} \to n_{\rm hi}$'
           '\nno model content here; read from $r$ itself',
           transform=S.transAxes, ha='right', va='top', fontsize=8.5, color='0.25')
    S.text(0.015, 0.96, 'low band $f < f_c$: model error + $n_{\\rm lo}$ mixed'
           '\n$\\to$ only the shape ratio $\\kappa_q$, taken from $q$',
           transform=S.transAxes, ha='left', va='top', fontsize=8.5, color=C_R)
    S.set_xlim(fr.min(), fmax)
    S.set_ylim(ymin, ymax)
    S.set_xlabel('frequency [Hz]', fontsize=9)
    S.set_ylabel('amplitude [deg/s]', fontsize=9)
    S.set_title('(d) amplitude spectra and the brick-wall split', fontsize=8.5)
    S.legend(fontsize=8, loc='lower left', framealpha=0.9)
    S.text(0.985, 0.05,
           '\n'.join((rf'$\mathrm{{RMS}}(r_{{\rm hi}}) = {deg(n_hi):.2f}$ °/s',
                      rf'$\kappa_q = \mathrm{{RMS}}(q_{{\rm lo}})/\mathrm{{RMS}}(q_{{\rm hi}}) = {kq:.2f}$',
                      rf'this run: $\mathrm{{RMS}}(n) = {deg(n_hi):.2f}\sqrt{{1+{kq:.2f}^2}} = {deg(n_run):.2f}$ °/s')),
           transform=S.transAxes, fontsize=8, ha='right', va='bottom',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    for ax in (A, B, Q, S):
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.4, which='both')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    main()
