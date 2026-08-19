#!/usr/bin/env python3
"""The spectral construction behind the noise estimate (19), drawn.

Panel (a): one run's amplitude spectra -- the windowed record, the
minimiser residual, and a quiet stretch of the same length -- with the
model's spectral home C2/2pi and the 5 Hz split marked.  Everything the
branch can draw lives below ~1 Hz; above 5 Hz the residual and the
quiet record coincide in shape (pure disturbance); between the two the
residual sits above the quiet spectrum by the in-window enhancement.

Panel (b): how the two constants of (19) are measured, across the
campaign.  For every run with a usable quiet stretch, the quiet shape
ratio kappa_q = RMS(<5 Hz)/RMS(>5 Hz) against the in-window implied
ratio kappa_imp = sqrt(RMS(r)^2 - RMS(n_hi)^2)/RMS(n_hi); the line is
the campaign-median enhancement s_med = median(kappa_imp/kappa_q).

The split is a brick-wall FFT partition (rfft, zero the bins above
f_c, irfft), so the two bands are exactly orthogonal and the
quadrature sum of (19) is Parseval-clean.

Usage: python analysis/noise_spectrum_figure.py [out.png]
"""
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failing_runs import split, FC
from rms_check import measure, amplitude_best
from tight_rms_bound import enrich

HERE = os.path.dirname(os.path.abspath(__file__))
C_Y, C_R, C_Q = '#2874a6', '#c0392b', '0.45'


def aspec(v, dt, npad=4096):
    """One-sided amplitude spectrum in deg/s, Hann-tapered and
    zero-padded FOR DISPLAY ONLY: the taper suppresses the edge-
    truncation leakage that would otherwise smear the smooth branch
    across the whole band, and the padding interpolates the DTFT.
    All NUMBERS in (19) use the raw unpadded bins of the residual,
    where the branch has been subtracted and edge leakage cancels."""
    vv = np.rad2deg(np.asarray(v, float))
    vv = vv - vv.mean()
    w = np.hanning(len(vv))
    f = np.fft.rfftfreq(npad, d=dt)
    a = np.abs(np.fft.rfft(vv * w, n=npad)) * 2.0 / w.sum()
    k = max(3, npad // len(vv)) | 1
    a = np.convolve(a, np.ones(k) / k, mode='same')
    return f[1:], a[1:]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'noise_spectrum.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))

    # a representative run: mid rate, quiet stretch available
    cand = [d for d in rows if d['kq'] is not None and d['rate'] == 0.30]
    d = sorted(cand, key=lambda r: abs(r['kq'] - 0.8))[0]
    _, rf = amplitude_best(d['tau'], d['om'], d['c2'])
    q = np.asarray(d['quiet'], float)
    fm = d['c2'] / (2.0 * np.pi)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.13,
                        wspace=0.24)

    om = np.asarray(d['om'], float)
    fy, ay = aspec(om, d['dt'])
    fmld, amld = aspec(om - rf, d['dt'])
    fr, ar = aspec(rf, d['dt'])
    fq, aq = aspec(q, d['dt'])
    a1.loglog(fy, ay, color=C_Y, lw=1.6, label='windowed record $y$')
    a1.loglog(fmld, amld, color='#e08214', lw=1.6,
              label='fitted branch (the model)')
    a1.loglog(fr, ar, color=C_R, lw=1.6, label='minimiser residual $r$')
    a1.loglog(fq, aq, color=C_Q, lw=1.4, ls='--',
              label='quiet stretch (same length)')
    a1.axvline(fm, color=C_Y, lw=1.0, ls=':', alpha=0.9)
    a1.axvline(FC, color='k', lw=1.2, ls='-.')
    ymax = max(ay.max(), ar.max()) * 1.8
    a1.text(fm * 1.06, ymax * 0.5, r'$C_2/2\pi$', fontsize=9, color=C_Y)
    a1.text(FC * 1.06, ymax * 0.5, r'$f_c = 5$ Hz', fontsize=9)
    a1.axvspan(FC, fy.max(), color='0.9', alpha=0.5, zorder=0)
    a1.text(np.sqrt(FC * fy.max()), ymax * 0.16,
            r'$n_{\rm hi}$: model-free band', ha='center', fontsize=9)
    a1.set_xlabel('frequency [Hz]', fontsize=10)
    a1.set_ylabel(r'amplitude spectrum [$^\circ$/s]', fontsize=10)
    a1.set_title(f'(a) one run ({d["rate"]:.2f} N m/s, {d["axis"]}): '
                 'where everything lives\nbrick-wall FFT split at '
                 f'{FC:.0f} Hz; model content at $C_2/2\\pi$ = '
                 f'{fm:.2f} Hz', fontsize=10.5)
    a1.legend(fontsize=8.5, loc='lower left')
    a1.grid(alpha=0.2, lw=0.4, which='both')

    kq = np.array([r_['kq'] for r_ in rows if r_['kq'] is not None])
    ki = np.array([r_['kimp'] for r_ in rows if r_['kq'] is not None])
    a2.scatter(kq, ki, s=22, c='#7b3294', alpha=0.6, lw=0)
    xg = np.linspace(0, kq.max() * 1.05, 50)
    a2.plot(xg, s_med * xg, 'k-', lw=1.4,
            label=f'$s_{{\\rm med}} = {s_med:.2f}$ (campaign median)')
    a2.plot(xg, xg, color='0.6', lw=1.0, ls=':', label='no enhancement')
    a2.set_xlabel(r'quiet shape ratio $\kappa_q$', fontsize=10)
    a2.set_ylabel(r'in-window implied ratio $\kappa_{\rm imp}$',
                  fontsize=10)
    a2.set_title('(b) the two constants of (19), measured\n'
                 f'{len(kq)} runs with a usable quiet stretch; '
                 'in-window disturbance is redder', fontsize=10.5)
    a2.legend(fontsize=9, loc='upper left')
    a2.grid(alpha=0.2, lw=0.4)

    fig.suptitle('The spectral construction behind the noise estimate '
                 '(19)', fontsize=12, y=0.97)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}   example run: rate {d['rate']}, axis {d['axis']}, "
          f"kq {d['kq']:.2f}, kimp {d['kimp']:.2f}; "
          f"s_med {s_med:.2f}, runs with quiet {len(kq)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
