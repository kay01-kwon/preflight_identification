#!/usr/bin/env python3
"""The noise estimate of (98)-(99), drawn for a slide: one run and the
campaign constants.

  (a) the raw FFT bins the numbers use: the fit residual r (dark red
      below f_c, light red above = n_hi) and the quiet stretch q of the
      same length (grey); no taper, no padding, Delta f = 1/T
  (b) amplitude spectra of the same run for display: the windowed
      record y, the fitted branch (the model), the residual r and the
      quiet stretch q, with C2*/2pi and the 5 Hz brick wall
  (c) the two campaign constants: per-run RMS(n_hi) against the quiet
      shape ratio kappa_q over the runs with a quiet stretch; the
      maxima give kappa_b and the single noise term of (96)

The residual r and the split are the ones of analysis/failing_runs.py
(amplitude channel at the calibrated C2*, rfft brick wall); the
campaign values come from its cache through rms_check.measure and
tight_rms_bound.enrich, so they are the numbers of the paper.

Usage: python analysis/noise_split_figure.py [case] [axis] [run] [out.png]
"""
import contextlib
import csv
import io
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, 'trapz'):                       # numpy >= 2
    np.trapz = np.trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import critical_value_getter_piecewise as cvp
from constrained_calibration import ROOT
from failing_runs import CACHE, FC, amplitude_best, split
from pnls_constants import PNLS_CONSTANTS
from rms_check import measure
from tight_rms_bound import enrich
from utils.extractor import load_excitation_dataset

C_R, C_RL, C_Q, C_Y, C_M = '#c0392b', '#e8a09a', '0.45', '#2874a6', '#e08214'


def aspec(v, dt, npad=4096):
    """One-sided amplitude spectrum [deg/s], Hann-tapered and padded for
    display only; the numbers use the raw bins."""
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

    # ── one run, exactly as analysis/hw_envelope_witness.py does it:
    #    free PNLS fit, residual over the WHOLE excitation window ───
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(ROOT / case / axis_dir)
        bag = next(b for b in bags if b.name == run)
        crit, pw = cvp.extract_piecewise(bag, axis, model='cosh', cosh_c2=None,
                                         ramp_gain=None, free_seed=False)
        sig = cvp.prepare_signals(bag, axis)
    rate = cvp.commanded_ramp_rate(bag.name)
    i0, i1 = cvp.detect_excitation_window(
        sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
    om_w, pred = sig['omega'][i0:i1 + 1], pw['omega_pred']
    n = min(len(om_w), len(pred))
    y_win, branch = om_w[:n], pred[:n]
    r = y_win - branch
    dt = float(np.median(np.diff(sig['t'][i0:i0 + n])))
    q = sig['omega'][:i0][-n:]                       # quiet stretch before the window (<= same length)
    r_lo, r_hi = split(r, dt)
    q_lo, q_hi = split(q - q.mean(), dt)
    n_hi = rms(r_hi)
    fm = c2 / (2 * np.pi)
    md = float(np.polyfit(sig['t'][i0:i0 + n], sig['moment'][i0:i0 + n], 1)[0])

    # ── the campaign: kappa_q per run from the quiet stretches
    #    (failing_runs cache, tight_rms_bound.enrich) and n_hi per run
    #    from docs/hw_env_noise_runs.csv (hw_envelope_witness.py) ──
    with open(CACHE, 'rb') as fh:
        rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))
    with open(os.path.join(os.path.dirname(HERE), 'docs', 'hw_env_noise_runs.csv')) as fh:
        csv_rows = list(csv.DictReader(fh))
    # both files walk the datasets in the same (sorted) order; the rate
    # alone is not a key because +/- runs share it, so pair by position
    # and check the labels agree
    assert len(csv_rows) == len(rows)
    for c, d in zip(csv_rows, rows):
        assert (c['case'], c['ax']) == (d['case'], d['axis']) and \
            abs(float(c['rate']) - d['rate']) < 1e-9, (c, d['case'], d['axis'], d['rate'])
    nhi_all = np.array([float(c['nhi']) for c in csv_rows])
    kq_all = np.array([d['kq'] for d in rows if d['kq'] is not None])
    nhi_q = np.array([float(c['nhi']) for c, d in zip(csv_rows, rows) if d['kq'] is not None])
    me = next(i for i, b_ in enumerate(bags) if b_.name == run)
    d_me = next(d for d, c in zip(rows, csv_rows)
                if (d['case'], d['axis']) == (case, axis_dir)
                and abs(d['rate'] - rate) < 1e-9 and d['sign'] == int(np.sign(md)))
    kq = d_me['kq']
    n_term = nhi_all.max() * np.sqrt(1 + kb ** 2)
    n_run = deg(n_hi) * np.sqrt(1 + kq ** 2)

    print(f'{case}/{axis_dir} {run}: full window {n} samples, dt = {dt * 1e3:.1f} ms, '
          f'df = {1 / (n * dt):.2f} Hz, C2*/2pi = {fm:.2f} Hz, Mdot = {md:+.3f}')
    print(f'  residual r : RMS {deg(rms(r)):.3f}  lo {deg(rms(r_lo)):.3f}  hi {deg(n_hi):.3f} deg/s')
    print(f'  quiet q    : RMS {deg(rms(q - q.mean())):.3f}  lo {deg(rms(q_lo)):.3f}  '
          f'hi {deg(rms(q_hi)):.3f} deg/s;  kappa_q (cache) = {kq:.2f}')
    print(f'  this run   : RMS(n) = {deg(n_hi):.2f} * sqrt(1 + {kq:.2f}^2) = {n_run:.2f} deg/s')
    print(f'  campaign   : {len(csv_rows)} runs, {len(kq_all)} with a quiet stretch')
    print(f'    RMS(n_hi): median {np.median(nhi_all):.2f}, 95% {np.percentile(nhi_all, 95):.2f}, '
          f'max {nhi_all.max():.2f} deg/s;  kappa_b = max kappa_q = {kb:.2f}')
    print(f'    noise term = {nhi_all.max():.2f} * sqrt(1 + {kb:.2f}^2) = {n_term:.2f} deg/s')

    fig, (A, B, C) = plt.subplots(1, 3, figsize=(13.2, 3.9))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.15, wspace=0.27)

    # (a) raw bins
    vv, qq = deg(r - r.mean()), deg(q - q.mean())
    fb = np.fft.rfftfreq(n, d=dt)
    ab = np.abs(np.fft.rfft(vv)) * 2.0 / n
    fbq = np.fft.rfftfreq(len(qq), d=dt)          # the quiet stretch can be shorter
    aq = np.abs(np.fft.rfft(qq)) * 2.0 / len(qq)
    df = fb[1] - fb[0]
    lo = fb <= FC
    A.bar(fb[lo & (fb > 0)] - 0.18 * df, ab[lo & (fb > 0)], 0.32 * df, color=C_R,
          label='residual bins, low band')
    A.bar(fb[~lo] - 0.18 * df, ab[~lo], 0.32 * df, color=C_RL,
          label=r'residual bins, high band ($n_{\rm hi}$)')
    A.bar(fbq[1:] + 0.18 * df, aq[1:], 0.32 * df, color=C_Q, alpha=0.75,
          label=f'quiet-stretch bins ({len(qq)} samples)')
    A.axvline(FC, color='k', lw=1.2, ls='-.')
    A.set_yscale('log')
    A.set_xlim(0, min(30.0, fb.max()))
    A.text(FC + 0.4, ab.max() * 0.8, r'$f_c$', fontsize=9)
    nlo = int(((fb > 0) & lo).sum())
    A.set_title(f'(a) the raw FFT bins the numbers use\n'
                rf'$\Delta f = 1/T = {df:.2f}$ Hz; {nlo} bins below $f_c$; no taper, no padding',
                fontsize=10)
    A.set_xlabel('frequency [Hz]', fontsize=10)
    A.set_ylabel(r'bin amplitude [$^\circ$/s]', fontsize=10)
    A.legend(fontsize=8, loc='upper right')
    A.grid(alpha=0.2, lw=0.4, axis='y')

    # (b) spectra
    fy, ay = aspec(y_win, dt)
    fmd, amd = aspec(branch, dt)
    fr, ar = aspec(r, dt)
    fq, aq2 = aspec(q, dt)
    B.loglog(fy, ay, color=C_Y, lw=1.6, label='windowed record $y$')
    B.loglog(fmd, amd, color=C_M, lw=1.6, label='free PNLS fit (the model)')
    B.loglog(fr, ar, color=C_R, lw=1.6, label='fit residual $r$ (whole window)')
    B.loglog(fq, aq2, color=C_Q, lw=1.4, ls='--', label='quiet stretch (before the window)')
    B.axvline(fm, color=C_Y, lw=1.0, ls=':', alpha=0.9)
    B.axvline(FC, color='k', lw=1.2, ls='-.')
    ymax = max(ay.max(), ar.max()) * 1.8
    B.axvspan(FC, fy.max(), color='0.9', alpha=0.5, zorder=0)
    B.text(fm * 1.06, ymax * 0.5, r'$C_2^{*}/2\pi$', fontsize=9, color=C_Y)
    B.text(FC * 1.06, ymax * 0.5, r'$f_c = 5$ Hz', fontsize=9)
    B.text(np.sqrt(FC * fy.max()), ymax * 2e-5 * 2.5, r'$n_{\rm hi}$: model-free band',
           ha='center', fontsize=9)
    B.set_ylim(ymax * 2e-5, ymax)
    B.set_xlabel('frequency [Hz]', fontsize=10)
    B.set_ylabel(r'amplitude spectrum [$^\circ$/s]', fontsize=10)
    B.set_title(f'(b) one run ({case}/{axis_dir}, {abs(md):.2f} N m/s): where everything lives\n'
                rf'brick-wall split at {FC:.0f} Hz; model content at $C_2^{{*}}/2\pi$ = {fm:.2f} Hz',
                fontsize=10)
    B.legend(fontsize=8, loc='lower left')
    B.grid(alpha=0.2, lw=0.4, which='both')

    # (c) the constants
    C.scatter(kq_all, nhi_q, s=22, c='#7b3294', alpha=0.6, lw=0,
              label=f'{len(kq_all)} runs with a quiet stretch')
    C.scatter([kq], [deg(n_hi)], s=70, marker='o', fc='none', ec=C_R, lw=1.6, zorder=5,
              label='this run')
    C.axvline(kb, color='k', lw=1.1, ls='--')
    C.axhline(nhi_all.max(), color='k', lw=1.1, ls='--')
    C.text(kb, 0.02, rf' $\kappa_b = \max\kappa_q = {kb:.2f}$', transform=C.get_xaxis_transform(),
           fontsize=9, ha='left', va='bottom')
    C.text(0.02, nhi_all.max(), rf'$\max\,\mathrm{{RMS}}(n_{{\rm hi}}) = {nhi_all.max():.2f}$ °/s '
           f'(all {len(csv_rows)} runs) ', transform=C.get_yaxis_transform(), fontsize=9,
           ha='left', va='bottom')
    C.text(0.98, 0.97,
           rf'$\mathrm{{RMS}}(n) = {nhi_all.max():.2f}\sqrt{{1+{kb:.2f}^2}} = {n_term:.2f}$ °/s'
           '\none additive term for every run (96)',
           transform=C.transAxes, fontsize=9, ha='right', va='top',
           bbox=dict(fc='white', ec='0.8', lw=0.6, pad=3))
    C.set_xlim(0, max(kb * 1.15, kq_all.max() * 1.05))
    C.set_ylim(0, nhi_all.max() * 1.38)
    C.set_xlabel(r'quiet shape ratio $\kappa_q = \mathrm{RMS}(q_{\rm lo})/\mathrm{RMS}(q_{\rm hi})$',
                 fontsize=10)
    C.set_ylabel(r'in-window $\mathrm{RMS}(n_{\rm hi})$ [$^\circ$/s]', fontsize=10)
    C.set_title('(c) the two constants of (98)–(99), measured per run\n'
                'the maxima over the campaign fix a single noise term', fontsize=10)
    C.legend(fontsize=8, loc='lower left')
    C.grid(alpha=0.2, lw=0.4)
    for ax in (A, B, C):
        ax.tick_params(labelsize=8.5)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print('written', out)


if __name__ == '__main__':
    main()
