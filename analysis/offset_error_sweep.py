#!/usr/bin/env python3
"""Eq. (120) drawn: the CoM-offset error under static ground effect,
swept over the true offset p_off in [-20, +20] mm.

Per direction the GE balance gives the identified moment
    M_s = [ s(W-f) l - s a f l + W p ] / (1 + a),        s = +-1,
and the no-GE reading of the offset from one direction, or from the
pair average, then carries the errors

    per direction :  dp_s = [ a M_s + s a f l_s ] / W
    pair average  :  dp   = a p/(1+a) + a f (l+ - l-) / (2W)

-- the paired error is a 4.13% multiplicative line through the
origin plus a sub-0.2 mm arm-asymmetry constant, while each single
direction is offset by ~5 mm.  The campaign's ten configurations are
overlaid as points (truth offsets from the load-cell table, channel
totals from mcrit_prediction.csv via docs/ge_offset_shift.tex).

Usage: python analysis/offset_error_sweep.py [out.png]
"""
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = 0.0431                    # interference gains at phi* = 0
M_KG, G = 3.220, 9.81
W = M_KG * G
F = 0.7 * W                   # collective at onset
L = 0.140                     # roll pivot arm [m]
DL = 0.008                    # observed arm asymmetry bound (Mx) [m]

# campaign points: (signed truth offset p [mm], paired GE error [mm])
# from docs/ge_offset_shift.tex Sec. 5 (S_off sign convention)
CAMPAIGN = [(-2.90, -0.167), (11.45, 0.636), (-14.29, -0.659),
            (9.90, 0.644), (-5.26, -0.430), (-3.14, 0.093),
            (6.67, 0.185), (-2.40, 0.158), (10.91, 0.395),
            (10.89, 0.669)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'offset_error_sweep.png'
    p = np.linspace(-20, 20, 201) * 1e-3          # true offset [m]

    def m_s(s, l):
        return (s*(W-F)*l - s*A*F*l + W*p) / (1.0 + A)

    dp_pos = (A*m_s(+1, L) + A*F*L) / W * 1e3     # [mm]
    dp_neg = (A*m_s(-1, L) - A*F*L) / W * 1e3
    dp_avg = 0.5*(dp_pos + dp_neg)
    band = A*F*DL/(2*W) * 1e3                     # arm-asymmetry constant

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axhspan(-1.64, 1.64, color='0.92', lw=0, zorder=0,
               label='load-cell validation RMS $\\pm 1.64$ mm')
    ax.plot(1e3*p, dp_pos, '--', color='#c0392b', lw=1.4,
            label='single direction, $+$ tip')
    ax.plot(1e3*p, dp_neg, '--', color='#1a5276', lw=1.4,
            label='single direction, $-$ tip')
    ax.fill_between(1e3*p, dp_avg-band, dp_avg+band, color='#148f77',
                    alpha=0.25, lw=0)
    ax.plot(1e3*p, dp_avg, '-', color='#148f77', lw=2.2,
            label=r'pair average: slope $\alpha/(1+\alpha) = 4.13\%$'
                  '\n' r'$\pm$ arm-asymmetry band ($|\Delta l_p|'
                  r'\leq 8$ mm)')
    cx = [c[0] for c in CAMPAIGN]; cy = [c[1] for c in CAMPAIGN]
    ax.plot(cx, cy, 'o', color='k', ms=6, mfc='none', mew=1.4,
            label='campaign, 10 configurations (paired)')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel(r'true CoM offset $p_{\rm off}$ [mm]', fontsize=11)
    ax.set_ylabel(r'offset reading error under GE [mm]', fontsize=11)
    ax.set_title('The ground-effect error of the identified CoM offset\n'
                 'single directions carry a $\\pm$5 mm bias; the pair '
                 'average keeps 4.13% of $p_{\\rm off}$', fontsize=11.5)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(-20, 20)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  slopes: single {1e3*(dp_pos[-1]-dp_pos[0])/40:.4f} mm/mm, "
          f"paired {1e3*(dp_avg[-1]-dp_avg[0])/40:.4f} mm/mm "
          f"(= a/(1+a) = {A/(1+A):.4f})")
    print(f"  single-direction offsets at p=0: {dp_pos[100]:+.2f} / "
          f"{dp_neg[100]:+.2f} mm;  paired band +-{band:.2f} mm")
    print(f"  paired error at p = +-20 mm: {dp_avg[-1]:+.2f} / "
          f"{dp_avg[0]:+.2f} mm")
    print(f"  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
