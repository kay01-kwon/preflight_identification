#!/usr/bin/env python3
"""Where the 5 Hz cutoff comes from: sweep f_c under the deployed
SG-based noise constant N_n = 3 N_med.

Two constraints close in from opposite sides.  From below, f_c must
clear the model's spectral home C2/2pi = 0.56-0.90 Hz, or the high
band inherits model content and the anchor stops being model-free;
and the low band must keep bins of its own (spacing 0.97-4.0 Hz, so
the shortest windows have none below 1-2 Hz).  From above, the
disturbance is coloured, so a high cutoff anchors on its quietest
corner -- the same failure mode as the textbook estimators.

Result: 2-6 Hz is a plateau (N_med 0.77-0.82, 140/140 throughout);
past 8 Hz the anchor reads progressively low and coverage breaks
(136/140 at 10 Hz, 117/140 at 15).

Usage: python analysis/cutoff_sweep.py
"""
import os, pickle, sys
import numpy as np
from scipy.signal import savgol_filter
sys.path.insert(0, '/home/user/preflight_identification/analysis')
from failing_runs import split
from fit_quality_bound import rho_bar
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from kernel_free_bound import model_term

HERE = '/home/user/preflight_identification/analysis'
with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
    rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))

hi = lambda v: float(np.sqrt(np.mean(v**2)))
for d in rows:
    rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
    de, dpre = model_term(d, rb)
    d['m'] = de + dpre

# bin spacing and model home, for the two constraints
df = [1.0/(len(d['om'])*d['dt']) for d in rows]
fm = [d['c2']/(2*np.pi) for d in rows]
print(f"model home C2/2pi : {min(fm):.2f}-{max(fm):.2f} Hz")
print(f"bin spacing 1/T   : {min(df):.2f}-{max(df):.2f} Hz")
print(f"Nyquist           : {1/(2*max(d['dt'] for d in rows)):.0f} Hz\n")

print(f"{'fc':>5}{'lo bins':>9}{'hi bins':>9}{'N_med':>9}{'N_n':>8}"
      f"{'cover':>9}{'mean used':>11}")
for fc in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]:
    sgs, nlo, nhi = [], [], []
    for d in rows:
        om = np.asarray(d['om'], float); dt, N = d['dt'], len(om)
        w = max(int(round(2.0/(fc*dt))) | 1, 7)
        w = min(w, N-1 if (N-1) % 2 else N-2)
        sgs.append(hi(split(om - savgol_filter(om, w, 3), dt, fc)[1]))
        f = np.fft.rfftfreq(N, d=dt)
        nlo.append(int(((f > 0) & (f <= fc)).sum())); nhi.append(int((f > fc).sum()))
    Nn = 3.0*float(np.rad2deg(np.median(sgs)))
    ins = sum(1 for d in rows if d['rms_min'] <= d['m'] + Nn)
    used = np.mean([d['rms_min']/(d['m'] + Nn) for d in rows])
    print(f"{fc:5.0f}{min(nlo):5d}-{max(nlo):<4d}{min(nhi):5d}-{max(nhi):<4d}"
          f"{np.rad2deg(np.median(sgs)):9.3f}{Nn:8.2f}{ins:6d}/140{used:11.3f}")
