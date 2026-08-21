import os, pickle, sys
import numpy as np
sys.path.insert(0, '/home/user/preflight_identification/analysis')
from fit_quality_bound import rho_bar
from rms_check import measure, PHI_BOX
from tight_rms_bound import enrich
from kernel_free_bound import model_term

HERE = '/home/user/preflight_identification/analysis'
with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
    rows, kqmax, s_med, kb = enrich(measure(pickle.load(fh)))

N_n = 2.315
print(f"{'Mdot':>6}{'GE share of rho_bar':>21}{'cap now':>9}"
      f"{'cap @ 2x GE':>13}{'change':>9}")
for rt in sorted({d['rate'] for d in rows}):
    v = [d for d in rows if d['rate'] == rt]
    shares, caps, caps2 = [], [], []
    for d in v:
        rb, r_grav, r_ge = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        shares.append(r_ge / rb)
        de, dpre = model_term(d, rb)
        caps.append(de + dpre + N_n)
        de2, dpre2 = model_term(d, rb + abs(r_ge))   # double the GE channel
        caps2.append(de2 + dpre2 + N_n)
    print(f"{rt:6.2f}{100*np.mean(shares):20.1f}%{np.mean(caps):9.2f}"
          f"{np.mean(caps2):13.2f}{100*(np.mean(caps2)/np.mean(caps)-1):8.2f}%")
