"""In-window gyro-bias drift: robust slope of omega over the guaranteed
pre-onset segment (|M| < 0.3 N·m, below any plausible M_crit)."""
import contextlib, io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset

drifts, spans, deltas = [], [], []
for d in sorted(Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
    for bag in bags:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
        except Exception:
            continue
        t, om, M = sig['t'], sig['omega'], sig['moment']
        i0, i1 = cvp.detect_excitation_window(M, moment_cap=cvp.MOMENT_CAP.get(axis))
        pre = np.where(np.abs(M[i0:i1+1]) < 0.30)[0]
        if len(pre) < 20:
            continue
        seg = slice(i0, i0 + pre[-1] + 1)
        ts, os_ = t[seg] - t[seg][0], om[seg]
        # robust slope: Theil-Sen on decimated pairs
        n = len(ts); k = max(1, n // 60)
        i = np.arange(0, n, k)
        sl = np.median([(os_[b]-os_[a])/(ts[b]-ts[a])
                        for a in i for b in i if ts[b] > ts[a] + 0.2*ts[-1]])
        drifts.append(1e3*sl); spans.append(ts[-1]); deltas.append(1e3*sl*ts[-1])
    print('done', d.parent.name, d.name, flush=True)

dr, sp, de = map(np.abs, map(np.array, (drifts, spans, deltas)))
print(f"\nn={len(dr)} runs; pre-onset span [s]: median {np.median(sp):.2f}, max {np.max(sp):.2f}")
print(f"|drift slope| [mrad/s per s]: median {np.median(dr):.2f}, p90 {np.percentile(dr,90):.2f}, max {np.max(dr):.2f}")
print(f"|bias change over the pre-onset span| [mrad/s]: median {np.median(de):.2f}, p90 {np.percentile(de,90):.2f}, max {np.max(de):.2f}")
