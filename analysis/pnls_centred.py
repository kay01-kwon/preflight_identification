"""Two-stage calibration: PNLS locates the constants, the score refines them.

Stage 1 -- per-run free nonlinear least squares (the PNLS mode: C1, C2, C
all free) on every gated run of a dataset.  Each run returns C2 directly
and K = C1 / Mdot.  The per-dataset medians give a physically located
centre that owes nothing to the ramp-invariance criterion.

Stage 2 -- the ramp-invariance score of constrained_calibration is then
minimised on a grid AROUND that centre, rather than over the wide box
[3,8] x [0.05,0.70] where it is free to wander onto a search bound.

Reports the resulting constants, the rate-consistency CV, and the
delivered CoM offset with the compensation moment M_ff.
"""
import contextlib, io, sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/stubs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from constrained_calibration import prepare, score, MASS_KG, G, ROOT

TRUTH = {('case_01','Mx'):-2.90,('case_01','My'):-11.45,('case_02','Mx'):-14.29,
         ('case_02','My'):-9.90,('case_03','Mx'):-5.26,('case_03','My'):3.14,
         ('case_04','Mx'):6.67,('case_04','My'):2.40,('case_05','Mx'):10.91,
         ('case_05','My'):-10.89}
SIGN = {'Mx': +1.0, 'My': -1.0}

out = {}
print("stage 1: per-run PNLS (free C1, C2, C) -> per-dataset medians\n")
print(f"{'case':<9}{'ax':<4}{'n':>4}{'C2 median':>12}{'K median':>11}"
      f"{'C2 IQR':>16}{'K IQR':>16}")
data = {}
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    key = (d.parent.name, d.name)
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits, _ = cvp.extract_piecewise_batch(bags, axis)
    gated = {c.bag_name for c in crits}
    by_bag = {b.name: b for b in bags}
    c2s, ks = [], []
    for name in sorted(gated):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, fit = cvp.extract_piecewise(by_bag[name], axis,
                                                  model='cosh', cosh_c2=None,
                                                  ramp_gain=None)
            c1, c2, _ = fit['params']
            mdot = abs(cvp.commanded_ramp_rate(name))
            if mdot > 1e-6 and np.isfinite(c1) and np.isfinite(c2):
                c2s.append(abs(c2)); ks.append(abs(c1) / mdot)
        except Exception:
            continue
    c2s, ks = np.array(c2s), np.array(ks)
    prep = prepare(bags, axis)
    data[key] = (axis, bags, prep, float(np.median(c2s)), float(np.median(ks)))
    print(f"{key[0]:<9}{key[1]:<4}{len(c2s):4d}{np.median(c2s):12.3f}"
          f"{np.median(ks):11.4f}"
          f"{np.percentile(c2s,25):8.2f}-{np.percentile(c2s,75):<7.2f}"
          f"{np.percentile(ks,25):8.3f}-{np.percentile(ks,75):<7.3f}")

print("\nstage 2: refine the score on a grid AROUND the PNLS centre\n")
print(f"{'case':<9}{'ax':<4}{'C2_0':>7}{'K_0':>8} -> {'C2*':>7}{'K*':>8}"
      f"{'score':>9}{'J_P':>8}{'Wz':>8}")
final = {}
for key, (axis, bags, prep, c2_0, k_0) in data.items():
    grid_c2 = c2_0 * np.linspace(0.75, 1.25, 9)
    grid_k = k_0 * np.linspace(0.60, 1.40, 9)
    best = (np.inf, c2_0, k_0)
    for c2 in grid_c2:
        for k in grid_k:
            s = score(prep, float(c2), float(k))
            if s < best[0]:
                best = (s, float(c2), float(k))
    s, c2s_, ks_ = best
    final[key] = (c2s_, ks_)
    print(f"{key[0]:<9}{key[1]:<4}{c2_0:7.3f}{k_0:8.4f} -> {c2s_:7.3f}"
          f"{ks_:8.4f}{s:9.4f}{1/(ks_*c2s_**2):8.3f}{1/ks_:8.2f}")

print("\nidentification with the PNLS-centred constants:\n")
print(f"{'case':<9}{'ax':<4}{'M_ff [mN.m]':>13}{'offset':>9}{'truth':>8}{'err':>7}")
errs = {}
for key, (axis, bags, prep, _, _) in data.items():
    c2, k = final[key]
    w = MASS_KG[key[0]] * G
    with contextlib.redirect_stdout(io.StringIO()):
        cr, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2, ramp_gain=k)
    p = [c.onset_moment for c in cr if c.bag_name.startswith('pos')]
    n = [c.onset_moment for c in cr if c.bag_name.startswith('neg')]
    mff = 0.5 * (np.mean(p) + np.mean(n))
    off = SIGN[key[1]] * 1e3 * mff / w
    e = off - TRUTH[key]; errs[key] = (1e3 * mff, off, e)
    print(f"{key[0]:<9}{key[1]:<4}{1e3*mff:+13.1f}{off:+9.2f}"
          f"{TRUTH[key]:+8.2f}{e:+7.2f}")
e = np.array([v[2] for v in errs.values()])
print(f"\nPNLS-centred  RMS {np.sqrt((e**2).mean()):.2f} mm  "
      f"max|e| {np.abs(e).max():.2f}  mean {e.mean():+.2f}")
with open(Path(__file__).resolve().parent / 'pnls_centred.pkl', 'wb') as f:
    pickle.dump({'constants': final, 'errs': errs}, f)
