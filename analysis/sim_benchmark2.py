"""Detector benchmark on the simulation campaign, statistics edition.

Same estimators and gated population as analysis/sim_benchmark.py,
with the theoretical critical moments evaluated exactly as the
manuscript's (7) and (14),

    M_crit,+ = +(W - f) l_+ + W s_off,
    M_crit,- = -(W - f) l_- + W s_off        (signs per axis),

with f the measured collective thrust at each method's own detected
onset. Emits per-run signed errors and the summary table with
empirical 95% intervals for both the critical-moment error and the
per-cell offset error (docs/tab_sim_benchmark.tex).
"""
import contextlib, csv, io, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset
from analysis.pelt_crosscheck import classic_onset_index, _window, CLASSIC

G = 9.81
Z_CAD = 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
# per-direction pivot lever arms of (7) and (14): equal on this frame
L_ARM = {'Mx': 0.110, 'My': 0.140}
CASES = {'S1':(-6.0,0.0,3.066),'S2':(0.0,10.0,3.066),'S3':(10.0,-5.0,3.066),
         'S4':(20.0,20.0,3.066),'S5':(20.0,-20.0,3.066),
         'S6':(-20.0,20.0,3.066),'S7':(-20.0,-20.0,3.066),
         'S8':(25.0,25.0,3.066),'S9':(32.0,32.0,3.066),
         'S11':(38.0,14.0,3.066),'S13':(25.0,25.0,3.220)}
SIGN = {'Mx': +1.0, 'My': -1.0}
METHODS = ['cosh', 'cosh_cad', 'nls'] + CLASSIC
LABEL = {'cosh': 'COSH', 'cosh_cad': 'COSH (CAD)', 'nls': 'PNLS',
         'pelt_normal': 'CPD (normal)', 'pelt_rbf': 'CPD (RBF)',
         'cusum': 'CUSUM'}


def m_theory(simax, direc, W, f, s_off_m):
    """Signed (7)/(14) with the measured onset collective f."""
    l = L_ARM[simax]
    off = W * s_off_m * SIGN[simax] * (+1 if simax == 'Mx' else +1)
    # (7):  Mx,+ = +(W-f) l + W y_off ; Mx,- = -(W-f) l + W y_off
    # (14): My,+ = +(W-f) l - W x_off ; My,- = -(W-f) l - W x_off
    core = (W - f) * l * (+1.0 if direc == 'pos' else -1.0)
    if simax == 'Mx':
        return core + W * s_off_m
    return core - W * s_off_m


rows = []
for case in sorted(CASES, key=lambda c: int(c[1:])):
    tx, ty, mass = CASES[case]
    for simax, axis in (('Mx','x'), ('My','y')):
        s_off = (ty if simax == 'Mx' else tx) * 1e-3
        j_cad = J_CAD[axis] + mass * (Z_CAD**2 + LP[axis]**2)
        c2_cad = float(np.sqrt(mass*G*Z_CAD/j_cad))
        k_cad = 1.0/(mass*G*Z_CAD)
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
            crits_a,_ = cvp.extract_piecewise_batch(bags, axis)
            crits_c,_ = cvp.extract_piecewise_batch(bags, axis,
                                                    cosh_c2=c2_cad,
                                                    ramp_gain=k_cad)
        gated = {c.bag_name for c in crits_a}
        by_bag = {b.name: b for b in bags}
        det = {'cosh': {c.bag_name: (c.onset_moment, c.onset_thrust)
                        for c in crits_a},
               'cosh_cad': {c.bag_name: (c.onset_moment, c.onset_thrust)
                            for c in crits_c}}
        det.update({m: {} for m in METHODS if m not in det})
        for name in sorted(gated):
            bag = by_bag[name]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    crit,_ = cvp.extract_piecewise(bag, axis, model='cosh',
                                                   cosh_c2=None,
                                                   ramp_gain=None)
                det['nls'][name] = (crit.onset_moment, crit.onset_thrust)
            except Exception as e:
                print(f'  nls failed on {case}/{simax}/{name}: {e}',
                      flush=True)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    base,i0,i1,win,guess,direction = _window(bag, axis)
                for m in CLASSIC:
                    j = classic_onset_index(m, base.omega[win], guess,
                                            direction)
                    det[m][name] = (float(base.moment[win][j]),
                                    float(base.f_col[win][j]))
            except Exception as e:
                print(f'  classic failed on {case}/{simax}/{name}: {e}',
                      flush=True)
        W = mass * G
        for name in sorted(gated):
            direc = 'pos' if name.startswith('pos') else 'neg'
            row = dict(case=case, axis=simax, bag=name, dir=direc,
                       truth_mm=(ty if simax=='Mx' else tx), mass=mass)
            for m in METHODS:
                if name in det[m]:
                    m_est, f_on = det[m][name]
                    m_th = m_theory(simax, direc, W, f_on, s_off)
                    row[f'mcrit_{m}'] = f'{m_est:.4f}'
                    row[f'err_{m}'] = f'{1e3*(m_est-m_th):.2f}'
                else:
                    row[f'mcrit_{m}'] = row[f'err_{m}'] = ''
            rows.append(row)
        print(f'done {case}/{simax}', flush=True)

with open('docs/sim_benchmark2_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# per-cell offsets (pair average) and per-method statistics
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in METHODS:
        if r[f'mcrit_{m}']:
            agg[(r['case'], r['axis'])][m][r['dir']].append(
                float(r[f'mcrit_{m}']))

stats = {}
for m in METHODS:
    errs = np.array([float(r[f'err_{m}']) for r in rows if r[f'err_{m}']])
    offs = []
    for (case, simax), g in agg.items():
        if not g[m]['pos'] or not g[m]['neg']:
            continue
        tx, ty, mass = CASES[case]
        truth = ty if simax == 'Mx' else tx
        W = mass * G
        mff = 0.5*(np.mean(g[m]['pos'])+np.mean(g[m]['neg']))
        offs.append(SIGN[simax]*1e3*mff/W - truth)
    offs = np.array(offs)
    cvs = []
    for (case, simax), g in agg.items():
        for d in ('pos', 'neg'):
            v = np.array(g[m][d])
            if len(v) > 1:
                cvs.append(100*v.std(ddof=1)/abs(v.mean()))
    stats[m] = dict(
        rmse_m=float(np.sqrt((errs**2).mean())),
        m_lo=float(np.percentile(errs, 2.5)),
        m_hi=float(np.percentile(errs, 97.5)),
        rmse_o=float(np.sqrt((offs**2).mean())),
        o_lo=float(np.percentile(offs, 2.5)),
        o_hi=float(np.percentile(offs, 97.5)),
        cv=float(np.median(cvs)))
    print(f"{m:12s} M: RMSE {stats[m]['rmse_m']:6.1f} "
          f"[{stats[m]['m_lo']:+.1f},{stats[m]['m_hi']:+.1f}] mN·m | "
          f"off: RMSE {stats[m]['rmse_o']:.2f} "
          f"[{stats[m]['o_lo']:+.2f},{stats[m]['o_hi']:+.2f}] mm | "
          f"med CV {stats[m]['cv']:.1f}%")

with open('docs/tab_sim_benchmark.tex', 'w') as fh:
    fh.write(
        '% Simulation benchmark with empirical 95% intervals, the\n'
        '% theory evaluated exactly as (7)/(14) with the measured\n'
        '% onset collective.  Generated by analysis/sim_benchmark2.py.\n'
        '\\begin{tabular}{l cc cc c}\n\\toprule\n'
        ' & \\multicolumn{2}{c}{$M_{crit,est}$ vs.\\ (7), (14) '
        '[mN$\\cdot$m]} & \\multicolumn{2}{c}{offset error [mm]} & \\\\\n'
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n'
        'method & RMSE & 95\\% interval & RMSE & 95\\% interval & '
        'median CV \\\\\n\\midrule\n')
    for m in METHODS:
        s = stats[m]
        fh.write(f'{LABEL[m]} & ${s["rmse_m"]:.1f}$ & '
                 f'$[{s["m_lo"]:+.0f},\\,{s["m_hi"]:+.0f}]$ & '
                 f'${s["rmse_o"]:.2f}$ & '
                 f'$[{s["o_lo"]:+.2f},\\,{s["o_hi"]:+.2f}]$ & '
                 f'{s["cv"]:.1f}\\% \\\\\n')
    fh.write('\\bottomrule\n\\end{tabular}\n')
print('written docs/sim_benchmark2_runs.csv, docs/tab_sim_benchmark.tex')
