#!/usr/bin/env python3
"""Generate the simulation sweep, with each case checked before it is run.

WHY THE SWEEP IS WIDENED. The three cases of Table 8 --- (-6, 0),
(0, 10) and (10, -5) mm --- reach 11 mm at most and two of them load a
single axis, so they neither cover the +/-20 mm design box the paper
declares in Table 2 nor exercise a genuinely asymmetric load.

WHERE THE METHOD ACTUALLY ENDS, which is what a robustness sweep should
find rather than avoid. From the critical moments,

    M_x,+ = (W-f) l_r + W p_y        M_x,- = -(W-f) l_l + W p_y
    M_y,+ = (W-f) l_f - W p_x        M_y,- = -(W-f) l_b - W p_x

the negative-direction moment vanishes at p = (1 - f/W) l, which is
33 mm on roll and 42 mm on pitch at the f/W = 0.70 used here. Past that
the corresponding contact carries no load once the excitation
collective is applied, that direction has no critical moment, and the
pairing has nothing to pair. The bound belongs to the excitation and
not to the airframe: with no collective the vehicle is statically
stable out to the landing-gear half-width itself, 110 and 140 mm. The
allocator saturates only later, at 46 and 49 mm, so within this
procedure it is the balance and not the actuation that binds.

The validity region is therefore a rectangle in the offset plane, and
a sweep that walks outward on more than one bearing crosses a
different side of it each time -- which is why the failures below are
placed on both axes rather than on one.

THE DESIGN. Four groups, each answering a different question:
  small      the three published cases, retained so tables 13-14 stay
             valid and the small-offset regime is still covered
  corner     all four corners of the declared box, both axes at their
             limit at once -- the asymmetric load the review asks for
  diagonal   both components grown together until the roll side of the
             validity rectangle is crossed
  shallow    grown on a shallow bearing so the pitch side is crossed
             instead, so the failure is not an artefact of one axis
  loaded     the same diagonal offset at the loaded mass

Each case is emitted with the critical moments it should produce and a
verdict --- ok, or the reason it is expected to fail --- so a run that
disagrees with its prediction is visible immediately rather than after
the fact.

Usage
-----
  python analysis/sim_case_list.py [--out DIR]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

G = 9.81
M_UNLOADED, M_LOADED = 3.066, 3.220
THRUST_FRAC = 0.70               # collective held during excitation
L_ROLL, L_PITCH = 0.110, 0.140   # pivot half-widths, table 2
M_MAX = dict(roll=2.371, pitch=2.737)   # allocator caps, figs 12-13
Z_COM = 0.261                    # CAD, for the simulator model


def bearing(mag, deg):
    r = np.radians(deg)
    return round(mag * np.cos(r), 1), round(mag * np.sin(r), 1)


CASES = (
    # the published three, retained so tables 13-14 stay valid and the
    # small-offset regime is still covered
    [dict(group='small', x=x, y=y) for x, y in
     ((-6.0, 0.0), (0.0, 10.0), (10.0, -5.0))]
    # both axes at the limit of the declared box, simultaneously
    + [dict(group='corner', x=x, y=y) for x, y in
       ((20.0, 20.0), (20.0, -20.0), (-20.0, 20.0), (-20.0, -20.0))]
    # walked outward at 45 deg: both components grow together until the
    # roll side of the validity rectangle is crossed
    + [dict(group='diagonal', x=v, y=v) for v in (25.0, 32.0, 38.0)]
    # walked outward on a shallow bearing, so the pitch side is crossed
    # instead -- the failure is not an artefact of one axis
    + [dict(group='shallow', x=x, y=y) for x, y in
       ((38.0, 14.0), (46.0, 17.0))]
    + [dict(group='loaded', x=25.0, y=25.0, mass=M_LOADED)]
)


def predict(c):
    """Critical moments and the verdict for one case."""
    m = c.get('mass', M_UNLOADED)
    W = m * G
    f = THRUST_FRAC * W
    d = (W - f)
    px, py = c['x'] * 1e-3, c['y'] * 1e-3
    M = dict(roll_pos=d * L_ROLL + W * py,
             roll_neg=-d * L_ROLL + W * py,
             pitch_pos=d * L_PITCH - W * px,
             pitch_neg=-d * L_PITCH - W * px)
    why = []
    for ax in ('roll', 'pitch'):
        # the pair only exists while the two directions keep their signs
        if M[f'{ax}_pos'] <= 0 or M[f'{ax}_neg'] >= 0:
            why.append(f'{ax}: contact unloaded under the excitation '
                       f'collective')
        elif max(abs(M[f'{ax}_pos']),
                 abs(M[f'{ax}_neg'])) > M_MAX[ax]:
            why.append(f'{ax}: excitation exceeds the allocator cap')
    return M, m, ('ok' if not why else '; '.join(why))


# Label placement, in millimetres of the plot's own units so the Python
# and MATLAB figures agree. The default sits up and to the right of its
# marker; the overrides move the few that would otherwise sit on the
# identifiable boundary or on a neighbouring marker.
LABEL_NUDGE = {
    (32.0, 32.0): (-2.5, 4.2),    # clears the roll bound drawn at y = 33
    (38.0, 14.0): (-9.5, -3.8),   # clears the failure marker to its right
    (25.0, 25.0): (2.0, -4.8),    # the loaded case, under the point it repeats
}
NUDGE_DEFAULT = (1.8, 1.8)


def nudge(r):
    d = LABEL_NUDGE.get((r['x_off_mm'], r['y_off_mm']), NUDGE_DEFAULT)
    # the unloaded twin of a repeated offset keeps the default
    if (r['x_off_mm'], r['y_off_mm']) == (25.0, 25.0) \
            and r['group'] != 'loaded':
        return NUDGE_DEFAULT
    return d


def figure(rows, outdir):
    """The sweep in the offset plane, which is where it reads at a glance.

    The identifiable region is a rectangle and not a disc -- the roll
    bound sits on the y component and the pitch bound on the x -- so
    coverage and the two crossings are geometric facts a reader can see,
    where a table asks them to compare columns of numbers.

    The landing-gear footprint, 110 by 140 mm, is left out of the axes
    deliberately: drawn to scale it would be seven times the area of
    everything else and compress the part that matters. It belongs in
    the caption.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    W = M_UNLOADED * G
    f = THRUST_FRAC * W
    ylim = (1 - f / W) * L_ROLL * 1e3
    xlim = (1 - f / W) * L_PITCH * 1e3

    def box(ax, bx, by, **kw):
        ax.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], **kw)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    box(ax, xlim, ylim, color='#0072B2', lw=1.8, zorder=2,
        label=f'identifiable, ${xlim:.0f}\\times{ylim:.0f}$ mm')
    box(ax, 20, 20, color='#D55E00', lw=1.4, ls='--', zorder=2,
        label='declared design box, $20\\times20$ mm')

    # the loaded case repeats an unloaded offset, so it needs its own
    # marker or the two sit on top of each other with their labels
    ok = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['verdict'] == 'ok' and r['group'] != 'loaded']
    ld = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['group'] == 'loaded']
    no = [(r['x_off_mm'], r['y_off_mm']) for r in rows
          if r['verdict'] != 'ok']
    ax.plot(*zip(*ok), 'o', ms=6.5, mfc='#4DAF4A', mec='0.2', mew=0.9,
            ls='', zorder=4, label='expected to apply')
    ax.plot(*zip(*ld), 's', ms=8.5, mfc='none', mec='#1B7837', mew=1.8,
            ls='', zorder=3, label=f'same offset, loaded ({M_LOADED} kg)')
    ax.plot(*zip(*no), 'X', ms=9.5, mfc='#C1121F', mec='0.2', mew=0.9,
            ls='', zorder=4, label='expected to fail')
    for k, r in enumerate(rows, 1):
        dx, dy = nudge(r)
        ax.annotate(f'S{k}', (r['x_off_mm'] + dx, r['y_off_mm'] + dy),
                    fontsize=7.5, color='0.25', zorder=5,
                    ha='center', va='center')

    ax.set_xlim(-58, 58)
    ax.set_ylim(-46, 46)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.8', lw=0.7, zorder=1)
    ax.axvline(0, color='0.8', lw=0.7, zorder=1)
    ax.grid(alpha=0.4, lw=0.7, color='0.65')
    ax.set_axisbelow(True)
    ax.set_xlabel('$x_{\\mathrm{off}}$ [mm]', fontsize=9.5)
    ax.set_ylabel('$y_{\\mathrm{off}}$ [mm]', fontsize=9.5)
    ax.legend(fontsize=7.6, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        fig.savefig(outdir / f'fig_sim_cases.{ext}',
                    bbox_inches='tight', **kw)
    plt.close(fig)


RAMP_RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)   # table 4
RAMP_FAIL = (0.30, 0.90)        # a case that fails does so at any rate
ABLATION_CASES = ('sim_02', 'sim_08')   # one small offset, one large


def checklist(rows, outdir):
    """A tickable run list, ordered so an interruption leaves whole cases.

    The rotor ablation comes first. It compares the identified
    second-order-plus-delay rotor against an ideal one, and if the ideal
    rotor does not recover the truth there is a fault to find before
    three hundred more runs are spent. Its ramp-rate axis is the point,
    not a formality: lag enters the excitation as a fraction of the
    window, so whatever it costs should grow with the commanded rate.

    Cases predicted to fail are run at two rates rather than seven. They
    fail because a contact carries no load once the collective is
    applied, which no ramp rate changes, so the remaining five runs
    would only repeat the demonstration.
    """
    by_id = {r['id']: r for r in rows}
    n_main = sum(len(RAMP_RATES if r['verdict'] == 'ok' else RAMP_FAIL)
                 * 4 for r in rows)
    n_abl = len(ABLATION_CASES) * len(RAMP_RATES) * 4

    def rate_row(label, rates):
        return (f'- {label:<6}' +
                ' '.join(f'[ ]{v:.2f}' for v in rates))

    L = [f'# Simulation run checklist',
         '',
         f'Generated by `analysis/sim_case_list.py`. '
         f'{n_abl + n_main} runs in total.',
         '',
         'Each entry is one (case, axis, ramp rate); both tip directions '
         'are run under it, so one tick covers two excitations. Rates are '
         'in N m/s.',
         '',
         '## Order',
         '',
         f'- [ ] **1. Rotor ablation, ideal rotor** ({n_abl} runs) '
         f'--- run this first',
         f'- [ ] **2. Main sweep, identified rotor** ({n_main} runs)',
         '',
         'The ablation is the same cases under an ideal rotor; the '
         'identified-rotor half of the comparison comes from the main '
         'sweep, so nothing is run twice.',
         '',
         '---',
         '',
         f'## 1. Rotor ablation --- ideal rotor (R0)',
         '',
         'Compare against the same cases in the main sweep, which uses '
         'the identified second-order-plus-delay rotor (R3). Plot the '
         'offset error against the ramp rate: R0 should be flat and R3 '
         'should not.',
         '']
    for cid in ABLATION_CASES:
        r = by_id[cid]
        k = rows.index(r) + 1
        L += [f'### S{k} (R0) --- ({r["x_off_mm"]:g}, {r["y_off_mm"]:g}) mm',
              rate_row('roll', RAMP_RATES),
              rate_row('pitch', RAMP_RATES), '']

    L += ['---', '', '## 2. Main sweep --- identified rotor (R3)', '']
    for k, r in enumerate(rows, 1):
        rates = RAMP_RATES if r['verdict'] == 'ok' else RAMP_FAIL
        note = ('' if r['verdict'] == 'ok'
                else f'  \n  **expected to fail** --- {r["verdict"]}')
        L += [f'### S{k} --- {r["group"]}, '
              f'({r["x_off_mm"]:g}, {r["y_off_mm"]:g}) mm, '
              f'{r["mass_kg"]} kg',
              f'predicted $M_{{crit}}$  roll {r["roll_neg"]:+.2f} / '
              f'{r["roll_pos"]:+.2f}   pitch {r["pitch_neg"]:+.2f} / '
              f'{r["pitch_pos"]:+.2f}{note}',
              '',
              rate_row('roll', rates),
              rate_row('pitch', rates),
              ('- [ ] identified offset within 1 mm of truth'
               if r['verdict'] == 'ok' else
               '- [ ] the predicted contact does lift at zero applied '
               'moment, and the pairing has no second direction'),
              '']

    (outdir / 'sim_checklist.md').write_text('\n'.join(L))
    return n_abl, n_main


def matlab(rows, outdir):
    """The same figure as a standalone .m, for the manuscript toolchain.

    The case data is written into the script rather than read from the
    CSV so the file runs on its own, and it is emitted from here rather
    than maintained by hand so the two figures cannot disagree.
    """
    W = M_UNLOADED * G
    f = THRUST_FRAC * W
    yl = (1 - f / W) * L_ROLL * 1e3
    xl = (1 - f / W) * L_PITCH * 1e3

    def arr(sel):
        pts = []
        for k, r in enumerate(rows, 1):
            if not sel(r):
                continue
            dx, dy = nudge(r)
            pts.append((r['x_off_mm'], r['y_off_mm'], k,
                        r['x_off_mm'] + dx, r['y_off_mm'] + dy))
        return ('[' + '; '.join(f'{x:g} {y:g} {k} {lx:g} {ly:g}'
                                for x, y, k, lx, ly in pts) + ']')

    ok = arr(lambda r: r['verdict'] == 'ok' and r['group'] != 'loaded')
    ld = arr(lambda r: r['group'] == 'loaded')
    no = arr(lambda r: r['verdict'] != 'ok')

    with open(outdir / 'plot_sim_cases.m', 'w') as fh:
        fh.write(f"""% PLOT_SIM_CASES  The simulation sweep in the offset plane.
%
% Generated by analysis/sim_case_list.py -- regenerate rather than edit,
% so this figure and its Python counterpart cannot disagree.
%
% The identifiable region is a rectangle and not a disc: the roll bound
% falls on y_off and the pitch bound on x_off, at (1 - f/W) l_p for the
% collective held during excitation. The landing-gear footprint,
% 110 by 140 mm, bounds the vehicle only with no collective applied and
% is deliberately off these axes -- drawn to scale it would be seven
% times the area of everything else.

close all; clear; clc

x_lim = {xl:.4f};      % mm, pitch bound  (1 - f/W) * l_p,theta
y_lim = {yl:.4f};      % mm, roll  bound  (1 - f/W) * l_p,phi
box_x = 20; box_y = 20;   % declared design box

ok   = {ok};
ld   = {ld};
fail = {no};

% closed polylines: plot objects are legendable, rectangle() is not
rect = @(a,b) deal([-a a a -a -a], [-b -b b b -b]);

fig = figure('Color','w','Units','inches','Position',[1 1 5.4 4.6]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on')

[rx,ry] = rect(x_lim, y_lim);
h1 = plot(ax, rx, ry, '-',  'Color',[0 0.447 0.741], 'LineWidth',1.8);
[bx,by] = rect(box_x, box_y);
h2 = plot(ax, bx, by, '--', 'Color',[0.851 0.325 0.098], 'LineWidth',1.4);

h3 = plot(ax, ok(:,1), ok(:,2), 'o', 'MarkerSize',6.5, ...
          'MarkerFaceColor',[0.302 0.686 0.290], 'MarkerEdgeColor',[.2 .2 .2]);
h4 = plot(ax, ld(:,1), ld(:,2), 's', 'MarkerSize',9, ...
          'MarkerFaceColor','none', 'MarkerEdgeColor',[0.106 0.471 0.216], ...
          'LineWidth',1.8);
h5 = plot(ax, fail(:,1), fail(:,2), 'x', 'MarkerSize',11, ...
          'Color',[0.757 0.071 0.122], 'LineWidth',2.4);

% columns 4-5 carry the label position, nudged where a default would
% land on the identifiable boundary or on a neighbouring marker
for T = {{ok, ld, fail}}
    P = T{{1}};
    for k = 1:size(P,1)
        text(ax, P(k,4), P(k,5), sprintf('S%d',P(k,3)), ...
             'FontSize',8, 'Color',[.25 .25 .25], ...
             'HorizontalAlignment','center', 'VerticalAlignment','middle');
    end
end

axis(ax,'equal');  xlim(ax,[-58 58]);  ylim(ax,[-46 46])
xlabel(ax,'$x_{{\\mathrm{{off}}}}$ [mm]','Interpreter','latex','FontSize',11)
ylabel(ax,'$y_{{\\mathrm{{off}}}}$ [mm]','Interpreter','latex','FontSize',11)
set(ax,'FontName','Times New Roman','FontSize',10,'GridAlpha',0.15)

legend(ax, [h1 h2 h3 h4 h5], ...
   {{sprintf('identifiable, %.0f \\\\times %.0f mm', x_lim, y_lim), ...
     'declared design box, 20 \\times 20 mm', ...
     'expected to apply', ...
     'same offset, loaded ({M_LOADED} kg)', ...
     'expected to fail'}}, ...
   'Location','southoutside', 'NumColumns',2, 'Box','off', ...
   'FontSize',9, 'Interpreter','tex')

exportgraphics(fig, 'fig_sim_cases.png', 'Resolution', 600);
exportgraphics(fig, 'fig_sim_cases.pdf', 'ContentType', 'vector');
fprintf('figure -> fig_sim_cases.png / .pdf\\n');
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'DataSet')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, c in enumerate(CASES, 1):
        M, m, verdict = predict(c)
        rows.append(dict(id=f'sim_{k:02d}', group=c['group'],
                         x_off_mm=c['x'], y_off_mm=c['y'],
                         mag_mm=round(float(np.hypot(c['x'], c['y'])), 1),
                         mass_kg=m, z_com_m=Z_COM,
                         **{k2: round(v, 3) for k2, v in M.items()},
                         verdict=verdict))

    with open(a.out / 'sim_cases.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # a YAML the simulator can read without a parser dependency here
    with open(a.out / 'sim_cases.yaml', 'w') as fh:
        fh.write('# Generated by analysis/sim_case_list.py -- do not edit.\n'
                 '# Offsets in mm from the geometric centre; the critical\n'
                 '# moments are what each case should produce, and the\n'
                 '# verdict is whether the procedure is expected to apply.\n'
                 'cases:\n')
        for r in rows:
            fh.write(f'  - id: {r["id"]}\n'
                     f'    group: {r["group"]}\n'
                     f'    x_off_mm: {r["x_off_mm"]}\n'
                     f'    y_off_mm: {r["y_off_mm"]}\n'
                     f'    mass_kg: {r["mass_kg"]}\n'
                     f'    z_com_m: {r["z_com_m"]}\n'
                     f'    expect:\n'
                     f'      roll:  [{r["roll_neg"]}, {r["roll_pos"]}]\n'
                     f'      pitch: [{r["pitch_neg"]}, {r["pitch_pos"]}]\n'
                     f'      verdict: "{r["verdict"]}"\n')

    # the paper table. Simulation cases are prefixed S so they cannot be
    # confused with the hardware cases 01-05, which the present text
    # numbers the same way.
    REGIME = {'small': 'small offset', 'corner': 'box corner',
              'diagonal': 'diagonal reach', 'shallow': 'shallow reach',
              'loaded': 'loaded'}
    with open(a.out / 'tab_sim_cases.tex', 'w') as fh:
        fh.write('% Generated by analysis/sim_case_list.py.\n'
                 '\\begin{tabular}{c l r r r c}\n'
                 '  \\toprule\n'
                 '  Case & regime & {$x_{\\mathrm{off}}$ (mm)} & '
                 '{$y_{\\mathrm{off}}$ (mm)} & {$|p_{\\mathrm{off}}|$ (mm)}'
                 ' & applies \\\\\n  \\midrule\n')
        prev = None
        for k, r in enumerate(rows, 1):
            if prev and r['group'] != prev:
                fh.write('  \\addlinespace\n')
            prev = r['group']
            ap = 'yes' if r['verdict'] == 'ok' else '\\textbf{no}'
            fh.write(f'  S{k} & {REGIME[r["group"]]} & '
                     f'${r["x_off_mm"]:g}$ & ${r["y_off_mm"]:g}$ & '
                     f'${r["mag_mm"]:.1f}$ & {ap} \\\\\n')
        fh.write('  \\bottomrule\n\\end{tabular}\n')

    print(f'{"id":<9}{"group":<9}{"x":>7}{"y":>7}{"|p|":>7}{"mass":>7}'
          f'{"roll -/+":>16}{"pitch -/+":>16}  verdict')
    for r in rows:
        print(f'{r["id"]:<9}{r["group"]:<9}{r["x_off_mm"]:>7.1f}'
              f'{r["y_off_mm"]:>7.1f}{r["mag_mm"]:>7.1f}{r["mass_kg"]:>7.3f}'
              f'{r["roll_neg"]:>8.2f}{r["roll_pos"]:>8.2f}'
              f'{r["pitch_neg"]:>8.2f}{r["pitch_pos"]:>8.2f}'
              f'  {r["verdict"]}')

    figure(rows, a.out)
    matlab(rows, a.out)
    n_abl, n_main = checklist(rows, a.out)

    n_ok = sum(r['verdict'] == 'ok' for r in rows)
    print(f'\n{len(rows)} cases, {n_ok} expected to apply, '
          f'{len(rows) - n_ok} expected to fail by design')
    print(f'runs: {n_abl} ablation + {n_main} sweep = {n_abl + n_main}')
    print(f'written to {a.out / "sim_cases.csv"} and sim_cases.yaml')


if __name__ == '__main__':
    main()
