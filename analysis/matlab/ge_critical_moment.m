% GE_CRITICAL_MOMENT  The critical moment under the static ground effect.
%
%   Just before onset the rotors are at partial collective, and the
%   ground-effect interference adds a moment in the tipping direction
%   with two channels (pivot decomposition, static balance at onset):
%
%       Delta M_GE = a + b M,        a = c_a f l_p ,
%
%   with f the total thrust and (c_a, b) the interference-model
%   coefficients.  The tip-over balance  M (1 + b) + a = M_crit,ideal
%   then RE-ADJUSTS the critical moment: the applied moment actually
%   required to tip is
%
%       M'_crit = (M_crit,ideal - a) / (1 + b) ,
%
%   below the ideal (no-GE) value through both channels.  This script
%   draws M'_crit against M_crit,ideal for the roll and pitch ranges,
%   with a band from the +/-20 mm CoM-offset uncertainty of the pivot
%   arm, and exports one PNG with exportgraphics.
%
%   Companion to analysis/ge_offset_effect.py (the same balance, at the
%   deliverable) and docs/access_tight_rms_bound.tex.

clear; close all; clc

% ------------------------------------------------------------------ config
f    = 3.066*9.81*0.7;          % N,  total thrust at onset (70% weight)
c_a  = 0.0431;                  % -,  thrust-height channel (interference)
b    = 0.04314;                 % -,  moment-proportional channel
doff = 0.020;                   % m,  CoM-offset margin on the pivot arm

ax(1) = struct('name', 'roll',  'lp', 0.140, 'M0', [0.7 2.1]);
ax(2) = struct('name', 'pitch', 'lp', 0.110, 'M0', [0.4 1.7]);

co = [0.000 0.447 0.741;        % blue,   roll
      0.850 0.325 0.098];       % orange, pitch

% ------------------------------------------------------------- figure
fig = figure('Color', 'w', 'Units', 'inches', ...
             'Position', [1 1 3.5 2.9]);               % IEEE single column
axh = axes(fig); hold(axh, 'on'); box(axh, 'on'); grid(axh, 'on');

m0max = max([ax.M0]);
hI = plot(axh, [0 1.05*m0max], [0 1.05*m0max], 'k--', 'LineWidth', 0.9);

h = gobjects(2, 1);
for k = 1:2
    m0 = linspace(ax(k).M0(1), ax(k).M0(2), 100);
    a_mid = c_a*f*ax(k).lp;
    a_lo  = c_a*f*(ax(k).lp - doff);
    a_hi  = c_a*f*(ax(k).lp + doff);
    mp    = (m0 - a_mid)/(1 + b);
    mp_up = (m0 - a_lo )/(1 + b);          % smaller a -> higher M'
    mp_dn = (m0 - a_hi )/(1 + b);
    fill(axh, [m0 fliplr(m0)], [mp_dn fliplr(mp_up)], co(k,:), ...
         'FaceAlpha', 0.16, 'EdgeColor', 'none');
    h(k) = plot(axh, m0, mp, '-', 'Color', co(k,:), 'LineWidth', 1.4);
    % percent shift at the range ends, annotated
    for e = [1 numel(m0)]
        text(axh, m0(e), mp(e) - 0.055*m0max, ...
             sprintf('$%+.0f\\%%$', 100*(mp(e)/m0(e) - 1)), ...
             'Interpreter', 'latex', 'FontSize', 7, 'Color', co(k,:), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
    end
end

xlabel(axh, 'Ideal critical moment $M_{\mathrm{crit}}$ (no GE) [N$\,$m]', ...
       'Interpreter', 'latex', 'FontSize', 9);
ylabel(axh, 'Required moment $M''_{\mathrm{crit}}$ [N$\,$m]', ...
       'Interpreter', 'latex', 'FontSize', 9);
legend(axh, [h; hI], ...
       {sprintf('roll, $l_p = %.3f$ m ($\\pm$%.0f mm)', ...
                ax(1).lp, 1e3*doff), ...
        sprintf('pitch, $l_p = %.3f$ m ($\\pm$%.0f mm)', ...
                ax(2).lp, 1e3*doff), ...
        'no ground effect'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', ...
       'FontSize', 8, 'Box', 'off');
set(axh, 'FontName', 'Times New Roman', 'FontSize', 9, ...
         'GridAlpha', 0.15, 'LineWidth', 0.6, 'Layer', 'top');
xlim(axh, [0, 1.05*m0max]);
ylim(axh, [0, 1.05*m0max]);

exportgraphics(fig, 'ge_critical_moment.png', 'Resolution', 600);
fprintf('figure -> ge_critical_moment.png\n');

% ------------------------------------------------------------- report
fprintf(['\nStatic GE at onset:  f = %.2f N,  c_a = %.4f,  b = %.5f\n' ...
         'M''_crit = (M_crit - c_a f l_p)/(1 + b)\n\n'], f, c_a, b);
fprintf('%7s%8s%10s%12s%12s%9s\n', 'axis', 'lp [m]', 'a [N m]', ...
        'M_crit', 'M''_crit', 'shift');
for k = 1:2
    a_mid = c_a*f*ax(k).lp;
    for m0 = ax(k).M0
        mp = (m0 - a_mid)/(1 + b);
        fprintf('%7s%8.3f%10.4f%12.2f%12.3f%8.1f%%\n', ...
                ax(k).name, ax(k).lp, a_mid, m0, mp, 100*(mp/m0 - 1));
    end
end
