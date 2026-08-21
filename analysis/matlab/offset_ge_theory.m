% OFFSET_GE_THEORY  The no-GE offset reading error, pure theory sweep.
%
%   Generates the identified moments from the matched-thrust GE balance,
%       (1+alpha) M_s + s (1+alpha) f l_s = W (s l_s + p_off),
%   inverts them (i) with the GE model -- exact recovery, the sanity
%   rail -- and (ii) without it, and sweeps the true offset over
%   +-20 mm.  Closed forms the curves must land on:
%
%       p_hat_s   = p/(1+a) - s a l_s/(1+a)          (f cancels exactly)
%       p_hat_avg = p/(1+a) + a (l_n - l_p)/(2(1+a))
%
%   so the pair-average error is the -a/(1+a) = -4.13% line plus an
%   f-independent arm-asymmetry constant.  Companion to
%   docs/ge_offset_shift.tex and analysis/offset_error_sweep.py.
%
%   LAYOUT NOTE.  The '-' tip curve reaches +5.4 mm at the left edge, so
%   a legend placed inside the axes at 'north' lands on top of it.  The
%   panels therefore share ONE legend in a south tile below the figure
%   (tiledlayout, R2019b+), which also removes the duplicate.

clear; close all; clc

p_off = (-0.020:0.001:0.020)';       % true offset [m]
g     = 9.81;
m     = [3.066; 3.220];              % kg
W     = m*g;
l_pp  = 0.120;                       % + tip arm [m]
l_pn  = 0.110;                       % - tip arm [m]
alpha = 0.0430;

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 3.5 5.4]);
tl  = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', ...
                  'Padding', 'compact');
co  = [0.850 0.325 0.098;            % + tip
       0.000 0.447 0.741;            % - tip
       0.466 0.674 0.188];           % pair average
h   = gobjects(5, 1);

for i = 1:2
    f_col = 0.70*W(i);               % per-vehicle thrust

    % GE-biased identified moments (matched-thrust balance)
    Mp = -f_col*l_pp + (l_pp + p_off)/(1+alpha)*W(i);
    Mn =  f_col*l_pn + (-l_pn + p_off)/(1+alpha)*W(i);

    % (i) inverted WITH the GE model: exact recovery, = p_off
    pp_GE = (1+alpha)*Mp/W(i) - (1-(1+alpha)*f_col/W(i))*l_pp;
    pn_GE = (1+alpha)*Mn/W(i) + (1-(1+alpha)*f_col/W(i))*l_pn;

    % (ii) inverted WITHOUT the GE model
    pp = Mp/W(i) - (1-f_col/W(i))*l_pp;
    pn = Mn/W(i) + (1-f_col/W(i))*l_pn;
    pavg = 0.5*(pp + pn);

    ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on')
    h(1) = plot(ax, 1e3*p_off, 1e3*(pp - p_off), '--', ...
                'Color', co(1,:), 'LineWidth', 1.2);
    h(2) = plot(ax, 1e3*p_off, 1e3*(pn - p_off), '--', ...
                'Color', co(2,:), 'LineWidth', 1.2);
    h(3) = plot(ax, 1e3*p_off, 1e3*(pavg - p_off), '-', ...
                'Color', co(3,:), 'LineWidth', 1.8);
    h(4) = plot(ax, 1e3*p_off, 1e3*(0.5*(pp_GE+pn_GE) - p_off), 'k:', ...
                'LineWidth', 1.0);   % sanity rail: identically zero
    % closed forms, overlaid as markers every 5th point
    ii = 1:5:numel(p_off);
    h(5) = plot(ax, 1e3*p_off(ii), 1e3*(-alpha*p_off(ii)/(1+alpha) ...
                + alpha*(l_pn-l_pp)/(2*(1+alpha))), 'o', ...
                'Color', co(3,:), 'MarkerSize', 3.5);
    if i == 2
        xlabel(ax, 'true $p_{\mathrm{off}}$ [mm]', ...
               'Interpreter', 'latex', 'FontSize', 9)
    end
    ylabel(ax, 'reading error $\hat{p} - p_{\mathrm{off}}$ [mm]', ...
           'Interpreter', 'latex', 'FontSize', 9)
    title(ax, sprintf('m = %.3f kg', m(i)), 'FontSize', 9)
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 9, ...
            'GridAlpha', 0.15)
    ylim(ax, [-7 7])
end

lg = legend(h, {'$\hat{p}_{\mathrm{off},+}$', ...
                '$\hat{p}_{\mathrm{off},-}$', 'pair average', ...
                'GE inversion (exact)', 'closed form'}, ...
            'Interpreter', 'latex', 'FontSize', 7.5, ...
            'NumColumns', 3, 'Box', 'off');
lg.Layout.Tile = 'south';

exportgraphics(fig, 'offset_ge_theory.png', 'Resolution', 600);
fprintf('figure -> offset_ge_theory.png\n');
fprintf(['at p = 0: +tip %.2f mm, -tip %.2f mm, average %.2f mm; ' ...
         'error slope %.4f\n'], ...
        -1e3*alpha*l_pp/(1+alpha), 1e3*alpha*l_pn/(1+alpha), ...
        1e3*alpha*(l_pn-l_pp)/(2*(1+alpha)), -alpha/(1+alpha));
