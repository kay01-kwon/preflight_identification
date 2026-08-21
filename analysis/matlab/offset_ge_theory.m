% OFFSET_GE_THEORY  The no-GE offset reading error, pure theory sweep.
%
%   Generates the identified moments from the matched-thrust GE balance,
%       (1+alpha) M_s + s (1+alpha) f l_s = W (s l_s + p_off),
%   inverts them (i) with the GE model -- exact recovery, the sanity
%   rail -- and (ii) without it, and sweeps the true offset over
%   +-20 mm.  Substituting the balance into the no-GE inversion leaves
%
%       p_hat_s - p = -alpha (l_s + s p)/(1+alpha)
%       p_hat_avg - p = -alpha p/(1+alpha)
%                       + alpha (l_n - l_p)/(2(1+alpha)),
%
%   in which BOTH W and f have cancelled identically: the reading error
%   depends only on (alpha, l_+, l_-, p_off).  The unloaded vehicle is
%   therefore not merely the conservative case, it is the only case --
%   3.220 kg gives the same curves to the last digit -- so one panel is
%   drawn.
%
%   Companion to docs/ge_offset_shift.tex and
%   analysis/offset_error_sweep.py.

clear; close all; clc

p_off = (-0.020:0.001:0.020)';       % true offset [m]
g     = 9.81;
m     = 3.066;                       % kg, unloaded
W     = m*g;
f_col = 0.70*W;
l_pp  = 0.120;                       % + tip arm [m]
l_pn  = 0.110;                       % - tip arm [m]
alpha = 0.0430;

% GE-biased identified moments (matched-thrust balance)
Mp = -f_col*l_pp + (l_pp + p_off)/(1+alpha)*W;
Mn =  f_col*l_pn + (-l_pn + p_off)/(1+alpha)*W;

% (i) inverted WITH the GE model: exact recovery, = p_off
pp_GE = (1+alpha)*Mp/W - (1-(1+alpha)*f_col/W)*l_pp;
pn_GE = (1+alpha)*Mn/W + (1-(1+alpha)*f_col/W)*l_pn;

% (ii) inverted WITHOUT the GE model
pp   = Mp/W - (1-f_col/W)*l_pp;
pn   = Mn/W + (1-f_col/W)*l_pn;
pavg = 0.5*(pp + pn);

co = [0.850 0.325 0.098;             % + tip
      0.000 0.447 0.741;             % - tip
      0.466 0.674 0.188];            % pair average

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 3.5 3.1]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on')

plot(ax, 1e3*p_off, 1e3*(pp - p_off), '--', 'Color', co(1,:), ...
     'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*(pn - p_off), '--', 'Color', co(2,:), ...
     'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*(pavg - p_off), '-', 'Color', co(3,:), ...
     'LineWidth', 1.8)
plot(ax, 1e3*p_off, 1e3*(0.5*(pp_GE+pn_GE) - p_off), 'k:', ...
     'LineWidth', 1.0)               % sanity rail: identically zero

xlabel(ax, 'true $p_{\mathrm{off}}$ [mm]', 'Interpreter', 'latex', ...
       'FontSize', 9)
ylabel(ax, 'reading error $\hat{p} - p_{\mathrm{off}}$ [mm]', ...
       'Interpreter', 'latex', 'FontSize', 9)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 9, 'GridAlpha', 0.15)
xlim(ax, [-20 20]);  ylim(ax, [-7 7])

legend(ax, {'$\hat{p}_{\mathrm{off},+}$', '$\hat{p}_{\mathrm{off},-}$', ...
            'pair average', 'GE inversion (exact)'}, ...
       'Interpreter', 'latex', 'FontSize', 8, 'NumColumns', 2, ...
       'Location', 'southoutside', 'Box', 'off')

exportgraphics(fig, 'offset_ge_theory.png', 'Resolution', 600);
fprintf('figure -> offset_ge_theory.png\n');
fprintf(['at p = 0: +tip %.2f mm, -tip %.2f mm, average %.2f mm; ' ...
         'error slope %.4f\n'], ...
        -1e3*alpha*l_pp/(1+alpha), 1e3*alpha*l_pn/(1+alpha), ...
        1e3*alpha*(l_pn-l_pp)/(2*(1+alpha)), -alpha/(1+alpha));
fprintf('pair-average error at p = +-20 mm: %+.2f / %+.2f mm\n', ...
        1e3*(pavg(end) - p_off(end)), 1e3*(pavg(1) - p_off(1)));
