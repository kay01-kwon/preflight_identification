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

clear; close all; clc

p_off = (-0.020:0.001:0.020)';       % true offset [m]
g     = 9.81;
m     = [3.066; 3.220];              % kg
W     = m*g;
l_pp  = 0.120;                       % + tip arm [m]
l_pn  = 0.110;                       % - tip arm [m]
alpha = 0.0430;

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 7.2 3.0]);
co  = [0.850 0.325 0.098;            % + tip
       0.000 0.447 0.741;            % - tip
       0.466 0.674 0.188];           % pair average

for i = 1:2
    f_col = 0.70*W(i);               % per-vehicle thrust (bug fix)

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

    subplot(1,2,i); hold on; box on; grid on
    plot(1e3*p_off, 1e3*(pp - p_off),   '--', 'Color', co(1,:), ...
         'LineWidth', 1.2)
    plot(1e3*p_off, 1e3*(pn - p_off),   '--', 'Color', co(2,:), ...
         'LineWidth', 1.2)
    plot(1e3*p_off, 1e3*(pavg - p_off), '-',  'Color', co(3,:), ...
         'LineWidth', 1.8)
    plot(1e3*p_off, 1e3*(0.5*(pp_GE+pn_GE) - p_off), 'k:', ...
         'LineWidth', 1.0)           % sanity rail: identically zero
    % closed forms, overlaid as markers every 5th point
    ii = 1:5:numel(p_off);
    plot(1e3*p_off(ii), 1e3*(-alpha*p_off(ii)/(1+alpha) ...
         + alpha*(l_pn-l_pp)/(2*(1+alpha))), 'o', 'Color', co(3,:), ...
         'MarkerSize', 3.5)
    xlabel('true $p_{\mathrm{off}}$ [mm]', 'Interpreter', 'latex', ...
           'FontSize', 9)
    ylabel('reading error $\hat{p} - p_{\mathrm{off}}$ [mm]', ...
           'Interpreter', 'latex', 'FontSize', 9)
    title(sprintf('m = %.3f kg', m(i)), 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 9, ...
             'GridAlpha', 0.15)
    ylim([-6 6])
    if i == 1
        legend({'$+$ tip alone', '$-$ tip alone', 'pair average', ...
                'GE-model inversion (exact)', ...
                'closed form $-\alpha p/(1{+}\alpha) + \alpha\Delta l/(2(1{+}\alpha))$'}, ...
               'Interpreter', 'latex', 'FontSize', 6.5, ...
               'Location', 'northeast', 'Box', 'off')
    end
end

exportgraphics(fig, 'offset_ge_theory.png', 'Resolution', 600);
fprintf('figure -> offset_ge_theory.png\n');
fprintf(['at p = 0: +tip %.2f mm, -tip %.2f mm, average %.2f mm; ' ...
         'error slope %.4f\n'], ...
        -1e3*alpha*l_pp/(1+alpha), 1e3*alpha*l_pn/(1+alpha), ...
        1e3*alpha*(l_pn-l_pp)/(2*(1+alpha)), -alpha/(1+alpha));
