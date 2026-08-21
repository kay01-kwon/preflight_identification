% OFFSET_GE_THEORY  Eq. (120) drawn: the CoM-offset error the no-GE
%   reading carries under static ground effect.
%
%   Identified moments come from the matched-thrust GE balance,
%       (1+alpha) M_s + s (1+alpha) f l_s = W (s l_s + p_off),
%   and are inverted (i) WITH the GE model -- exact recovery, the
%   sanity rail -- and (ii) WITHOUT it, over p_off in +-20 mm.
%
%   The pair-average error is drawn TWICE, from the two directions it
%   can be reached from, and they coincide to ~1e-18 m:
%     * Eq. (120) itself (solid), built from the IDENTIFIED moments,
%           alpha (M_+ + M_-)/(2W) + alpha f (l_p,+ - l_p,-)/(2W),
%       which is what the experiment evaluates; and
%     * the closed form (markers), built from the TRUE offset and the
%       geometry alone,
%           alpha p_off/(1+alpha) + alpha (l_p,+ - l_p,-)/(2(1+alpha)),
%       with W and f already cancelled analytically.
%   A run-time check also confirms Eq. (120) equals p_off - p_hat_avg
%   to machine precision (~1e-17 m over the sweep).
%
%   SIGN, worth stating once.  Eq. (120)'s right-hand side is
%   (true offset) - (no-GE reading): it is the correction the ground
%   effect hides, positive when the reading under-reads.  If the
%   manuscript's left-hand side is written as
%   p_hat_off,avg - p_off,GE,avg -- reading minus truth -- then the
%   two sides differ by a sign and one of them needs flipping.  The
%   ordinate here follows the right-hand side.
%
%   Substituting the balance also cancels W and f identically,
%       p_off - p_hat_s   = alpha (l_s + s p_off)/(1+alpha),
%       p_off - p_hat_avg = alpha p_off/(1+alpha)
%                           + alpha (l_p,+ - l_p,-)/(2(1+alpha)),
%   so the error depends only on (alpha, l_+, l_-, p_off): the
%   unloaded vehicle is not merely the conservative case, it is the
%   only case, and one panel suffices.
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
pp = Mp/W - (1-f_col/W)*l_pp;
pn = Mn/W + (1-f_col/W)*l_pn;

% the pair average, straight from Eq. (120) -- built from the
% IDENTIFIED moments, so it is what the experiment would evaluate
err_avg_120 = alpha*(Mp + Mn)/(2*W) ...
            + alpha*f_col*(l_pp - l_pn)/(2*W);

% the same quantity in closed form -- built from the TRUE offset and
% the geometry alone, with W and f already cancelled analytically
err_avg_cf  = alpha*p_off/(1+alpha) ...
            + alpha*(l_pp - l_pn)/(2*(1+alpha));

co = [0.850 0.325 0.098;             % + tip
      0.000 0.447 0.741;             % - tip
      0.466 0.674 0.188];            % pair average, Eq. (120)

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 3.5 3.1]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on')

plot(ax, 1e3*p_off, 1e3*(p_off - pp), '--', 'Color', co(1,:), ...
     'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*(p_off - pn), '--', 'Color', co(2,:), ...
     'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*err_avg_120, '-', 'Color', co(3,:), ...
     'LineWidth', 1.8)
ii = 1:4:numel(p_off);
plot(ax, 1e3*p_off(ii), 1e3*err_avg_cf(ii), 'o', 'Color', co(3,:), ...
     'MarkerSize', 3.8, 'LineWidth', 0.9)
plot(ax, 1e3*p_off, 1e3*(p_off - 0.5*(pp_GE+pn_GE)), 'k:', ...
     'LineWidth', 1.0)               % sanity rail: identically zero

xlabel(ax, 'true $p_{\mathrm{off}}$ [mm]', 'Interpreter', 'latex', ...
       'FontSize', 9)
ylabel(ax, '$p_{\mathrm{off}} - \hat{p}_{\mathrm{off}}$ [mm]', ...
       'Interpreter', 'latex', 'FontSize', 9)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 9, 'GridAlpha', 0.15)
xlim(ax, [-20 20]);  ylim(ax, [-7 7])

legend(ax, {'$+$ tip alone', '$-$ tip alone', ...
            'pair average, Eq. (120)', ...
            'theory: $\frac{\alpha p_{\mathrm{off}}}{1+\alpha} + \frac{\alpha \Delta l_p}{2(1+\alpha)}$', ...
            'GE inversion (exact)'}, ...
       'Interpreter', 'latex', 'FontSize', 7.5, 'NumColumns', 2, ...
       'Location', 'southoutside', 'Box', 'off')

exportgraphics(fig, 'offset_ge_theory.png', 'Resolution', 600);
fprintf('figure -> offset_ge_theory.png\n');

% Eq. (120) against the direct difference, and the two terms
chk = max(abs(err_avg_120 - (p_off - 0.5*(pp + pn))));
fprintf('Eq.(120) vs (p_off - p_hat_avg): max |diff| = %.2e m\n', chk);
fprintf('Eq.(120) vs closed form         : max |diff| = %.2e m\n', ...
        max(abs(err_avg_120 - err_avg_cf)));
fprintf(['at p = 0: moment term %+.3f mm, thrust term %+.3f mm, ' ...
         'total %+.3f mm\n'], ...
        1e3*alpha*(Mp(p_off==0) + Mn(p_off==0))/(2*W), ...
        1e3*alpha*f_col*(l_pp - l_pn)/(2*W), ...
        1e3*err_avg_120(p_off==0));
fprintf('pair average at p = -20 / +20 mm: %+.3f / %+.3f mm\n', ...
        1e3*err_avg_120(1), 1e3*err_avg_120(end));
fprintf('single direction at p = 0: %+.3f / %+.3f mm\n', ...
        1e3*alpha*l_pp/(1+alpha), -1e3*alpha*l_pn/(1+alpha));
