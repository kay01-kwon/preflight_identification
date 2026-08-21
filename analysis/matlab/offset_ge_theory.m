% OFFSET_GE_THEORY  The CoM offset identified WITHOUT ground effect
%   versus the one identified WITH it -- the difference, generalised.
%
%   THE QUESTION.  A vehicle really tips with ground effect acting on
%   BOTH channels: the collective thrust is augmented by (1+alpha_f)
%   and the applied moment by (1+alpha_M).  The measured onset moments
%   are therefore
%
%       M_s = [ W (s l_s + p_off) - s (1+alpha_f) f l_s ] / (1+alpha_M)
%
%   with s = +-1 the tip direction.  Two models are then handed the
%   SAME moments and asked for the offset:
%
%     (a) NO ground effect anywhere -- neither channel:
%           M_s + s f l_s = W (s l_s + p_hat_s)
%     (b) FULL ground effect -- both channels:
%           (1+alpha_M) M_s + s (1+alpha_f) f l_s = W (s l_s + p_hat_s)
%
%   Model (b) returns p_off identically (verified at run time), so the
%   quantity drawn here,
%
%       p_hat_GE - p_hat_noGE   ( = p_off - p_hat_noGE ),
%
%   is exactly the disagreement between the two identifications.
%
%   THE CLOSED FORM, for arbitrary gains.  Substituting the balance
%   into (a) cancels W and f from the leading behaviour and leaves
%
%     per direction:
%       p_hat_GE,s - p_hat_noGE,s = alpha_M (p_off + s l_s)/(1+alpha_M)
%                        - s f l_s (alpha_M - alpha_f)/(W (1+alpha_M))
%     pair average:
%       = alpha_M p_off/(1+alpha_M)
%         + [ alpha_M - f (alpha_M - alpha_f)/W ]
%           (l_p,+ - l_p,-) / (2 (1+alpha_M))
%
%   Two things follow, and they are the point of the figure.  The
%   SLOPE in p_off is alpha_M/(1+alpha_M) -- the moment channel alone;
%   the thrust channel cannot touch it.  The thrust channel only
%   scales the constant term, through the ARM ASYMMETRY l_+ - l_-,
%   and drops out entirely when alpha_f = alpha_M.  With equal gains
%   the pair average reduces to manuscript Eq. (120).
%
%   Because W and f survive only in that asymmetry term, the error is
%   set by (alpha_f, alpha_M, l_+, l_-, p_off): the unloaded vehicle
%   is not merely the conservative case, it is the only case, and one
%   panel suffices.
%
%   SIGN.  The ordinate is (true offset) - (no-GE reading): the
%   correction the ground effect hides, positive when the no-GE model
%   under-reads.  A manuscript left-hand side written the other way
%   round, p_hat_off,avg - p_off,GE,avg, differs from it by a sign.
%
%   Companion to docs/ge_offset_shift.tex and
%   analysis/offset_error_sweep.py.

clear; close all; clc

p_off   = (-0.020:0.001:0.020)';     % true offset [m]
g       = 9.81;
m       = 3.066;                     % kg, unloaded
W       = m*g;
f_col   = 0.70*W;
l_pp    = 0.120;                     % + tip arm [m]
l_pn    = 0.110;                     % - tip arm [m]
alpha_f = 0.0430;                    % thrust-channel gain
alpha_M = 0.0430;                    % moment-channel gain

% ---- what the vehicle actually does: GE in BOTH channels ----------
Mp = ( W*( l_pp + p_off) - (1+alpha_f)*f_col*l_pp ) / (1+alpha_M);
Mn = ( W*(-l_pn + p_off) + (1+alpha_f)*f_col*l_pn ) / (1+alpha_M);

% ---- (a) identified with NO ground effect anywhere ----------------
pp_0 = Mp/W - (1 - f_col/W)*l_pp;
pn_0 = Mn/W + (1 - f_col/W)*l_pn;

% ---- (b) identified with FULL ground effect, both channels --------
pp_G = ((1+alpha_M)*Mp + (1+alpha_f)*f_col*l_pp)/W - l_pp;
pn_G = ((1+alpha_M)*Mn - (1+alpha_f)*f_col*l_pn)/W + l_pn;

d_p   = pp_G - pp_0;                 % + tip alone
d_n   = pn_G - pn_0;                 % - tip alone
d_avg = 0.5*(pp_G + pn_G) - 0.5*(pp_0 + pn_0);

% ---- the same pair average in closed form, arbitrary gains --------
d_cf = alpha_M*p_off/(1+alpha_M) ...
     + (alpha_M - f_col*(alpha_M - alpha_f)/W) ...
       *(l_pp - l_pn)/(2*(1+alpha_M));

co = [0.850 0.325 0.098;             % + tip
      0.000 0.447 0.741;             % - tip
      0.466 0.674 0.188];            % pair average

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 3.5 3.1]);
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on')

plot(ax, 1e3*p_off, 1e3*d_p, '--', 'Color', co(1,:), 'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*d_n, '--', 'Color', co(2,:), 'LineWidth', 1.2)
plot(ax, 1e3*p_off, 1e3*d_avg, '-', 'Color', co(3,:), 'LineWidth', 1.8)
ii = 1:4:numel(p_off);
plot(ax, 1e3*p_off(ii), 1e3*d_cf(ii), 'o', 'Color', co(3,:), ...
     'MarkerSize', 3.8, 'LineWidth', 0.9)
plot(ax, 1e3*p_off, 1e3*(0.5*(pp_G + pn_G) - p_off), 'k:', ...
     'LineWidth', 1.0)               % rail: GE inversion is exact

xlabel(ax, 'true $p_{\mathrm{off}}$ [mm]', 'Interpreter', 'latex', ...
       'FontSize', 9)
ylabel(ax, ['$\hat{p}_{\mathrm{GE}} - \hat{p}_{\mathrm{no\,GE}}$ ' ...
            '[mm]'], 'Interpreter', 'latex', 'FontSize', 9)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 9, 'GridAlpha', 0.15)
xlim(ax, [-20 20]);  ylim(ax, [-7 7])

legend(ax, {'$+$ tip alone', '$-$ tip alone', 'pair average', ...
            'closed form', 'GE inversion $-\ p_{\mathrm{off}}$'}, ...
       'Interpreter', 'latex', 'FontSize', 7.5, 'NumColumns', 2, ...
       'Location', 'southoutside', 'Box', 'off')

exportgraphics(fig, 'offset_ge_theory.png', 'Resolution', 600);
fprintf('figure -> offset_ge_theory.png\n');

% ---- run-time checks ---------------------------------------------
fprintf('GE inversion recovers p_off : max err %.2e m\n', ...
        max(abs(0.5*(pp_G + pn_G) - p_off)));
fprintf('pair average vs closed form : max err %.2e m\n', ...
        max(abs(d_avg - d_cf)));
if abs(alpha_f - alpha_M) < eps
    eq120 = alpha_M*(Mp + Mn)/(2*W) ...
          + alpha_M*f_col*(l_pp - l_pn)/(2*W);
    fprintf('pair average vs Eq. (120)   : max err %.2e m\n', ...
            max(abs(d_avg - eq120)));
end
fprintf(['slope %.5f (= alpha_M/(1+alpha_M), thrust channel absent); ' ...
         'at p = 0: %+.3f mm\n'], ...
        alpha_M/(1+alpha_M), 1e3*d_avg(p_off == 0));
fprintf('pair average at p = -20 / +20 mm : %+.3f / %+.3f mm\n', ...
        1e3*d_avg(1), 1e3*d_avg(end));
fprintf('single direction at p = 0        : %+.3f / %+.3f mm\n', ...
        1e3*d_p(p_off == 0), 1e3*d_n(p_off == 0));
