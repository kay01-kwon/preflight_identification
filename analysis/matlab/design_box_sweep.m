% DESIGN_BOX_SWEEP  The design-box sweep of the residual bound and the
%   critical-moment shift ceiling, as two single-column figures.
%
%   MATLAB port of analysis/design_range_chart.py (same numbers).  For
%   every combination in the vehicle design box
%
%       mass   3.000 - 3.220 kg   (5)      l_p    0.110 - 0.140 m  (4)
%       p_off  -0.020 - 0.020 m   (5)      z_CoM  0.20  - 0.30  m  (6)
%       J_CoM  0.050 kg m^2 fixed          7 protocol rates
%
%   with phi_max = 5 deg, beta_M = -0.03446 and shape safety 1.05, the
%   window is tilt-limited (sinh x - x = phi_max W z C2 / Mdot) and the
%   script evaluates, all closed form:
%
%     Fig. 1  the model term of the residual cap (20):
%             (M2 rho2_dot + M1 rho1_dot)/(W z) + Delta_pre   [deg/s]
%     Fig. 2  the critical-moment shift ceiling of (21), artanh exact
%             with rho_bar inside:  (Mdot/C2) artanh(rho_bar C2/Mdot),
%             against its small-u limit rho_bar               [mN m]
%
%   Each figure is exported separately with exportgraphics to PNG.
%   Companion to docs/access_tight_rms_bound.tex, Sec. 9.

clear; close all; clc

% ------------------------------------------------------------------ config
G       = 9.81;                       % m/s^2
Jcom    = 0.050;                      % kg m^2, inertia about the CoM
beta_M  = 0.03446;                    % GE moment coefficient (magnitude)
phi_max = deg2rad(5);                 % rad, excitation tilt cap
safety  = 1.05;                       % shape-transfer safety factor
rates   = [0.10 0.20 0.30 0.45 0.65 0.90 1.20];   % N m/s
mass    = linspace(3.000, 3.220, 5);  % kg
lp      = linspace(0.110, 0.140, 4);  % m
poff    = linspace(-0.020, 0.020, 5); % m
zcom    = linspace(0.20, 0.30, 6);    % m

nr = numel(rates);
nc = numel(mass)*numel(lp)*numel(poff)*numel(zcom);
rms_b = zeros(nc, nr);                % residual model bound   [deg/s]
dmc   = zeros(nc, nr);                % shift ceiling, exact   [mN m]
ceilr = zeros(nc, nr);                % small-u limit rho_bar  [mN m]
Wv    = zeros(nc, 1);                 % per-combo weight       [N]
armv  = zeros(nc, 1);                 % per-combo arm l_p-p_off [m]
zv    = zeros(nc, 1);                 % per-combo z_CoM        [m]

% ------------------------------------------------------------------ sweep
ic = 0;
for m = mass
    for l = lp
        for p = poff
            for z = zcom
                ic = ic + 1;
                W   = m*G;
                arm = l - p;
                Wv(ic) = W;  armv(ic) = arm;  zv(ic) = z;
                wz  = W*z;
                jp  = Jcom + m*(z^2 + arm^2);
                c2  = sqrt(wz/jp);
                for ir = 1:nr
                    md  = rates(ir);
                    % tilt-limited window: phi_nom(tau_end) = phi_max
                    rhs = phi_max*wz*c2/md;
                    x   = fzero(@(q) sinh(q) - q - rhs, [0.05 25]);
                    T   = x/c2;
                    dmw = md*T;
                    % rho_bar: Wz term dropped by sign (a >= z tan(phi/2))
                    rb  = (1/7)*0.5*W*arm*phi_max^2 ...
                        + (1/5)*beta_M*dmw*phi_max;
                    % true end-rate anchor: nominal end rate + envelope
                    om  = md*(cosh(x) - 1)/wz + rb*sinh(x)/(jp*c2);
                    rd2 = safety*(W*arm*phi_max)*om;
                    rd1 = beta_M*(md*phi_max + dmw*om);
                    mdl = (M2(x)*rd2 + M1(x)*rd1)/wz;
                    % pre-onset term (18)
                    c1   = md/wz;
                    beta = rb/(jp*c2);
                    dt   = atanh(min(beta/c1, 0.99))/c2;
                    a    = c2*dt;
                    I    = 1.5*dt + sinh(2*a)/(4*c2) - 2*sinh(a)/c2;
                    dpre = c1*sqrt(max(I, 0)/T);
                    rms_b(ic, ir) = rad2deg(mdl + dpre);
                    dmc(ic, ir)   = 1e3*md*dt;   % = (Mdot/C2) artanh(u)
                    ceilr(ic, ir) = 1e3*rb;
                end
            end
        end
    end
end

lo_r = min(rms_b);  hi_r = max(rms_b);  md_r = median(rms_b);
lo_d = min(dmc);    hi_d = max(dmc);    md_d = median(dmc);
lo_c = min(ceilr);  hi_c = max(ceilr);

cA = [0.482 0.196 0.580];             % purple, residual bound
cB = [0.157 0.455 0.651];             % blue,   shift ceiling
cC = [0.878 0.510 0.078];             % orange, rho_bar limit

% ------------------------------------ Fig. 1: residual model bound
f1 = figure('Color', 'w', 'Units', 'inches', ...
            'Position', [1 1 3.5 2.7]);               % IEEE single column
a1 = axes(f1); hold(a1, 'on'); box(a1, 'on'); grid(a1, 'on');

hb = fill(a1, [rates fliplr(rates)], [lo_r fliplr(hi_r)], cA, ...
          'FaceAlpha', 0.16, 'EdgeColor', cA, 'EdgeAlpha', 0.45, ...
          'LineWidth', 0.5);
hm = plot(a1, rates, md_r, '-o', 'Color', cA, ...
          'MarkerFaceColor', cA, 'LineWidth', 1.2, 'MarkerSize', 4.5);

% the corners that set the band ends at the fastest rate, annotated
% (two lines each, right-aligned under/over the band end, clear of
%  the legend in the upper left)
[~, imx] = max(rms_b(:, end));
[~, imn] = min(rms_b(:, end));
text(a1, rates(end), hi_r(end) + 0.10*max(hi_r), ...
     {sprintf('box max: $l_p{-}p_{\\mathrm{off}}=%.3f$ m,', armv(imx)), ...
      sprintf('$W=%.1f$ N, $z_{CoM}=%.2f$ m', Wv(imx), zv(imx))}, ...
     'Interpreter', 'latex', 'FontSize', 7, 'Color', cA, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
text(a1, rates(end), lo_r(end) - 0.04*max(hi_r), ...
     {sprintf('box min: $l_p{-}p_{\\mathrm{off}}=%.3f$ m,', armv(imn)), ...
      sprintf('$W=%.1f$ N, $z_{CoM}=%.2f$ m', Wv(imn), zv(imn))}, ...
     'Interpreter', 'latex', 'FontSize', 7, 'Color', cA, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

set(a1, 'XScale', 'log', 'XTick', rates, ...
        'XTickLabel', compose('%.2f', rates), 'XMinorTick', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 9, ...
        'GridAlpha', 0.15, 'LineWidth', 0.6, 'Layer', 'top');
xlim(a1, [0.093 1.29]);
ylim(a1, [0, 1.42*max(hi_r)]);
xlabel(a1, '$\dot{M}$ [N$\,$m/s]', 'Interpreter', 'latex', 'FontSize', 9);
ylabel(a1, 'Bound on RMS$(\delta e_\omega)$ [$^\circ$/s]', ...
       'Interpreter', 'latex', 'FontSize', 9);
legend(a1, [hm hb], {'design-box median', 'box min--max'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', ...
       'FontSize', 8, 'Box', 'off');

exportgraphics(f1, 'design_box_rms.png', 'Resolution', 600);
fprintf('figure -> design_box_rms.png\n');

% ------------------------------------ Fig. 2: shift ceiling, artanh exact
f2 = figure('Color', 'w', 'Units', 'inches', ...
            'Position', [1 1 3.5 2.7]);
a2 = axes(f2); hold(a2, 'on'); box(a2, 'on'); grid(a2, 'on');

hb2 = fill(a2, [rates fliplr(rates)], [lo_d fliplr(hi_d)], cB, ...
           'FaceAlpha', 0.16, 'EdgeColor', cB, 'EdgeAlpha', 0.45, ...
           'LineWidth', 0.5);
hm2 = plot(a2, rates, md_d, '-o', 'Color', cB, ...
           'MarkerFaceColor', cB, 'LineWidth', 1.2, 'MarkerSize', 4.5);
plot(a2, rates, lo_c, '--', 'Color', cC, 'LineWidth', 1.1);
hc2 = plot(a2, rates, hi_c, '--', 'Color', cC, 'LineWidth', 1.1);

% the corners that set the band ends at the slowest rate, annotated
% (left side, clear of the legend in the lower right)
[~, jmx] = max(dmc(:, 1));
[~, jmn] = min(dmc(:, 1));
text(a2, rates(1), hi_d(1) + 0.06*max(hi_d), ...
     {sprintf('box max: $l_p{-}p_{\\mathrm{off}}=%.3f$ m,', armv(jmx)), ...
      sprintf('$W=%.1f$ N, $z_{CoM}=%.2f$ m', Wv(jmx), zv(jmx))}, ...
     'Interpreter', 'latex', 'FontSize', 7, 'Color', cB, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
text(a2, rates(1), lo_d(1) - 0.05*max(hi_d), ...
     {sprintf('box min: $l_p{-}p_{\\mathrm{off}}=%.3f$ m,', armv(jmn)), ...
      sprintf('$W=%.1f$ N, $z_{CoM}=%.2f$ m', Wv(jmn), zv(jmn))}, ...
     'Interpreter', 'latex', 'FontSize', 7, 'Color', cB, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

set(a2, 'XScale', 'log', 'XTick', rates, ...
        'XTickLabel', compose('%.2f', rates), 'XMinorTick', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 9, ...
        'GridAlpha', 0.15, 'LineWidth', 0.6, 'Layer', 'top');
xlim(a2, [0.093 1.29]);
ylim(a2, [0, 1.30*max(hi_d)]);
xlabel(a2, '$\dot{M}$ [N$\,$m/s]', 'Interpreter', 'latex', 'FontSize', 9);
ylabel(a2, '$|\Delta M_{\mathrm{crit}}|$ [mN$\,$m]', ...
       'Interpreter', 'latex', 'FontSize', 9);
legend(a2, [hm2 hb2 hc2], ...
       {['ceiling $(\dot{M}/C_2)\,\mathrm{artanh}' ...
         '(\bar{\rho}C_2/\dot{M})$, median'], ...
        'box min--max', ...
        'small-$u$ limit $\bar{\rho}$ (box min/max)'}, ...
       'Interpreter', 'latex', 'Location', 'southeast', ...
       'FontSize', 7.5, 'Box', 'off');

exportgraphics(f2, 'design_box_shift.png', 'Resolution', 600);
fprintf('figure -> design_box_shift.png\n');

% ------------------------------------------------------------- report
fprintf('\nDesign box sweep: %d combinations x %d rates\n', nc, nr);
fprintf('%6s %22s %22s %14s\n', 'Mdot', 'RMS bound [deg/s]', ...
        'shift ceiling [mN m]', 'rho_bar');
fprintf('%6s %22s %22s %14s\n', '', 'min   med   max', ...
        'min   med   max', 'min - max');
for ir = 1:nr
    fprintf('%6.2f %7.2f%7.2f%7.2f  %7.2f%7.2f%7.2f  %6.1f -%5.1f\n', ...
            rates(ir), lo_r(ir), md_r(ir), hi_r(ir), ...
            lo_d(ir), md_d(ir), hi_d(ir), lo_c(ir), hi_c(ir));
end

% ------------------------------------------------- shape ceilings (17a/b)
function v = M2(x)
    u = linspace(-x, 0, 1501);
    b = (sinh(u + x) + exp(-2*x)*sinh(-u))/sinh(x) - exp(2*u);
    v = max(abs(b))/3;
end

function v = M1(x)
    u = linspace(-x, 0, 1501);
    b = (x/2)*sinh(u + x)/sinh(x) - (u + x).*exp(u)/2;
    v = max(abs(b));
end
