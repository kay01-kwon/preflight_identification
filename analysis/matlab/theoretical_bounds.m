function theoretical_bounds()
% THEORETICAL_BOUNDS  A priori error bounds for the COSH onset method.
%
%   Closed-form bounds on (i) the shape deviation of the angular rate and
%   (ii) the identified critical moment, produced by the two model
%   truncation channels:
%
%     rho_phi = W a (1 - cos dphi) + W z_CoM (sin dphi - dphi)   [gravity]
%     rho_b   = beta_M * dM_win * dphi                           [GE bilinear]
%
%   No dataset, no moment of inertia, and -- in the normalised forms --
%   no calibrated constants at all.
%
%   THE CHAIN
%   ---------
%   The deviation obeys  J_P e'' - D e = rho  with zero onset conditions,
%   so e is the Duhamel integral alone.  Two facts close everything:
%
%     (a) the convolution kernel cosh(C2 (tau-s)) DECREASES in s, and
%         both rho channels INCREASE in tau, so Chebyshev's integral
%         inequality replaces max|rho| by the TIME AVERAGE rho_bar;
%     (b) the onset sensitivity reduces to a normalised weighted average,
%         dM_crit = -int rho w,  with w >= 0 and int w = 1,  w decreasing,
%         so the same inequality gives |dM_crit| <= rho_bar.
%
%   With x = C2 * tau_end and R(x) = rho_bar / rho_max,
%
%     |dM_crit|              <= R(x) rho_max          <= (1/7) rho_phi
%                                                      + (1/5) rho_b
%     |dw|_end / w_end       <= Phi(x) rho_max/dM_win <= (1/2) rho_phi/dM_win
%                                                      + (1)   rho_b/dM_win
%     |dw|_end     [rad/s]   =  R(x) rho_max * K * C2 * sinh(x)
%
%   where Phi(x) = x R(x) coth(x/2).  The first two are uniform in x, so
%   they need neither C2 nor J_P; only the absolute rate bound does, and
%   there C2 and K are calibrated quantities, not computed ones.
%
%   Usage:  theoretical_bounds
%
%   Companion to analysis/error_budget.py (the measured counterpart).

% ----------------------------------------------------------------- config
c.W        = 31.59;              % N, loaded weight (3.220 kg)
c.lp       = struct('roll', 0.140, 'pitch', 0.110);   % m, pivot offsets
c.lam_off  = 0.020;              % m, worst |CoM offset| added to the arm
c.zCoM     = 0.20;               % m, WORST case (smallest -> largest rho)
c.beta_M   = 0.0345;             % 1/rad, GE moment-coefficient slope
c.dM_win   = 1.30;               % N.m, commanded moment increment over the
                                 %      fit window   <-- CHECK against data
% Fit-window excursions.  NOTE: the implementation truncates the window by
% MOMENT, not by angle, so 5 deg is the excitation-DESIGN cap, not what the
% windows realise.  The operative figures are the measured excursions.
c.caps_deg = [5 6.1 9.4 10];     % design cap | measured median | measured
                                 % max | largest absolute tilt
c.x_meas   = [1.08 2.67 5.20];   % measured C2*tau_end: min, median, max
c.C2_rep   = 5.0;                % rad/s, representative calibrated C2
c.K_rep    = 0.19;               % 1/(N.m), representative calibrated K

fprintf('\n=== A priori bounds, COSH onset method ===\n');
fprintf('W = %.2f N, z_CoM = %.2f m (worst), lambda_off = %.3f m\n', ...
        c.W, c.zCoM, c.lam_off);
fprintf('beta_M = %.4f 1/rad, dM_win = %.2f N.m\n\n', c.beta_M, c.dM_win);

% ------------------------------------------------- 1. shape factors R, Phi
check_series_and_quadrature();
report_limits();

% ------------------------------------------------------- 2. cross table
axes_ = {'roll', 'pitch'};
fprintf(['\n%-6s %-6s %8s %9s %9s | %10s %10s | %9s %9s\n'], ...
        'axis', 'cap', 'a [m]', 'rho_phi', 'rho_b', 'dM_crit', ...
        'dCoM', 'dw/w', 'dw');
fprintf(['%-6s %-6s %8s %9s %9s | %10s %10s | %9s %9s\n'], ...
        '', '[deg]', '', '[mN.m]', '[mN.m]', '[mN.m]', '[mm]', ...
        '[%]', '[mrad/s]');
fprintf('%s\n', repmat('-', 1, 92));

for ia = 1:numel(axes_)
    ax = axes_{ia};
    a  = c.lp.(ax) + c.lam_off;
    for cap = c.caps_deg
        dphi = deg2rad(cap);
        rp   = rho_phi_max(c.W, a, c.zCoM, dphi);       % N.m
        rb   = c.beta_M * c.dM_win * dphi;              % N.m

        % (i) deliverable bound, uniform in x  -> no C2, no J_P
        dM   = rp/7 + rb/5;                             % N.m
        dCoM = dM / c.W * 1e3;                          % mm

        % (ii) relative shape deviation, uniform in x -> no C2, no J_P
        drel = (0.5*rp + 1.0*rb) / c.dM_win * 100;      % %

        % (iii) absolute shape deviation at the worst measured x
        %       (this one does use the CALIBRATED C2, K -- still no J_P)
        x    = min(c.x_meas);                           % smallest x = largest R
        rbar = R_sa(x)*rp + R_b(x)*rb;
        dw   = rbar * c.K_rep * c.C2_rep * sinh(x) * 1e3;   % mrad/s

        fprintf('%-6s %-6d %8.3f %9.1f %9.2f | %10.2f %10.3f | %9.2f %9.1f\n', ...
                ax, cap, a, rp*1e3, rb*1e3, dM*1e3, dCoM, drel, dw);
    end
end

fprintf(['\nBounds (i) and (ii) are uniform in x = C2*tau_end: they use\n' ...
         'R_sa <= 1/7, R_b <= 1/5, Phi_sa <= 1/2, Phi_b <= 1, so neither\n' ...
         'C2 nor J_P enters.  Only (iii) uses the calibrated (C2, K).\n']);

% ----------------------------------------------------------- 3. figures
plot_shape_factors(c);

end % =====================================================================


function r = rho_phi_max(W, a, z, dphi)
% Exact gravity linearisation remainder at excursion dphi.
% Leading orders:  +W a dphi^2/2  -  W z dphi^3/6.  The cubic is negative,
% so dropping it recovers the familiar quadratic bound conservatively, and
% the whole z_CoM box moves rho_phi by only a few percent.
r = W*a*(1 - cos(dphi)) + W*z*(sin(dphi) - dphi);
end


% ------------------------------------------------------------------------
%  Shape factors R(x) = rho_bar / rho_max.
%
%  rho_sa ~ (sinh v - v)^2   (order tau^6 at the anchor) -> R -> 1/7
%  rho_b  ~ v (sinh v - v)   (order tau^4 at the anchor) -> R -> 1/5
%
%  The closed forms below suffer heavy cancellation for small x, so a
%  series is used there; check_series_and_quadrature() validates
%  against adaptive quadrature.
% ------------------------------------------------------------------------

function R = R_sa(x)
R = zeros(size(x));
for k = 1:numel(x)
    xk = x(k);
    if xk < 0.5
        R(k) = (1/7) * (1 - xk^2/45);          % series, cancellation-free
    elseif xk > 50
        R(k) = 1/(2*xk);                       % asymptote, overflow-free
    else
        num  = 0.5*sinh(xk)*cosh(xk) - xk/2 - 2*xk*cosh(xk) ...
               + 2*sinh(xk) + xk^3/3;
        R(k) = num / (xk * (sinh(xk) - xk)^2);
    end
end
end


function R = R_b(x)
R = zeros(size(x));
for k = 1:numel(x)
    xk = x(k);
    if xk < 0.5
        R(k) = (1/5) * (1 - xk^2/70);
    elseif xk > 50
        R(k) = (xk - 1)/xk^2;                  % asymptote, overflow-free
    else
        num  = xk*cosh(xk) - sinh(xk) - xk^3/3;
        R(k) = num / (xk^2 * (sinh(xk) - xk));
    end
end
end


function P = Phi_sa(x), P = x .* R_sa(x) .* coth(x/2); end
function P = Phi_b(x),  P = x .* R_b(x)  .* coth(x/2); end


function check_series_and_quadrature()
% Cross-check the closed forms against adaptive quadrature of the
% defining integrals, and verify monotonicity of R and Phi.
xs = logspace(-1, 1, 60);
q_sa = zeros(size(xs)); q_b = zeros(size(xs));
for k = 1:numel(xs)
    xk = xs(k);
    q_sa(k) = integral(@(v)(sinh(v)-v).^2, 0, xk) / (xk*(sinh(xk)-xk)^2);
    q_b(k)  = integral(@(v) v.*(sinh(v)-v), 0, xk) / (xk^2*(sinh(xk)-xk));
end
e_sa = max(abs(R_sa(xs) - q_sa) ./ q_sa);
e_b  = max(abs(R_b(xs)  - q_b ) ./ q_b );
fprintf('closed form vs quadrature (x in [0.1, 10]):\n');
fprintf('   max rel. error   R_sa %.2e    R_b %.2e\n', e_sa, e_b);

mono = @(y) all(diff(y) < 0);
fprintf('   R_sa decreasing  %d      R_b decreasing  %d\n', ...
        mono(R_sa(xs)), mono(R_b(xs)));
fprintf('   Phi_sa increasing %d     Phi_b increasing %d\n', ...
        all(diff(Phi_sa(xs)) > 0), all(diff(Phi_b(xs)) > 0));
end


function report_limits()
xs = [1e-3 1.08 2.67 5.20 1e3];   % last two branches are asymptotic
fprintf('\n%8s %10s %10s %10s %10s\n', 'x', 'R_sa', 'R_b', 'Phi_sa', 'Phi_b');
for x = xs
    fprintf('%8.3g %10.4f %10.4f %10.4f %10.4f\n', ...
            x, R_sa(x), R_b(x), Phi_sa(x), Phi_b(x));
end
fprintf(['limits:  R_sa 1/7=%.4f -> 1/(2x),   R_b 1/5=%.4f -> (x-1)/x^2\n' ...
         '         Phi_sa 2/7=%.4f -> 1/2,     Phi_b 2/5=%.4f -> 1\n'], ...
        1/7, 1/5, 2/7, 2/5);
end


function plot_shape_factors(c)
x = linspace(0.05, 8, 400);
figure('Color', 'w', 'Position', [100 100 900 340]);

subplot(1,2,1); hold on; box off;
plot(x, R_sa(x), 'LineWidth', 1.6);
plot(x, R_b(x),  'LineWidth', 1.6);
yline(1/7, ':'); yline(1/5, ':');
for xm = c.x_meas, xline(xm, 'Color', [.8 .8 .8]); end
xlabel('x = C_2 \tau_{end}'); ylabel('R(x) = $\bar\rho/\rho_{max}$', ...
       'Interpreter', 'latex');
legend({'gravity  (\rightarrow 1/7)', 'bilinear (\rightarrow 1/5)'}, ...
       'Location', 'northeast'); legend boxoff;
title('deliverable factor');

subplot(1,2,2); hold on; box off;
plot(x, Phi_sa(x), 'LineWidth', 1.6);
plot(x, Phi_b(x),  'LineWidth', 1.6);
yline(0.5, ':'); yline(1.0, ':');
for xm = c.x_meas, xline(xm, 'Color', [.8 .8 .8]); end
xlabel('x = C_2 \tau_{end}'); ylabel('\Phi(x) = x R(x) coth(x/2)');
legend({'gravity  (\rightarrow 1/2)', 'bilinear (\rightarrow 1)'}, ...
       'Location', 'southeast'); legend boxoff;
title('shape-deviation factor');

if exist('exportgraphics', 'file')
    exportgraphics(gcf, 'theoretical_bounds.pdf', 'ContentType', 'vector');
    fprintf('\nfigure -> theoretical_bounds.pdf\n');
end
end
