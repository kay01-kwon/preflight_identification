% MDOT_FLOOR  Guaranteed lower limit on the ramp rate, over the design box.
%
%   The witness of the residual bound is admissible only while
%
%       u = rho_bar * C2 / Mdot < 1        i.e.   Mdot > rho_bar * C2,
%
%   so the admissibility number of the bound doubles as an a priori LOWER
%   limit on the ramp rate.  Everything it needs is known before the
%   experiment: the coarse design box (mass, arm, z_CoM, J_CoM) and the
%   tilt cap phi_max.  This script evaluates that floor across the arm
%   range for three CoM heights.
%
%   Conservative variant: rho_max (the peak remainder) is used in place of
%   the window average rho_bar.  Chebyshev's integral inequality gives
%   rho_bar <= rho_max/7 on the gravity channel, so the floor drawn here is
%   about seven times above the sharp one -- deliberately, since a rate
%   schedule wants margin, not sharpness.
%
%   Companion to docs/access_tight_rms_bound.tex, eq. (9a).

clear; close all; clc

% ------------------------------------------------------------------ config
m       = 3.220;                    % kg,  worst-case (heaviest) mass
g       = 9.81;                     % m/s^2
W       = m*g;                      % N,   weight
l_arm   = 0.090:0.010:0.160;        % m,   pivot arm, l_p - p_off
Jcom    = 0.050;                    % kg m^2, inertia about the CoM
phi_max = deg2rad(5);               % rad, tilt cap of the excitation design
z       = [0.20; 0.25; 0.30];       % m,   CoM height
Mdot_slow = 0.10;                   % N m/s, slowest rate flown

N  = numel(l_arm);
Nz = numel(z);

% ----------------------------------------------------------- the floor
% Peak gravity-linearisation remainder, quadratic order:
rho_max  = 0.5*W*l_arm*phi_max^2;                       % N m
Mdot_min = zeros(N, Nz);
for j = 1:Nz
    for i = 1:N
        % parallel-axis inertia about the pivot, and the growth rate
        Jp   = Jcom + m*(l_arm(i)^2 + z(j)^2);          % kg m^2
        C2   = sqrt(W*z(j)/Jp);                         % 1/s
        Mdot_min(i,j) = rho_max(i)*C2;                  % N m/s
    end
end

% ------------------------------------------------------------- figure
fig = figure('Color', 'w', 'Units', 'inches', ...
             'Position', [1 1 3.5 2.7]);               % IEEE single column
ax  = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

co  = [0.000 0.447 0.741;      % blue
       0.850 0.325 0.098;      % orange
       0.466 0.674 0.188];     % green
mk  = {'o', 's', '^'};

h = gobjects(Nz+1, 1);
for j = 1:Nz
    h(j) = plot(ax, l_arm, Mdot_min(:,j), ['-' mk{j}], ...
                'Color', co(j,:), 'MarkerFaceColor', co(j,:), ...
                'LineWidth', 1.2, 'MarkerSize', 4.5);
end
h(Nz+1) = plot(ax, l_arm([1 end]), Mdot_slow*[1 1], 'k--', 'LineWidth', 1.0);

xlabel(ax, 'Pivot arm $a = l_p - p_{\mathrm{off}}$ [m]', ...
       'Interpreter', 'latex', 'FontSize', 9);
ylabel(ax, '$\dot{M}_{\min} = \rho_{\max}\,C_2$ [N$\,$m/s]', ...
       'Interpreter', 'latex', 'FontSize', 9);
legend(ax, h, {'$z_{CoM} = 0.20$ m', '$z_{CoM} = 0.25$ m', ...
               '$z_{CoM} = 0.30$ m', 'slowest rate flown'}, ...
       'Interpreter', 'latex', 'Location', 'southeast', ...
       'FontSize', 8, 'Box', 'off');
set(ax, 'FontName', 'Times New Roman', 'FontSize', 9, ...
        'GridAlpha', 0.15, 'LineWidth', 0.6, 'Layer', 'top');
xlim(ax, [l_arm(1)-0.004, l_arm(end)+0.004]);
ylim(ax, [0, 1.18*max([Mdot_min(:); Mdot_slow])]);

if exist('exportgraphics', 'file')
    exportgraphics(fig, 'mdot_floor.pdf', 'ContentType', 'vector');
    fprintf('figure -> mdot_floor.pdf\n');
end

% ------------------------------------------------------------- report
fprintf('\nGuaranteed ramp-rate floor, m = %.3f kg, phi_max = %g deg\n', ...
        m, rad2deg(phi_max));
fprintf('%8s', 'a [m]'); fprintf('%12s', 'z=0.20', 'z=0.25', 'z=0.30');
fprintf('\n');
for i = 1:N
    fprintf('%8.3f%12.4f%12.4f%12.4f\n', l_arm(i), Mdot_min(i,:));
end
fprintf(['\nbox worst (largest floor): %.4f N m/s at a = %.3f m, ' ...
         'z = %.2f m\n'], max(Mdot_min(:)), l_arm(end), z(1));
fprintf('margin of the slowest rate flown: %.2fx\n\n', ...
        Mdot_slow/max(Mdot_min(:)));
