%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Endogenous Labor Supply %%%%%%%%%%%%%%%%%%%
%%
clear; clc;

%% STEP 1: Parameters
beta   = 0.96; r   = 0.04; varphi    = 2.0; rho    = 0.9; sigma  = 0.15;
gamma  = 1; tol    = 1e-9; maxits = 1e7; w_bar  = 2; small_positive = 1e-4;

%% STEP 2: Steady-State Labor Calibration
n_ss   = 40 / 168;
c_ss   = w_bar * n_ss; % At SS, r=0 & a_prime=a
Omega = (c_ss) / (n_ss^(1 + 1/varphi)) * (1 + 1/varphi);  % Labor disutility scale

%% STEP 3: Grid Setup
c_dim = 5;     % income states
r_dim = 50;    % asset grid
a_dim = r_dim; % future asset choices
n_dim = 60;    % labor grid, also Assets evolve slowly than Labor

%% STEP 4: Tauchen Discretization
mu = 0; m = 3;
[Z, P] = Tauchen(c_dim, mu, rho, sigma, m);
Y = exp(Z);  % income levels as model uses log utility, income requires >0

%% STEP 5: Asset Grid (Labor-Adjusted Borrowing Limit)
n_grid = linspace(0.01, 0.25, n_dim);  % feasible labor supply
w_min = exp(Z(1));                    % worst-case wage
n_max = max(n_grid);                  % max feasible labor
a_min = - (w_min * n_max) / r;        % labor-adjusted borrowing limit
a_max = 4 * exp(Z(end));              % generous upper bound
a = linspace(a_min, a_max, r_dim)';
a_prime = a;                          % future asset choices

%% STEP 6: Mesh and Initialization
[A, Zgrid] = ndgrid(a, Z);                          % Asset × log-income grid
w1 = A + Zgrid;                                     % Total resources in log-space
c1 = max(w1 - small_positive, small_positive);      % Initial guess for consumption
c2 = ones(r_dim, c_dim);                            % Placeholder for updated consumption
w_orig = linspace(min(a) + min(Z), max(a) + max(Z), r_dim)';  % Expanded domain

%% STEP 7: Initial Utility and Value Function
disutility_ss = Omega * (n_ss^(1 + 1/varphi)) / (1 + 1/varphi);  % disutility at steady-state labor
Y_grid = w_bar * Y;  % wage × income state
net_c_grid = Y_grid * n_ss - disutility_ss;  % net consumption at steady-state labor

utility = log(max(net_c_grid, 1e-8));  % ensure feasibility, size = [1 × c_dim]

V0 = repmat(utility', r_dim, 1);  %  % replicate across asset grid
V = V0;
V_new = V;

%% STEP 8: Value Function Iteration
policy_c = Mohi_zeros(r_dim, c_dim);  % Optimal consumption
policy_n = Mohi_zeros(r_dim, c_dim);  % Optimal labor
policy_a = Mohi_zeros(r_dim, c_dim);  % Optimal savings

tic
for iter = 1:maxits
    for iy = 1:c_dim
        y = Y(iy);  % income state
        for ia = 1:r_dim
            a_now = a(ia);  % current asset
            max_val = -inf;
            best_n = NaN;
            best_idx = NaN;

            for in = 1:n_dim
                n = n_grid(in);  % labor choice
                disutility = Omega * (n^(1 + 1/varphi)) / (1 + 1/varphi);

                % Future consumption for each asset choice
                c_prime = a_now + w_bar * n - a_prime / (1 + r);
                net_c_prime = c_prime - disutility;

                % Clamp to avoid log(0) or negative
                net_c_prime(net_c_prime <= small_positive) = small_positive;
                u_vec = log(net_c_prime);  % utility

                % Expected future value
                EV_vec = V * P(iy, :)';  % [r_dim × c_dim] × [c_dim × 1] → [r_dim × 1]
                EV_vec(isnan(EV_vec) | isinf(EV_vec)) = -1e10;

                % Total value
                value = u_vec + beta * EV_vec;

                % Find best future asset choice
                [val, idx] = max(value);
                if val > max_val
                    max_val = val;
                    best_n = n;
                    best_idx = idx;
                end
            end

            if isfinite(max_val)
                policy_c(ia, iy) = max(c_prime(best_idx), small_positive);
                policy_n(ia, iy) = best_n;
                policy_a(ia, iy) = a_prime(best_idx);
            else
                warning('No feasible labor for ia=%d, iy=%d', ia, iy);
                policy_c(ia, iy) = small_positive;
                policy_n(ia, iy) = 0;
                policy_a(ia, iy) = a(ia);  % stay put
            end

            V_new(ia, iy) = max_val;
        end
    end

    % Convergence check
    dif = max(max(abs(V_new - V)));
    V = V_new;

    if mod(iter, 100) == 0
        fprintf('Iter %d, diff = %.2e\n', iter, dif);
    end

    if dif < tol
        fprintf('Converged at iter %d with diff = %.2e\n', iter, dif);
        break;
    end
end
toc
%% STEP 9: Diagnostic Check
if any(policy_n(:) == 0)
    warning('Some labor choices remain zero — check feasibility or grid resolution.');
end

%% STEP 10: Display Results
fprintf('\n=== VFI with Endogenous Labor (Labor-Adjusted Borrowing Limit) ===\n');
fprintf('Iterations: %d\n', iter);
fprintf('Final diff: %.2e\n', dif);

disp('Sample Value Function V(a, y):');
disp(V(1:5, :));

disp('Sample Consumption Policy c(a, y):');
disp(policy_c(1:5, :));

disp('Sample Labor Policy n(a, y):');
disp(policy_n(1:5, :));

disp('Sample Asset Policy a''(a, y):');
disp(policy_a(1:5, :));

%% STEP 11: Plot Value Function
figure;
surf(a, Y, V');
xlabel('Assets (a)');
ylabel('Income (y)');
zlabel('Value Function V(a, y)');
title('Converged Value Function');
colormap('parula');
shading interp;
view(45, 30);

%% STEP 1: Simulation Setup
T = 1000; burn_in = 500;
rng(123);  % reproducibility

% Simulate log income process using AR(1)
Z_sim = Mohi_zeros(T, 1);
Z_sim(1) = Z(round(c_dim / 2));  % start from middle income state

for t = 2:T
    eps_t = sigma * randn;
    Z_sim(t) = (1 - rho) * log(w_bar) + rho * Z_sim(t-1) + eps_t;
end

w_sim = exp(Z_sim);  % convert to income levels

%% STEP 2: Initialize Paths
a_sim = Mohi_zeros(T, 1);  % asset path
n_sim = Mohi_zeros(T, 1);  % labor path
c_sim = Mohi_zeros(T, 1);  % consumption path

% Start from middle asset and income index
a_idx = round(r_dim / 2);
w_idx = round(c_dim / 2);

for t = 1:T
    % Match simulated income to closest discrete income state
    [~, w_idx] = min(abs(Y - w_sim(t)));

    % Retrieve optimal policies
    a_next = policy_a(a_idx, w_idx);
    c_now  = policy_c(a_idx, w_idx);
    n_now  = policy_n(a_idx, w_idx);

    % Store simulated paths
    a_sim(t) = a_next;
    c_sim(t) = c_now;
    n_sim(t) = n_now;

    % Update asset index for next period
    [~, a_idx] = min(abs(a - a_next));
end

%% STEP 3: Discard Burn-In
w_plot = w_sim(burn_in+1:end);
a_plot = a_sim(burn_in+1:end);
n_plot = n_sim(burn_in+1:end);
c_plot = c_sim(burn_in+1:end);
T_plot = length(w_plot);

%% STEP 4: Basic Tiled Layout Plot
figure;
tiledlayout(2,2);

nexttile;
plot(1:T_plot, w_plot, 'b');
title('Simulated Income $w_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Income');

nexttile;
plot(1:T_plot, a_plot, 'r');
title('Simulated Assets $a_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Assets');

nexttile;
plot(1:T_plot, n_plot, 'g');
title('Simulated Labor $n_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Labor');

nexttile;
plot(1:T_plot, c_plot, 'k');
title('Simulated Consumption $c_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Consumption');

sgtitle('Simulated Household Paths (Post Burn-In)', 'FontWeight','bold', 'FontSize',14);

%% STEP 5: Enhanced Plot with Styling
figure('Units','normalized','Position',[0.1 0.1 0.8 0.6]);
tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(1:T_plot, w_plot, 'b', 'LineWidth', 1.2);
title('Simulated Income $w_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Income');
grid on; box on;

nexttile;
plot(1:T_plot, a_plot, 'r', 'LineWidth', 1.2);
title('Simulated Assets $a_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Assets');
grid on; box on;

nexttile;
plot(1:T_plot, n_plot, 'g', 'LineWidth', 1.2);
title('Simulated Labor $n_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Labor');
grid on; box on;

nexttile;
plot(1:T_plot, c_plot, 'k', 'LineWidth', 1.2);
title('Simulated Consumption $c_t$', 'Interpreter','latex');
xlabel('Time'); ylabel('Consumption');
grid on; box on;

sgtitle('Simulated Household Paths (Post Burn-In)', 'FontWeight','bold', 'FontSize',14);

%% STEP 1: Calculate Standard Deviation of Simulated Labor
std_n = std(n_plot);
fprintf('\n=== Labor Volatility Diagnostic ===\n');
fprintf('Standard deviation of simulated labor n_t: %.4f\n\n', std_n);

%% STEP 2: Qualitative Scenario Analysis
fprintf('=== Qualitative Effects on std(n_t) ===\n\n');

% (a) Borrowing constraint were zero
fprintf('(a) If the borrowing constraint were zero:\n');
if std_n < 0.05
    fprintf('    → Expected std(n_t): Likely to INCREASE from %.4f\n', std_n);
else
    fprintf('    → Expected std(n_t): May increase slightly from %.4f\n', std_n);
end
fprintf('    → Reason: With perfect consumption smoothing, labor becomes a flexible margin to absorb shocks.\n\n');

% (b) Relative risk aversion doubled
fprintf('(b) If relative risk aversion doubled:\n');
if std_n > 0.03
    fprintf('    → Expected std(n_t): Likely to DECREASE from %.4f\n', std_n);
else
    fprintf('    → Expected std(n_t): Already low at %.4f, may remain stable or decrease slightly.\n', std_n);
end
fprintf('    → Reason: More cautious households prefer smoother consumption, reducing labor volatility.\n\n');

% (c) Frisch elasticity doubled
fprintf('(c) If Frisch labor supply elasticity doubled:\n');
if std_n < 0.05
    fprintf('    → Expected std(n_t): Likely to INCREASE from %.4f\n', std_n);
else
    fprintf('    → Expected std(n_t): Already responsive at %.4f, may increase further.\n', std_n);
end
fprintf('    → Reason: Labor supply becomes more elastic, amplifying responsiveness to wage shocks.\n\n');

% (d) Real wage volatility doubled
fprintf('(d) If real wage volatility doubled:\n');
if std_n < 0.05
    fprintf('    → Expected std(n_t): Likely to INCREASE from %.4f\n', std_n);
else
    fprintf('    → Expected std(n_t): Already volatile at %.4f, may increase further.\n', std_n);
end
fprintf('    → Reason: Wage shocks directly affect labor income, prompting stronger labor adjustments.\n\n');