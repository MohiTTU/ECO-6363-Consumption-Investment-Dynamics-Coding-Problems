%% VARIABLE CAPITAL UTILIZATION MODEL COMPONENTS
% Purpose: Define all primitives for dynamic programming model of firm with variable utilization
%Solving the Bellman equation for an infinitely lived firm:
% V(k, u) = \max_{n \geq 0, inv' \geq 0} \left\{ (uk)^\alpha n^{1-\alpha} - inv' - un + \beta \mathbb{E}[V(k', u')] \right\}
% Model Components
% Production function: f(k, n, u) = (uk)^\alpha n^{1-\alpha}
% Capital law of motion: k' = [1 - \delta(u)]k + inv'
% Depreciation function: \delta(u) = \delta_0 + \phi_1(u - 1)^2
% Stochastic process for utilization: AR(1) process (details assumed from previous section)

clearvars; clc; close all;

%% 1. PARAMETERS
% Preferences and technology
beta    = 0.96;     % Discount factor
alpha   = 0.4;      % Capital share

% Depreciation function parameters
delta_0  = 0.1;      % Baseline depreciation
phi_2    = 0.2;      % Curvature of utilization-dependent depreciation
% Derived parameter
phi_1 = (1 / beta) - (1 - delta_0);  % consistent with question
 
%% 3. INCOME PROCESS (Tauchen)
w_dim   = 7;   w_bar = 2.5; rho_w = 0.9; sigma_w = 0.15;
[w_grid, P_w] = Tauchen(w_dim, w_bar, rho_w, sigma_w, 3);  % Tauchen already in directory

%% 2. Grids for Firm Decisions 
k_dim   = 40;  k_min = 0.1; k_max = 4 * w_bar;
k_grid  = linspace(k_min, k_max, k_dim)';

n_dim   = 20;   n_min = 0.001; n_max = 2 * max(w_grid);
n_grid  = linspace(n_min, n_max, n_dim);
inv_dim = 30;   inv_min = 0; inv_max = 2 * k_max;
inv_grid = logspace(log10(0.01), log10(0.5), inv_dim);  % nonlinear spacing
u_dim = 10; u_min = 0.01; u_max = 1.5;
u_grid = linspace(u_min, u_max, u_dim);


%% 4. FUNCTION DEFINITIONS

% Production function: f(k, n, u) = (uk)^α * n^(1−α)
prod_fn   = @(k, n, u) (u .* k).^alpha .* n.^(1 - alpha);
% Depreciation function: δ(u) = δ_0 + φ_1(u − 1) + (φ_2 / 2)(u − 1)^2
delta_fn = @(u) delta_0 + phi_1 .* (u - 1) + 0.5 .* phi_2 .* (u - 1).^2;
% Capital accumulation: k' = (1 − δ(u)) * k + inv'
k_next_fn = @(k, inv, u) (1 - delta_fn(u)) .* k + inv;
% Profit function: output − investment − utilization cost
profit_fn = @(k, n, u, inv, w) prod_fn(k, n, u) - inv - w .* n;
    
%% 5. INITIALIZE VALUE AND POLICY MATRICES
Mohi_zeros = @(r, c) zeros(r, c);

V0 = Mohi_zeros(k_dim, w_dim);
for ik = 1:k_dim
    for iw = 1:w_dim
        k = k_grid(ik);
        w = w_grid(iw);
        V0(ik, iw) = prod_fn(k, 1, 1) - 1 * w;  % crude guess: u = 1, n = 1
    end
end

V = V0;
policy = struct('n', Mohi_zeros(k_dim, w_dim), ...
                'inv', Mohi_zeros(k_dim, w_dim), ...
                'u', Mohi_zeros(k_dim, w_dim), ...
                'kp', Mohi_zeros(k_dim, w_dim));
%% 6. BELLMAN ITERATION
tol = 1e-9; dif = Inf; iter = 0;
tic
while dif > tol
    V_new = V;
    for ik = 1:k_dim
        for iw = 1:w_dim
            k = k_grid(ik); w = w_grid(iw);
            best_val = -Inf;

            for u = u_grid
                d = delta_fn(u);
                for n = n_grid
                    for inv = inv_grid
                        kp = (1 - d) * k + inv;
                        kp_idx = max(1, min(k_dim, round((kp - k_min) / (k_max - k_min) * (k_dim - 1)) + 1));
                        EV = sum(P_w(iw,:) .* V(kp_idx,:));
                        val = profit_fn(k, n, u, inv, w) + beta * EV;

                        if val > best_val
                            best_val = val;
                            policy.n(ik, iw)   = n;
                            policy.inv(ik, iw) = inv;
                            policy.u(ik, iw)   = u;
                            policy.kp(ik, iw)  = kp;
                            V_new(ik, iw)      = val;
                        end
                    end
                end
            end
        end
    end
    dif = max(abs(V_new(:) - V(:)));
    V = V_new;
    iter = iter + 1;
    if mod(iter, 50) == 0
        fprintf('Iter %d, diff = %.2e\n', iter, dif);
    end
end
fprintf('\n✅ Converged at iter %d with diff = %.2e\n', iter, dif);
toc


%% 7. DISPLAY FINAL POLICY SNAPSHOTS
fprintf('\n--- Final Policy Function Matrix ---\n');
for ik = 1:k_dim
    fprintf('\nCapital level: k = %.2f\n', k_grid(ik));
    for iw = 1:w_dim
    fprintf('  Income w = %.2f | k′ = %.4f | u = %.4f | n = %.4f | inv′ = %.4f\n', ...
        w_grid(iw), ...
        policy.kp(ik, iw), ...
        policy.u(ik, iw), ...
        policy.n(ik, iw), ...
        policy.inv(ik, iw));
    end
end

%%  Graph the converged value function in (k, w) space for all w
figure;
surf(k_grid, w_grid, V');  % Transpose V to align axes: rows = w, columns = k
xlabel('Capital k');
ylabel('Income w');
zlabel('Value Function V(k, w)');
title('Converged Value Function');
colormap('parula');        % Smooth color gradient
shading interp;            % Interpolated surface shading
view(45, 30);              % 3D viewing angle
grid on;

%% Simulate 1000 periods of income and firm decision
T = 1000;
sim_w_idx = zeros(T,1); sim_w_idx(1) = randi(w_dim);  % Random initial state
sim_k = zeros(T,1); sim_k(1) = k_grid(round(k_dim/2));  % Start from middle capital
sim_n = zeros(T,1); sim_inv = zeros(T,1); sim_u = zeros(T,1); sim_kp = zeros(T,1);

for t = 1:T-1
    iw = sim_w_idx(t);
    ik = max(1, min(k_dim, round((sim_k(t) - k_min)/(k_max - k_min)*(k_dim - 1)) + 1));
    
    sim_n(t)   = policy.n(ik, iw);
    sim_inv(t) = policy.inv(ik, iw);
    sim_u(t)   = policy.u(ik, iw);
    sim_kp(t)  = policy.kp(ik, iw);
    
    sim_k(t+1) = sim_kp(t);
    sim_w_idx(t+1) = randsample(1:w_dim, 1, true, P_w(iw,:));
end
%% Plot Simulated Series (after discarding first 500):
start = 501;
figure;
tiledlayout(5,1);

nexttile; plot(sim_w_idx(start:end)); title('Simulated Income State w');
nexttile; plot(sim_kp(start:end)); title('Simulated Capital k′');
nexttile; plot(sim_n(start:end)); title('Simulated Labor n');
nexttile; plot(sim_inv(start:end)); title('Simulated Investment inv′');
nexttile; plot(sim_u(start:end)); title('Simulated Utilization u');

sgtitle('Simulated Firm Decisions over Time');



%% Calculate standard deviations and analyze effect of doubling φ₂
%Compute Standard Deviations:
std_n   = std(sim_n(start:end));
std_u   = std(sim_u(start:end));
std_inv = std(sim_inv(start:end));

fprintf('Std Dev of n: %.4f\n', std_n);
fprintf('Std Dev of u: %.4f\n', std_u);
fprintf('Std Dev of inv′: %.4f\n', std_inv);

fprintf('Δn across w at k = %.2f: %.4f\n', k_grid(ik), max(policy.n(ik,:)) - min(policy.n(ik,:)));
