%% Variable Capital Utilization
clear; clc;
%% Parameters
alpha    = 0.4;beta     = 0.96; delta_0  = 0.1;
phi_2    = 0.2; r        = 0.04;
% Derived parameter
phi_1 = (1 / beta) - (1 - delta_0);  % consistent with question

%% === Tauchen Discretization for Income w ===
rho     = 0.9; sigma   = 0.15; w_bar   = 2.5;
w_dim   = 7; m       = 4;
[Z, P] = Tauchen(w_dim, w_bar, rho, sigma, m);  % Z = income grid, P = transition matrix
w_grid = Z;  % income levels

%% === Grids for Firm Decisions ===
k_dim   = 40;  k_min = 0.1; k_max = 4 * w_bar;
k_grid  = linspace(k_min, k_max, k_dim)';
n_dim   = 20;   n_min = 0.01; n_max = 1.5 * exp(max(w_grid));
n_grid  = linspace(n_min, n_max, n_dim);
inv_dim = 30;   inv_min = 0; inv_max = 2 * k_max;
inv_grid = logspace(log10(0.01), log10(0.5), inv_dim);  % nonlinear spacing
u_dim = 10; u_min = 0.01; u_max = 1.5;
u_grid = linspace(u_min, u_max, u_dim);
%% === Depreciation Function ===
delta_u = @(u) delta_0 + phi_1 * (u - 1) + 0.5 * phi_2 * (u - 1).^2;

%% === Production Function ===
f = @(k, u, n) (u .* k).^alpha .* n.^(1 - alpha);

%% === Initialization ===
V0 = Mohi_zeros(k_dim, w_dim);
for ik = 1:k_dim
    for iw = 1:w_dim
        k = k_grid(ik);
        w = w_grid(iw);
        % Crude static profit guess: full utilization, 1 unit labor, no investment
        V0(ik, iw) = f(k, 1, 1) - w * 1;
    end
end

V      = V0;     % Initial guess for value function
V1  = V;

policy = struct('n', Mohi_zeros(k_dim, w_dim), ...
                'inv', Mohi_zeros(k_dim, w_dim), ...
                'u', Mohi_zeros(k_dim, w_dim), ...
                'kp', Mohi_zeros(k_dim, w_dim));

%% Bellman Iteration
tol = 1e-9; dif = Inf; iter = 0;
tic
while dif > tol
    V_new = V;
    for ik = 1:k_dim
        for iw = 1:w_dim
            k = k_grid(ik); w = w_grid(iw); best_val = -Inf;
            best_n = NaN; best_inv = NaN; best_u = NaN; best_kp = NaN;

            %% Step 1: Fix u = 1, search over n and inv
            u_fixed = 1;
            d_fixed = delta_u(u_fixed);
            for n = n_grid
                for inv = inv_grid
                    kp = (1 - d_fixed) * k + inv;
                    kp_idx = max(1, min(k_dim, round((kp - k_min) / (k_max - k_min) * (k_dim - 1)) + 1));
                    EV = P(iw,:) * V(kp_idx,:)';
                    val = f(k, u_fixed, n) - inv - w * n + beta * EV;

                    if val > best_val
                        best_val = val;
                        best_n = n;
                        best_inv = inv;
                        best_kp = kp;
                        best_u = u_fixed;
                    end
                end
            end

%% Step 2: Refine u locally around best_u
            u_local = linspace(max(u_min, best_u - 0.2), min(u_max, best_u + 0.2), 5);
            for u = u_local
                d = delta_u(u);
                kp = (1 - d) * k + best_inv;
                kp_idx = max(1, min(k_dim, round((kp - k_min) / (k_max - k_min) * (k_dim - 1)) + 1));
                EV = P(iw,:) * V(kp_idx,:)';
                val = f(k, u, best_n) - best_inv - w * best_n + beta * EV;

                if val > best_val
                    best_val = val;
                    best_u = u;
                    best_kp = kp;
                end
            end

            %% Store results
            V_new(ik, iw) = best_val;
            policy.n(ik, iw) = best_n;
            policy.u(ik, iw) = best_u;
            policy.inv(ik, iw) = best_inv;
            policy.kp(ik, iw) = best_kp;
        end
    end
    dif = max(max(abs(V_new - V)));
    V = V_new;
    iter = iter + 1;
    if mod(iter, 50) == 0
        fprintf('Iter %d, diff = %.2e\n', iter, dif);
    end
end
fprintf('\n✅ Converged at iter %d with diff = %.2e\n', iter, dif);
toc

%% (b) Display the converged value function over (k, w)
disp('--- Full Value Function V(k, w) ---');
disp(V);

disp('--- Full Capital Policy k′(k, w) ---');
disp(policy.kp);

disp('--- Full Utilization Policy u(k, w) ---');
disp(policy.u);

disp('--- Full Labor Policy n(k, w) ---');
disp(policy.n);

disp('--- Full Investment Policy inv′(k, w) ---');
disp(policy.inv);

%% 
figure;
surf(k_grid, w_grid, V');  % transpose V for correct orientation
xlabel('Capital k');
ylabel('Income w');
zlabel('Value Function V(k, w)');
title('Converged Value Function');
colormap('parula'); shading interp; view(45, 30);
fprintf('inv′ min = %.4f, max = %.4f\n', min(policy.inv(:)), max(policy.inv(:)));
%% === Generate a series of 1000 simulated income innovations ===
T = 1000; discard = 500;
rng(123);  % reproducibility

% Simulate income process: AR(1) centered at w_bar
epsilon = sigma * randn(T,1);
w_sim = Mohi_zeros(T,1);
w_sim(1) = w_bar;
for t = 2:T
    w_sim(t) = (1 - rho) * w_bar + rho * w_sim(t-1) + epsilon(t);
end

% Initialize firm paths
k_sim   = Mohi_zeros(T,1);
inv_sim = Mohi_zeros(T,1);
n_sim   = Mohi_zeros(T,1);
u_sim   = Mohi_zeros(T,1);
c_sim   = Mohi_zeros(T,1);

k_sim(1) = k_grid(round(k_dim/2));  % start at mid-capital

% Helper function for grid indexing
findClosest = @(grid, val) find(abs(grid - val) == min(abs(grid - val)), 1);

for t = 1:T-1
    ik = findClosest(k_grid, min(max(k_sim(t), k_min), k_max));
    iw = findClosest(w_grid, min(max(w_sim(t), min(w_grid)), max(w_grid)));

    inv_sim(t) = policy.inv(ik, iw);
    n_sim(t)   = policy.n(ik, iw);
    u_sim(t)   = policy.u(ik, iw);
    k_sim(t+1) = policy.kp(ik, iw);

    % Compute consumption: output - wage bill - investment
    output = f(k_sim(t), u_sim(t), n_sim(t));
    c_sim(t) = output - w_sim(t) * n_sim(t) - inv_sim(t);
end

% Discard burn-in
w_sim   = w_sim(discard+1:end);
k_sim   = k_sim(discard+1:end);
inv_sim = inv_sim(discard+1:end);
n_sim   = n_sim(discard+1:end);
u_sim   = u_sim(discard+1:end);
c_sim   = c_sim(discard+1:end);

%% === Plot Tiled Layout ===
figure;
tiledlayout(2,2);
nexttile; plot(w_sim); title('Simulated Income w');
nexttile; plot(k_sim); title('Capital a′');
nexttile; plot(n_sim); title('Labor n');
nexttile; plot(c_sim); title('Consumption c');

figure;
subplot(2,2,1); plot(w_sim); title('Simulated Income w');
ylim([min(w_sim)-0.1, max(w_sim)+0.1]);

subplot(2,2,2); plot(k_sim); title('Capital a′');
ylim([min(k_sim)-0.1, max(k_sim)+0.1]);

subplot(2,2,3); plot(n_sim); title('Labor n');
ylim([min(n_sim)-0.01, max(n_sim)+0.01]);

subplot(2,2,4); plot(c_sim); title('Consumption c');
ylim([min(c_sim)-0.01, max(c_sim)+0.01]);

%% === Standard Deviations of Simulated Paths ===
std_n   = std(n_sim);
std_u   = std(u_sim);
std_inv = std(inv_sim);

fprintf('\nStandard Deviations:\n');
fprintf('Labor (n):        %.4f\n', std_n);
fprintf('Utilization (u):  %.4f\n', std_u);
fprintf('Investment (inv′): %.4f\n', std_inv);

%% === Economic Intuition Commentary ===
fprintf('\nEconomic Intuition: What Happens When...\n');

fprintf('\n(a) phi_2 Doubled (Higher curvature in depreciation)\n');
fprintf('- Utilization u becomes more costly when deviating from 1\n');
fprintf('- Firms will flatten their utilization path to avoid nonlinear depreciation\n');
fprintf('- This reduces volatility in u, and indirectly in inv′ and n because:\n');
fprintf('  • Less aggressive utilization → less incentive to expand capital → smoother investment\n');
fprintf('  • Less capital growth → less labor demand\n');

fprintf('\nQualitative Expectation:\n');
fprintf('%-12s | %-30s\n', 'Variable', 'Std Dev Effect');
fprintf('-------------+--------------------------------\n');
fprintf('%-12s | %-30s\n', 'u',     '↓ (less variation)');
fprintf('%-12s | %-30s\n', 'inv′',  '↓ (less aggressive investment)');
fprintf('%-12s | %-30s\n', 'n',     '↓ (less labor scaling)');

fprintf('\n(b) Real Interest Rate r Doubled\n');
fprintf('- Higher r → lower present value of future profits\n');
fprintf('- Firms become more short-term focused\n');
fprintf('- They may invest and hire less, but also adjust more aggressively to shocks\n');

fprintf('\nQualitative Expectation:\n');
fprintf('%-12s | %-30s\n', 'Variable', 'Std Dev Effect');
fprintf('-------------+--------------------------------\n');
fprintf('%-12s | %-30s\n', 'u',     '↑ or ↓ depending on shock sensitivity');
fprintf('%-12s | %-30s\n', 'inv′',  '↓ (less investment overall)');
fprintf('%-12s | %-30s\n', 'n',     '↓ (lower labor demand)');