clearvars; clc;

%% STEP 1: Model Parameters
beta  = 0.96;      % Discount factor
gamma = 1.3;       % CRRA coefficient
r     = 0.04;      % Interest rate
rho   = 0.9;       % Persistence of log income
sigma = 0.04;      % Std dev of income shocks

%% STEP 2: Asset Grid Dimensions
c_dim = 5;         % Number of income states
r_dim = 500;       % Number of asset grid points

%% STEP 3: Tauchen Discretization
mu = 0;            % Mean of log income
m  = 3;            % Range: ±3 standard deviations
[Z, Zprob] = Tauchen(c_dim, mu, rho, sigma, m);  % Z: log-income grid, Zprob: transition matrix

%% STEP 4: Asset Grid Construction
a_min = -exp(Z(1)) / r;       % Borrowing limit using lowest income state
a_max = 10;                   % Max asset level
a     = linspace(a_min, a_max, r_dim)';  % Asset grid


%% STEP 4.5: Utility Function Definition
if gamma == 1
    u_c = @(c) log(c);
else
    u_c = @(c) (c.^(1 - gamma)) / (1 - gamma);
end

%% STEP 5: Preallocation and Concave Initialization
V0 = zeros(r_dim, c_dim);
for j = 1:c_dim
    y = exp(Z(j));
    coh = a + y;
    V0(:, j) = u_c(coh);
end

V1       = Mohi_zeros(r_dim, c_dim);
a_policy = Mohi_zeros(r_dim, c_dim);
c_policy = Mohi_zeros(r_dim, c_dim);

%% STEP 6: Value Function Iteration
tol    = 1e-9;
dif    = Inf;
iter   = 0;
maxits = 1e6;

fprintf('\n--- Starting Value Function Iteration ---\n');
tic;  

while dif > tol && iter < maxits
    iter = iter + 1;

    for j = 1:c_dim
        y = exp(Z(j));  % Income level

        for i = 1:r_dim
            a_now = a(i);       % Current asset
            coh   = a_now + y;  % Cash-on-hand
            v     = -Inf * ones(r_dim, 1);  % Bellman RHS for each a'

            for a_prime = 1:r_dim
                c = coh - a(a_prime) / (1 + r);  % Consumption

                if c > 0
                    EV = Zprob(j,:) * V0(a_prime,:)';   % Expected future value
                    v(a_prime) = u_c(c) + beta * EV;    % Bellman RHS
                end
            end

            % ✅ Fallback if all v are -Inf (invalid consumption)
            if all(~isfinite(v))
                V1(i, j)       = -1e10;       % Large negative fallback
                a_policy(i, j) = a(1);        % Default to lowest asset
                c_policy(i, j) = NaN;         % Consumption undefined
            else
                [V1(i, j), idx] = max(v);     % Optimal value
                a_policy(i, j) = a(idx);      % Optimal asset choice
                c_policy(i, j) = coh - a_policy(i, j) / (1 + r);  % Optimal consumption
            end
        end
    end

    dif = max(abs(V1(:) - V0(:)));  % Convergence check
    V0 = V1;                        % Update for next iteration
end

toc;  

fprintf('\n--- Convergence Report ---\n');
fprintf('Final convergence difference: %.2e\n', dif);

if dif < tol
    fprintf('✅ Converged successfully within tolerance (tol = %.1e).\n', tol);
    fprintf('Total iterations: %d\n', iter);
else
    fprintf('⚠️ Did not converge within tolerance (tol = %.1e).\n', tol);
    fprintf('Consider increasing maxits (currently %d) or refining the asset/income grid.\n', maxits);
end
fprintf('---------------------------\n\n');
%% STEP 7: Plot Value Function V0(a, y) for All y
figure;
hold on;
colors = lines(c_dim);  % Distinct colors for each income state

for y1 = 1:c_dim
    plot(a, V0(:, y1), 'LineWidth', 1.5, 'Color', colors(y1,:));
end

xlabel('Assets a');
ylabel('Value Function V(a, y)');
title('Converged Value Function for All Income States');
legend(arrayfun(@(i) sprintf('y_%d', i), 1:c_dim, 'UniformOutput', false));
grid on;
hold off;





%% STEP 8: Simulation with Income Innovations
T      = 1000;       % Total simulation periods
T_burn = 500;        % Burn-in periods

% Generate income innovations from N(0, sigma)
rng(123);  % For reproducibility
eps_sim = normrnd(0, sigma, T, 1);  % Income shocks

% Simulate log-income path using AR(1)
log_income_path = zeros(T, 1);
log_income_path(1) = 0;  % Start from mean of AR(1)

for t = 2:T
    log_income_path(t) = rho * log_income_path(t-1) + eps_sim(t);
end

% Convert to actual income
income_path = exp(log_income_path);

% Initialize state trackers
income_state     = zeros(T, 1); income_state(1) = find(Z == min(abs(Z)));
asset_state      = zeros(T, 1); asset_state(1)  = round(r_dim / 2);
asset_path       = zeros(T, 1);
consumption_path = zeros(T, 1);

% Simulate asset and consumption paths
for t = 2:T
    % Map log-income to nearest discrete state
    [~, income_state(t)] = min(abs(Z - log_income_path(t)));

    % Apply policy functions using previous asset and income states
    asset_path(t)       = a_policy(asset_state(t-1), income_state(t-1));
    consumption_path(t) = c_policy(asset_state(t-1), income_state(t-1));

    % Update asset state index for next period
    [~, asset_state(t)] = min(abs(a - asset_path(t)));
end

% Drop burn-in periods
income_path      = income_path(T_burn+1:end);
asset_path       = asset_path(T_burn+1:end);
consumption_path = consumption_path(T_burn+1:end);
figure;
tiledlayout(3,1);

nexttile; plot(income_path, 'LineWidth', 1.2); title('Simulated Income Path');
nexttile; plot(asset_path, 'LineWidth', 1.2); title("Simulated Asset Path");
nexttile; plot(consumption_path, 'LineWidth', 1.2); title('Simulated Consumption Path');


%% STEP 13: Standard Deviation of Consumption
std_consumption = std(consumption_path);
disp(['Standard deviation of simulated consumption: ', num2str(std_consumption)]);

%% Cases
function std_c = simulate_model(r_dim, c_dim, gamma, r, sigma, borrowing_limit)
    % Tauchen Discretization
    [Z, Zprob] = Tauchen(c_dim, 0, 0.9, sigma, 3);

    % Asset Grid Construction
    a_min = borrowing_limit;
    a_max = 10;
    a     = linspace(a_min, a_max, r_dim)';

    % Utility Function
    if gamma == 1
        u_c = @(c) log(c);
    else
        u_c = @(c) (c.^(1 - gamma)) / (1 - gamma);
    end

    % Concave Initialization of V0
    V0 = zeros(r_dim, c_dim);
    for j = 1:c_dim
        y = exp(Z(j));
        coh = a + y;
        V0(:, j) = u_c(coh);
    end

    % Preallocation
    V1       = zeros(r_dim, c_dim);
    a_policy = zeros(r_dim, c_dim);
    c_policy = zeros(r_dim, c_dim);

    % Parameters
    beta = 0.96;
    tol  = 1e-9;
    dif  = Inf;

    % Value Function Iteration
    while dif > tol
        for j = 1:c_dim
            y = exp(Z(j));
            for i = 1:r_dim
                coh = a(i) + y;
                v_temp = -Inf * ones(r_dim, 1);
                for a_prime = 1:r_dim
                    c = coh - a(a_prime) / (1 + r);
                    if c > 0
                        EV = Zprob(j,:) * V0(a_prime,:)';
                        v_temp(a_prime) = u_c(c) + beta * EV;
                    end
                end
                if all(~isfinite(v_temp))
                    V1(i, j)       = -1e10;
                    a_policy(i, j) = a(1);
                    c_policy(i, j) = NaN;
                else
                    [V1(i, j), idx] = max(v_temp);
                    a_policy(i, j) = a(idx);
                    c_policy(i, j) = coh - a_policy(i, j) / (1 + r);
                end
            end
        end
        dif = max(abs(V1(:) - V0(:)));
        V0 = V1;
    end

    % Simulation Parameters
    T         = 1000;
    T_burn    = 500;
    income_state     = zeros(T, 1); income_state(1) = ceil(c_dim / 2);
    asset_state      = zeros(T, 1); asset_state(1)  = ceil(r_dim / 2);
    consumption_path = zeros(T, 1);

    rng(123);
    for t = 2:T
        income_state(t) = find(rand <= cumsum(Zprob(income_state(t-1), :)), 1);
        next_a = a_policy(asset_state(t-1), income_state(t-1));
        [~, asset_state(t)] = min(abs(a - next_a));
        consumption_path(t) = c_policy(asset_state(t-1), income_state(t-1));
    end

    % Drop burn-in
    consumption_path = consumption_path(T_burn+1:end);
    std_c = std(consumption_path, 'omitnan');
end

% Baseline
baseline_std     = simulate_model(500, 5, 1.3, 0.04, 0.04, -exp(-1)/0.04);

% (a) No borrowing constraint
zero_borrow_std  = simulate_model(500, 5, 1.3, 0.04, 0.04, 0);

% (b) Double risk aversion
high_risk_std    = simulate_model(500, 5, 2.6, 0.04, 0.04, -exp(-1)/0.04);

% (c) Double interest rate
high_r_std       = simulate_model(500, 5, 1.3, 0.08, 0.04, -exp(-1)/0.08);

% (d) Double income volatility
high_sigma_std   = simulate_model(500, 5, 1.3, 0.04, 0.08, -exp(-1)/0.04);

% Display Results

fprintf('\n--- STEP 12: Standard Deviation of Consumption ---\n');

tic;
fprintf('Baseline std(c): %.5f\n', baseline_std);
fprintf('(a) Zero borrowing: %.5f\n', zero_borrow_std);
fprintf('(b) Double risk aversion: %.5f\n', high_risk_std);
fprintf('(c) Double interest rate: %.5f\n', high_r_std);
fprintf('(d) Double income volatility: %.5f\n', high_sigma_std);
toc;

%% STEP 11: Grid Resolution Stability Test
%{
grid_sizes    = [300, 700];  % Asset grid resolutions to test
income_states = [5, 7];      % Income state counts to test

% Model Parameters (match main model)
beta  = 0.96;
gamma = 1.3;
r     = 0.04;
rho   = 0.9;
sigma = 0.04;

% Tauchen Parameters
mu = 0;
m  = 3;

% Utility Function
if gamma == 1
    u_c = @(c) log(c);
else
    u_c = @(c) (c.^(1 - gamma)) / (1 - gamma);
end

% Convergence Settings
tol    = 1e-9;
maxits = 1000;

for r_dim = grid_sizes
    for c_dim = income_states
        fprintf('\n--- Testing r_dim = %d, c_dim = %d ---\n', r_dim, c_dim);
        tic;

        % Tauchen Discretization
        [Z, Zprob] = Tauchen(c_dim, mu, rho, sigma, m);

        % Asset Grid Construction
        a_min = -exp(Z(1)) / r;
        a_max = 10;
        a     = linspace(a_min, a_max, r_dim)';

        % Concave Initialization of V0
        V0 = zeros(r_dim, c_dim);
        for j = 1:c_dim
            y = exp(Z(j));
            coh = a + y;
            V0(:, j) = u_c(coh);
        end

        % Preallocation
        V1 = zeros(r_dim, c_dim);

        % Value Function Iteration
        dif  = Inf;
        iter = 0;

        while dif > tol && iter < maxits
            iter = iter + 1;

            for j = 1:c_dim
                y = exp(Z(j));

                for i = 1:r_dim
                    coh = a(i) + y;
                    v   = -Inf * ones(r_dim, 1);

                    for a_prime = 1:r_dim
                        c = coh - a(a_prime) / (1 + r);
                        if c > 0
                            EV = Zprob(j,:) * V0(a_prime,:)';
                            v(a_prime) = u_c(c) + beta * EV;
                        end
                    end

                    % Fallback if all v are invalid
                    if all(~isfinite(v))
                        V1(i, j) = -1e10;
                    else
                        V1(i, j) = max(v);
                    end
                end
            end

            dif = max(abs(V1(:) - V0(:)));
            V0 = V1;
        end

        % Convergence Report
        if isnan(dif)
            fprintf('⚠️ Convergence failed: Final difference is NaN\n');
        else
            fprintf('✅ Converged in %d iterations. Final difference: %.2e\n', iter, dif);
        end
        toc;
    end
end

%}