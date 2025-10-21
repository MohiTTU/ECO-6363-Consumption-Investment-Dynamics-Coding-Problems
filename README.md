
Part 1 : Dynamic Programming Simulation: Consumption and Asset Policy
------------------------------------------------------------
Author: Mohi, Date: October 2025
Language: MATLAB
Purpose: 
1) Simulate household decision-making under income uncertainty using dynamic programming. 
2) Compute optimal consumption and asset policies, simulate economic paths, and evaluate the impact of key parameters on consumption volatility.
------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
1. Start fresh:
   clearvars; clc;
2. Run the full script from top to bottom. It includes:
   - Model setup
   - Value function iteration
   - Policy function extraction
   - Simulation of income, asset, and consumption paths
   - Standard deviation analysis across scenarios
3. Ensure required functions are available:
   - Tauchen.m for discretizing income shocks
   - Mohi_zeros.m for preallocation (or replace with zeros)

------------------------------------------------------------
KEY STEPS
------------------------------------------------------------
STEP 1–5: Model Setup
---------------------
- Defines parameters: beta, gamma, r, rho, sigma
- Constructs asset grid and income states via Tauchen
- Initializes value function V0 using concave utility over cash-on-hand
IMPROVEMENT:
Before concave initialization, convergence required 547 iterations.
? Now with concave initialization:
   Final convergence difference: 9.94e-10
   Total iterations: 996
STEP 6: Value Function Iteration
--------------------------------
- Iterates over asset and income states to solve the Bellman equation
- Uses fallback logic for infeasible consumption
- Tracks convergence with tic/toc
STEP 7: Value Function Plot
---------------------------
- Plots V(a, y) for all income states
- Shows how value increases with assets and income
STEP 8: Simulation
------------------
- Simulates 1000 periods of income shocks (AR(1))
- Applies policy functions to generate asset and consumption paths
- Drops first 500 periods as burn-in
Standard deviation of simulated consumption: 0.093875
STEP 12–13: Volatility Analysis
-------------------------------
Compares baseline and counterfactual scenarios:
    Baseline std(c): 0.13088
    (a) Zero borrowing: 0.11521
    (b) Double risk aversion: 0.12659
    (c) Double interest rate: 0.11231
    (d) Double income volatility: 0.23952
Elapsed time for volatility analysis: 0.010186 seconds
------------------------------------------------------------
OUTPUT SUMMARY
------------------------------------------------------------
- Policy functions: a_policy, c_policy
- Simulated paths: income_path, asset_path, consumption_path
- Diagnostic plots: Value function and time series
- Volatility metrics: std(c) across scenarios
------------------------------------------------------------
NOTES
------------------------------------------------------------
- The simulation uses CRRA utility and borrowing constraints.
- All random draws are seeded (rng(123)) for reproducibility.
- You can adjust r_dim, c_dim, gamma, r, or sigma to explore sensitivity.
- The concave initialization of V0 significantly improves convergence speed and stability.
------------------------------------------------------------
END OF FILE
