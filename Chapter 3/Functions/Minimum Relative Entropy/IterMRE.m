function [X_pos, p_pos, theta_view, mh, mgrad, ens] = IterMRE(z_view, eta_view_ineq, eta_view_eq, logf_pri, X_pri, p_pri, j_, threshold)
% Estimate the updated exponential family distribution under generic base 
% and generalized (in)equality views on expectations via iterative MRE 
%  INPUTS
%   z_view          [function] view function
%   eta_view_ineq   [vector]   (k_ineq x 1) inequality features
%   eta_view_eq     [vector]   (k_eq x 1) equality features
%   logf_pri        [function] base log-pdf (up to normalization constant)
%   X_pri           [matrix]   (n_ x j_) panel matrix of base scenarios
%   p_pri           [vector]   (1 x j_) vector of base probabilities
%   j_              [scalar]   number of scenarios
%   threshold       [scalar]   convergence threshold
%  OUTPUTS
%   X_pos           [matrix]   (n_ x j_ x i_) panel matrix of updated scenarios for each i-th iteration
%   p_pos           [vector]   (1 x j_ x i_) vector of updated probabilities for each i-th iteration
%   theta_view      [matrix]   (k_ineq + k_eq) x i_  optimal Lagrangian multipliers for each i-th iteration
%   mh              [vector]   (1 x i_) value of the Dual Lagrangian at the optimum for each i-th iteration
%   mgrad           [matrix]   (n_ x i_) gradient of the Dual Lagrangian at the optimum for each i-th iteration
%   ens             [vector]   (1 x i_) effective number of scenarios for each i-th iteration
%% Code

%% 0. Initialize
i = 1;
X_pos(:, :, i) = X_pri;
Z_pri = z_view(X_pos(:, :, i));
[theta_view(:, i), mh(i), mgrad(:, i)] = FitLagrMultMRE(eta_view_ineq, eta_view_eq, Z_pri, p_pri);
p_pos(i, :) = UpdateProbMRE(theta_view, Z_pri, p_pri);
ens(i) = EffectiveScenarios(p_pos(i, :))/j_;

while ens(i) < 1 - threshold
i = i + 1;

%% 1. Update scenarios
x_0 = X_pos(:, :, i-1) * p_pos(i-1, :)';
X_pos(:, :, i) = UpdateScenMRE(theta_view(:, i-1), z_view, logf_pri, x_0, j_);
    
%% 2. Update Lagrange multipliers
Z_pri = z_view(X_pos(:, :, i));
[epsi_view, mh(i), mgrad(:, i)] = FitLagrMultMRE(eta_view_ineq, eta_view_eq, Z_pri, p_pri);
theta_view(:, i) = theta_view(:, i-1) + epsi_view;

%% 3. Update probabilities
p_pos(i, :) = UpdateProbMRE(epsi_view, Z_pri, p_pri);
    
%% 4. Check convergence
ens(i) = EffectiveScenarios(p_pos(i, :))/j_;
end
end