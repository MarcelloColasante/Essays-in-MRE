function [theta_sig2_view, mu_pos, sig2_pos] = FitMRE2MomentsN(z_sig, eta_sig2_view, mu_pri, sig2_pri, threshold)
% Compute the optimal Lagrangian multipliers, updated expectation and 
% covariance under normal base and views on linear combinations of 
% second non-central moments via iterative implementation
% update
%  INPUTS
%   z_sig           [matrix]   k_ x n_ pick matrix
%   eta_sig2_view   [matrix]   k_ x k_ features
%   mu_pri          [vector]   n_ x 1 base expectation
%   sig2_pri        [matrix]   n_ x n_ base covariance
%   threshold       [scalar]   convergence threshold
%  OUTPUTS
%   theta_sig2_view [matrix]   k_ x k_ optimal Lagrangian multipliers
%   mu_pos          [vector]   n_ x 1 updated expectation
%   sig2_pos        [matrix]   n_ x n_ updated covariance
%% 0. Initialize
error = 10^6;
i = 1;

eta_sig_view(:, i) = z_sig * mu_pri;
sig2_view_pri = z_sig * sig2_pri * z_sig';

while error>threshold
%% 1. Update features
i = i+1;
eta_sig_view(:, i) = ((eta_sig2_view - eta_sig_view(:, i-1) * eta_sig_view(:, i-1)') / sig2_view_pri) * eta_sig_view(:, 1);

%% 2. Check convergence
error = norm(eta_sig_view(:, i) - eta_sig_view(:, i-1)) / norm(eta_sig_view(:, i));
end

%% Output Lagrange multipliers, expectation and covariance
sig2_view = sig2_view_func(eta_sig2_view, eta_sig_view(:, i));
z_sig_dag = PseudoInverse(z_sig, sig2_pri);

theta_sig2_view = 0.5 * (inv(sig2_view_pri) - inv(sig2_view));
mu_pos = mu_pri + z_sig_dag' * (eta_sig_view(:, i) - z_sig * mu_pri);
sig2_pos = sig2_pri + z_sig_dag' * (sig2_view - sig2_view_pri) * z_sig_dag;
end

function sig2_view = sig2_view_func(eta_sig2_view, eta_sig_view)
sig2_view = eta_sig2_view - eta_sig_view * eta_sig_view';
end