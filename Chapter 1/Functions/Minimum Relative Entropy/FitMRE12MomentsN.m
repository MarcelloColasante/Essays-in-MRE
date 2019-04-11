function [theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMRE12MomentsN(z_mu, z_sig, eta_mu_view, eta_sig2_view, mu_pri, sig2_pri, threshold)
% Compute the optimal Lagrangian multipliers, updated expectation and 
% covariance under normal base and views on linear combinations of 
% first and second non-central moments via iterative implementation
% update
%  INPUTS
%   z_mu            [matrix]   k_mu x n_ pick matrix for first moments
%   z_sig           [matrix]   k_sig x n_ pick matrix for second moments
%   eta_mu_view     [vector]   k_mu x 1 features for first moments
%   eta_sig2_view   [vector]   k_sig x k_sig features for second moments
%   mu_pri          [vector]   n_ x 1 base expectation
%   sig2_pri        [matrix]   n_ x n_ base covariance
%   threshold       [scalar]   convergence threshold
%  OUTPUTS
%   theta_mu_view   [vector]   k_mu x 1 optimal Lagrangian multipliers
%   theta_sig2_view [vector]   k_sig x k_sig optimal Lagrangian multipliers
%   mu_pos          [vector]   n_ x 1 updated expectation
%   sig2_pos        [matrix]   n_ x n_ updated covariance
%% 0. Initialize
error = 10^6;
i = 1;

eta_sig_view(:, i) = z_sig * mu_pri;
sig2_view_pri = z_sig * sig2_pri * z_sig';
z_sig_dag = PseudoInverse(z_sig, sig2_pri);

while error>threshold
%% 1. Update features
i = i+1;

sig2_view = sig2_view_func(eta_sig2_view, eta_sig_view(:, i-1));
sig2_pos = sig2_pri + z_sig_dag' * (sig2_view - sig2_view_pri) * z_sig_dag;

z_mu_dag = PseudoInverse(z_mu, sig2_pos);
mu_sig = mu_pri + z_sig_dag' * ((sig2_view / sig2_view_pri) *  z_sig * mu_pri - z_sig * mu_pri);

eta_sig_view(:, i) = (sig2_view / sig2_view_pri) *  z_sig * mu_pri + z_sig * z_mu_dag' * (eta_mu_view - z_mu * mu_sig);

%% 2. Check convergence
error = norm(eta_sig_view(:, i) - eta_sig_view(:, i-1)) / norm(eta_sig_view(:, i));
end
%% Output Lagrange multipliers, expectation and covariance
sig2_view = sig2_view_func(eta_sig2_view, eta_sig_view(:, i));
theta_sig2_view = 0.5 * (inv(sig2_view_pri) - inv(sig2_view));

sig2_pos = sig2_pri + z_sig_dag' * (sig2_view - sig2_view_pri) * z_sig_dag;

mu_sig = mu_pri + z_sig_dag' * ((sig2_view / (z_sig * sig2_pri * z_sig')) *  z_sig * mu_pri - z_sig * mu_pri);
theta_mu_view = (z_mu * sig2_pos * z_mu') \ (eta_mu_view - z_mu * mu_sig);

z_mu_dag = PseudoInverse(z_mu, sig2_pos); 
mu_pos = mu_sig + z_mu_dag' * (eta_mu_view - z_mu * mu_sig);
end

function sig2_view = sig2_view_func(eta_sig2_view, eta_sig_view)
sig2_view = eta_sig2_view - eta_sig_view * eta_sig_view';
end