function [theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMREExpCovN(z_mu, z_sig, mu_view, sig2_view, mu_pri, sig2_pri)
% Compute the updated Lagrangian multipliers for views on linear combinations
% of expectations and covariances
%  INPUTS
%   z_mu            [matrix]   k_mu x n_ pick matrix
%   z_sig           [matrix]   k_sig x n_ pick matrix
%   mu_view         [vector]   k_mu x 1 features
%   sig2_view       [matrix]   k_sig x k_sig features
%   mu_pri          [vector]   n_ x 1 base expectation
%   sig2_pri        [matrix]   n_ x n_ base covariance
%   threshold       [scalar]   convergence threshold
%  OUTPUTS
%   theta_mu_view   [vector]   k_mu x 1 optimal Lagrangian multipliers
%   theta_sig2_view [matrix]   k_sig x k_sig optimal Lagrangian multipliers
%   mu_pos          [vector]   n_ x 1 updated expectation
%   sig2_pos        [matrix]   n_ x n_ updated covariance
%% Compute updated covariance
theta_sig2_view = 0.5 * (inv(z_sig * sig2_pri * z_sig') - inv(sig2_view)); % optimal Lagrange multipliers

z_sig_dag = PseudoInverse(z_sig, sig2_pri);
sig2_pos = sig2_pri + z_sig_dag' * (sig2_view - z_sig * sig2_pri * z_sig') * z_sig_dag;


%% Compute updated expectation
mu_sig = mu_pri + z_sig_dag' * ((sig2_view / (z_sig * sig2_pri * z_sig')) *  z_sig * mu_pri - z_sig * mu_pri);
theta_mu_view = (z_mu * sig2_pos * z_mu') \ (mu_view - z_mu * mu_sig); % optimal Lagrange multipliers

z_mu_dag = PseudoInverse(z_mu, sig2_pos); 
mu_pos = mu_sig + z_mu_dag' * (mu_view - z_mu * mu_sig);