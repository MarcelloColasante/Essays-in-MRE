function [theta_mu_view, mu_pos, sig2_pos] = FitMREIneq1MomentsN(z_mu_ineq, z_mu_eq, eta_mu_view, mu_pri, sig2_pri)
% Compute the optimal Lagrangian multipliers, updated expectation and 
% covariance under normal base and (in)equality views on linear combinations 
% of first moments via numerical optimization
% update
%  INPUTS
%   z_mu_ineq       [matrix]   k_ineq x n_ pick matrix for inequaity views
%   z_mu_eq         [matrix]   k_eq x n_ pick matrix for equality views
%   eta_mu_view     [vector]   k_ x 1 features for first moments
%   mu_pri          [vector]   n_ x 1 base expectation
%   sig2_pri        [matrix]   n_ x n_ base covariance
%  OUTPUTS
%   theta_mu_view   [vector]   k_mu x 1 optimal Lagrangian multipliers
%   mu_pos          [vector]   n_ x 1 updated expectation
%   sig2_pos        [matrix]   n_ x n_ updated covariance
%% Code
options = optimoptions('quadprog', 'Display', 'none');

%% Compute optimal Lagrange multipliers
z_mu = [z_mu_ineq; z_mu_eq];

k_ineq = size(z_mu_ineq, 1);
k_eq = size(z_mu_eq, 1);
k_ = k_ineq + k_eq;

theta_0 = 0.001 * rand(k_ , 1); % initial guess

s2 = z_mu * sig2_pri * z_mu';
s2 = (s2 + s2')/2; % ensure Hessian is symmetric

if isempty(z_mu_eq) % no equality views
    
theta_mu_view = quadprog(s2, z_mu * mu_pri - eta_mu_view, [], [], [], [], [], zeros(k_, 1), theta_0, options);

else

a = [eye(k_ineq), zeros(k_ineq, k_eq)];
b = zeros(k_ineq, 1);

theta_mu_view = quadprog(s2, z_mu * mu_pri - eta_mu_view, a, b, [], [], [], [], theta_0, options);
end
%% Compute posterior expectation and covariance
mu_pos = mu_pri + sig2_pri * z_mu' * theta_mu_view;
sig2_pos = sig2_pri;

