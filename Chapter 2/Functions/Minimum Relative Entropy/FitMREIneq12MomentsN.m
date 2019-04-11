function [theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMREIneq12MomentsN(z_mu_ineq, z_mu_eq, z_sig_ineq, eta_mu_view,  eta_sig2_view, mu_pri, sig2_pri, km)
% Compute the optimal Lagrangian multipliers, updated expectation and 
% covariance under normal base and (in)equality views on linear combinations 
% of first moments and inequality views on second moments via numerical optimization
% update
%  INPUTS
%   z_mu_ineq       [matrix]   k_mu_ineq x n_ pick matrix for inequaity views on first moments
%   z_mu_eq         [matrix]   k_mu_eq x n_ pick matrix for equality views on first moments
%   z_sig_ineq      [matrix]   k_sig x n_ pick matrix for inequality views on second moments
%   eta_mu_view     [vector]   k_mu x 1 features for first moments
%   eta_sig2_view   [matrix]   k_sig x k_sig features for second moments
%   mu_pri          [vector]   n_ x 1 base expectation
%   sig2_pri        [matrix]   n_ x n_ base covariance
%   km              [matrix]   n_^2 x n_^2 commutation matrix
%  OUTPUTS
%   theta_mu_view   [vector]   k_mu x 1 optimal Lagrangian multipliers
%   theta_sig2_view [matrix]   k_sig x k_sig optimal Lagrangian multipliers
%   mu_pos          [vector]   n_ x 1 updated expectation
%   sig2_pos        [matrix]   n_ x n_ updated covariance
%% Code
z_mu = [z_mu_ineq; z_mu_eq];
[k_mu_ineq, n_] = size(z_mu_ineq);
k_mu_eq = size(z_mu_eq, 1);
k_mu = k_mu_ineq + k_mu_eq;

z_sig = z_sig_ineq;
k_sig_ineq = size(z_sig_ineq, 1);
k_sig = k_sig_ineq;

k_ = k_mu + k_sig_ineq^2;  % total number of views

%% Set dual Lagrangian objective
function [mh, mgrad, mHess] = DualLagrangian(t)
[t_mu, t_sig2, mu, sig2] = param(t);

mh = LogPartFuncNorm(mu, sig2) - t_mu' * eta_mu_view - trace(t_sig2' * eta_sig2_view); % value

mgrad = gradient(t);
mHess = hessian(t);
end

function mgrad = gradient(t)
[t_mu, t_sig2, mu, sig2] = param(t);

grad_t_mu = z_mu * mu - eta_mu_view;
grad_t_sig2 = z_sig * (sig2 + mu * mu') * z_sig' - eta_sig2_view;

mgrad = [grad_t_mu; grad_t_sig2(:)]; % gradient
end

function mHess = hessian(t, lambda)
[t_mu, t_sig2, mu, sig2] = param(t);
    
hess_mu2 = z_mu * sig2 * z_mu';

z_sig2 = kron(z_sig, z_sig);
hess_sigmu = z_sig2 * (kron(mu, sig2) + kron(sig2, mu)) * z_mu';


s2_w = (eye(n_^2) + km) * kronecker(sig2, sig2);
a =  kronecker(eye(n_), mu) +  kronecker(mu, eye(n_));
alpha = a * sig2 * a';
beta = (kronecker(sig2, mu) + kronecker(mu, sig2)) * a';
s2 = s2_w - alpha + beta + beta';
hess_sig2 = z_sig2 * s2 * z_sig2';

mHess = [hess_mu2, hess_sigmu'; hess_sigmu, hess_sig2]; % Hessian
end

%% Compute optimal Lagrange multipliers
t_mu_0 = zeros(k_mu , 1);
t_sig2_0 = 0.01 * eye(k_sig);
theta_0 = [t_mu_0; t_sig2_0(:)]; % initial guess 
a_ineq = [eye(k_mu_ineq, k_mu_ineq), zeros(k_mu_ineq, k_mu_eq + k_sig_ineq^2); zeros(k_sig_ineq^2, k_mu), eye(k_sig_ineq^2, k_sig_ineq^2)];
b_ineq = zeros(k_mu_ineq + k_sig_ineq^2,1);
options = optimoptions('fmincon',...
        'SpecifyObjectiveGradient', true, 'HessianFcn', @hessian,...
        'Display','off', 'MaxIter', 1e8, 'TolFun', 1e-16, 'MaxFunEvals',1e8);


theta_view = fmincon(@(theta) DualLagrangian(theta), theta_0, a_ineq, b_ineq,[],[],[],[],[],options);

%% Output re-shaped parameters
[theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = param(theta_view);

function [t_mu, t_sig2, mu, sig2] = param(t)
% Re-shape variable t
t_mu = t(1: k_mu);
t_sig2 = t(k_mu + 1 : k_);
t_sig2 = reshape(t_sig2, k_sig, k_sig);
t_sig2 = (t_sig2 + t_sig2')/2; % ensure t_sig2 is symmetric

% Compute expectation and covariance
sig2_view_pri = z_sig * sig2_pri * z_sig';
p_sig = ((sig2_pri * z_sig') /(0.5 * inv(t_sig2) - sig2_view_pri)) * z_sig;
sig2 = sig2_pri + p_sig * sig2_pri;
sig2 = (sig2 + sig2')/2; % ensure sig2 is symmetric
mu = mu_pri + p_sig * mu_pri + sig2 * z_mu' * t_mu;
end
end

