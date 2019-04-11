function [theta_view, mh, mgrad] = FitLagrMultMRE(eta_view_ineq, eta_view_eq, Z_pri, p_pri)
% Estimate the optimal Lagrange multipliers for the updated exponential 
% family distribution under generic base and generalized (in)equality views 
% on expectations
%  INPUTS
%   eta_view_ineq   [vector]   (k_ineq x 1) inequality features
%   eta_view_eq     [vector]   (k_eq x 1) equality features
%   X_pri           [matrix]   (n_ x j_) panel matrix of base scenarios
%   p_pri           [vector]   (1 x j_) vector of base probabilities
%  OUTPUTS
%   theta_view      [vector]   (k_ineq + k_eq) x 1  optimal Lagrangian multipliers
%   mh              [vector]   value of the Dual Lagrangian at the optimum 
%   mgrad           [vector]   (n_ x 1) gradient of the Dual Lagrangian at the optimum
%% Code
k_ineq = length(eta_view_ineq);
k_eq = length(eta_view_eq);

if k_ineq == 0
    % no inequality constraints
    Z_pri_ineq = [];
    eta_view_ineq = [];
    Z_pri_eq = Z_pri;
        
elseif k_eq == 0
    % no equality constraints
    Z_pri_ineq = Z_pri;
    Z_pri_eq = [];
    eta_view_eq = [];
        
else   
    Z_pri_ineq = Z_pri(1:k_ineq, :);
    Z_pri_eq = Z_pri(k_ineq + 1 : k_ineq + k_eq, :);
    
end

%% Concatenate the constrataints
if isempty(Z_pri_ineq)
    % no inequality constraints
    Z_pri = Z_pri_eq;
    mu_view = eta_view_eq;
    k_ineq = 0;
    k_eq = length(eta_view_eq);
    
elseif isempty(Z_pri_eq)
    % no equality constraints;
    Z_pri = Z_pri_ineq;
    mu_view = eta_view_ineq;
    k_ineq = length(eta_view_ineq);
    k_eq = 0;
    
else
    Z_pri = [Z_pri_ineq; Z_pri_eq];
    mu_view = [eta_view_ineq; eta_view_eq];
    k_ineq = length(eta_view_ineq);
    k_eq = length(eta_view_eq);
end

k_ = k_ineq + k_eq;  % total number of views 

%% Set dual Lagrangian objective
function [mh, mgrad, mHess] = DualLagrangian(theta)
%
psi = LogPartitionFunc(theta, Z_pri, p_pri);
mh =  psi - theta' * mu_view; % value  

mgrad = gradient(theta);
mHess = hessian(theta);
end

function mgrad = gradient(theta)
    p = UpdateProbMRE(theta, Z_pri, p_pri);
    mgrad = Z_pri * p' - mu_view; % gradient
end

function mHess = hessian(theta, lambda)
    p = UpdateProbMRE(theta, Z_pri, p_pri);
    [~, mHess] = FPmeancov(Z_pri,p); % hessian
end

%% Compute optimal Lagrange multipliers
theta_0 = 0.001 * rand(k_ , 1); % initial guess

if k_ineq == 0
    % if no constraints, then perform the Newton conjugate gradient
    % trust-region algorithm
    options = optimoptions('fminunc','Algorithm','trust-region',...
        'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective', ...
        'Display','off', 'MaxIter', 1e8, 'TolFun', 1e-16, 'MaxFunEvals',1e8);
    theta_view = fminunc(@(theta) DualLagrangian(theta), theta_0, options);
    
else
    % otherwise perform sequential least squares programming
    options = optimoptions('fmincon',...
        'SpecifyObjectiveGradient', true, 'HessianFcn', @hessian,...
        'Display','off', 'MaxIter', 1e8, 'TolFun', 1e-16, 'MaxFunEvals',1e8);
    
    alpha = eye(k_ineq, k_);
    beta = zeros(k_ineq,1);
    
    theta_view = fmincon(@(theta) DualLagrangian(theta), theta_0, alpha,beta,[],[],[],[],[],options);
end

%% Output dual Lagrangian value and gradient
[mh, mgrad] = DualLagrangian(theta_view);
end

