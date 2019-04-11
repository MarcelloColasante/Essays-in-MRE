function [obj, grad, hess] = REnormLRD(theta, mu_pri, invsigma2_pri, n_, k_, Supplied)
% Relative entropy with low-rank-diagonal "factor" structure on covariance

%  INPUTS 
%   theta           [vector]: (n_ * (2 + k_) x 1) vector of variables (mu; b(:); d)
%   mu_pri          [vector]: (n_ x 1) vector of expectations (prior)
%   invsigma2_pri   [matrix]: (n_ x n_) inverse covariance matrix (prior) 
%   n_              [scalar]:  market dimension
%   k_              [scalar]:  nb of factors
%   Supplied        [struct]:  settings for user supplied derivatives

%  OUTPUTS
%   obj         [scalar]:  entropy
%   grad        [vector]: (n_ * (2 + k_) x 1) gradient
%   hess        [matrix]: (n_ * (2 + k_) x n_ * (2 + k_)) hessian

%% code
if nargin < 6 || isempty(Supplied); Supplied = []; end
if ~isfield(Supplied, 'Grad') || isempty(Supplied.Grad); Supplied.Grad.false = false; end
if ~isfield(Supplied.Grad, 'false') || isempty(Supplied.Grad.false); Supplied.Grad.false = false; end
if ~isfield(Supplied, 'Hess') || isempty(Supplied.Hess); Supplied.Hess.false = false; end
if ~isfield(Supplied.Hess, 'false') || isempty(Supplied.Hess.false); Supplied.Hess.false = false; end
if ~isfield(Supplied, 'matrix') || isempty(Supplied.matrix); Supplied.matrix = []; end

% relative entropy
[mu, sigma2, b, d] = theta2param(theta, n_, k_);
obj = 0.5 * (trace(sigma2 * invsigma2_pri) - log(det(sigma2 * invsigma2_pri)) + (mu - mu_pri)' * invsigma2_pri *(mu - mu_pri) - n_);

% Gradient
if nargout > 1
    % compute inverse of sigma2 using binomial inverse theorem    
    d2 = d.^2;
    diag_ = diag(1 ./ d2);
    tmp = (b' * diag_ * b + eye(k_)) \ (b' * diag_);
    invsigma2 = diag_ - (diag_ * b * tmp);
        
    if Supplied.Grad.false
        % compute numerical gradient
        f = @(theta_) entropy(theta_, mu_pri, invsigma2_pri, n_, k_);
        grad = NumGrad(f, theta); 
    else       
        % compute analytical gradient
        grad_mu = invsigma2_pri * (mu - mu_pri);
        v = (invsigma2_pri - invsigma2);
        q = v * b;
        grad_b = q(:);
        grad_d = diag(v) .* d;
        grad = [grad_mu; grad_b; grad_d];
    end
end

% Hessian
if nargout > 2       
    if Supplied.Hess.false
        % compute numerical Hessian
        f = @(theta_) entrpoy(theta_, mu_pri, invsigma2_pri, n_, k_);
        hess = NumHess(f, theta); 
    else
        % compute analytical Hessian
        if nargin < 6 || isempty(Supplied.matrix)
           Supplied.matrix = SupplyFEP(N, K, Supplied);
        end
        a = sparse(1 : k_, 1 : k_, 1); 
        v = (invsigma2_pri - invsigma2);
        d = diag(d);
        grad2_mumu = invsigma2_pri;
        grad2_dd = (2 * d * invsigma2) .* (invsigma2 * d) + diag(diag(v));
        grad2_bd = Kronecker(b' * invsigma2, invsigma2 * d) * 2 * Supplied.matrix.hm1;
        grad2_bb = Kronecker(b' * invsigma2* b, invsigma2) + Supplied.matrix.km * Kronecker(invsigma2 * b, b' * invsigma2) + Kronecker(a, v);
        hess = [grad2_mumu          zeros(n_, n_ * k_)   zeros(n_, n_)
                zeros(n_ * k_, n_)  grad2_bb             grad2_bd
                zeros(n_, n_)      (grad2_bd)'           grad2_dd];
                 
    end
end
end

function [mu, sigma2, b, d] = theta2param(theta, n_, k_)
% Reparametrization from theta matrix

%% code
id = 1 : n_;
mu = reshape(theta(id), [], 1);
id = (n_+1) : n_ + (n_*k_);
b = reshape(theta(id), n_, k_);
id = n_ + (n_*k_) + 1 : n_ * (2+k_);
d = reshape(theta(id), [], 1);

sigma2 = b * b' + diag(d.^2);

end

function [obj] = entropy(theta, mu_pri, invsigma2_pri, n_, k_)
% Relative entropy

%% code
[mu, sigma2] = theta2param(theta, n_, k_);
obj = 0.5 * (trace(sigma2 * invsigma2_pri) - log(det(sigma2 * invsigma2_pri)) + (mu - mu_pri)' * invsigma2_pri *(mu - mu_pri) - n_);
end