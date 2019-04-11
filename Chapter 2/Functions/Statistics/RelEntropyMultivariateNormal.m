function [obj] = RelEntropyMultivariateNormal(mu, sigma2, mu_pri, sigma2_pri)
% This function computes the relative entropy between normal distributions 
  
%  INPUTS
%   mu          [vector]: (n_ x 1) vector of expectations
%   sigma2      [matrix]: (n_ x n_) covariance matrix
%   mu_pri      [vector]: (n_ x 1) vector of expectations (prior)
%   sigma2_pri  [matrix]: (n_ x n_) covariance matrix (prior)


%  OUTPUTS
%   obj         [scalar]: entropy

%% code	
n_ = size(sigma2_pri, 1);
invsigma2_pri = sigma2_pri \ eye(n_);

mu__ = (mu - mu_pri);
obj = 0.5 * mu__' * (invsigma2_pri * mu__);
obj = obj + 0.5 * logdet(sigma2_pri) - 0.5 * logdet(sigma2);
obj = obj + 0.5 * sum(diag(invsigma2_pri * sigma2));
obj = obj - 0.5 * n_;

end

function [v] = logdet(a)
% Fast logarithm-determinat of large matrix

%% code
try
    v = 2 * sum(log(diag(chol(a))));
catch  %#ok<CTCH>
    [dummy, u, p] = lu(a);  %#ok<ASGLU>
    du = diag(u);
    c = det(p) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));
    v = real(v);
end

end