function [lpdf, glpdf] = NormLogPdf(x, mu, inv_sig2)
% Compute the log-pdf and respective gradient for a normal distribution
% and base
%  INPUTS
%   x               [vector] (n_ x 1) argument
%   mu              [vector]  (n_ x 1)  location (expectation)
%   inv_sig2        [matrix]  (n_ x n_) inverse squared-dispersion (inverse covariance)
%  OUTPUTS
%   lpdf           [scalar]   log-pdf at x
%   glpdf          [vector]   (n_ x 1) gradient of log-pdf at x
%% Code
lpdf = - 0.5 * ((x-mu)' * inv_sig2)*(x-mu); % log-pdf  
glpdf =  - inv_sig2 * (x-mu); % gradient

end