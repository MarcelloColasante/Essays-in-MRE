function [mu, sigma2] = EwmaFP(epsi, lambda)
% This function computes the exponentially weighted moving average (EWMA) 
% expectations and covariances for time series of invariants
%  INPUTS
%   epsi   : [matrix] (n_ x t_) matrix of invariants observations
%   lambda : [scalar]           half-life parameter
%  OUTPUTS
%   mu     : [vector] (n_ x 1)  EWMA expectations
%   sigma2 : [matrix] (n_ x n_) EWMA covariances

% For details on the exercise, see here .
%% Code
[~, t_] = size(epsi);
p = exp(-lambda*(t_-1:-1:0))/sum(exp(-lambda*(t_-1:-1:0))); % flexible probabilities
[mu, sigma2] = FPmeancov(epsi, p);