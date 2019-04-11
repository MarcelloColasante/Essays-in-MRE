function  [mu, sig2] = Theta2Norm(theta_mu, theta_sig2)
% Compute the expetcation and covariance of a normal distribution from its
% canonical coordinates
%  INPUTS
%   theta_mu   [vector]   n_ x 1 canonical coordinates wrt mu
%   sig2       [matrix]   n_ x n_ canonical coordinates wrt sig2
%  OUTPUTS
%   mu         [vector]   n_ x 1 expectation
%   sig2       [matrix]   n_ x n_ covariance
%%
mu = -0.5 * (theta_sig2 \ theta_mu);
sig2 = -0.5 * inv(theta_sig2);
end