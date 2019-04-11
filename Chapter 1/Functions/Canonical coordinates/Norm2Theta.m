function  [theta_mu, theta_sig2] = Norm2Theta(mu, sig2)
% Compute the canonical coordinates of a normal distribution from its
% expetcation and covariance
%  INPUTS
%   mu         [vector]   n_ x 1 expectation
%   sig2       [matrix]   n_ x n_ covariance
%  OUTPUTS
%   theta_mu   [vector]   n_ x 1 canonical coordinates wrt mu
%   sig2       [matrix]   n_ x n_ canonical coordinates wrt sig2
%%
theta_mu = sig2\ mu;
theta_sig2 = -0.5 * inv(sig2);
end