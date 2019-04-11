function p = UpdateProbMRE(t, Z_pri, p_pri)
% Compute the probabilities via exponential twisting of base probabilities
%  INPUTS
%   t               [vector]   (k_ x 1) Lagrange multipliers
%   Z_pri           [matrix]   (n_ x j_) panel matrix of base view scenarios
%   p_pri           [vector]   (1 x j_) vector of base probabilities
%  OUTPUTS
%   p               [vector]   (1 x j_) vector of exponentially twisted probabilities
%% Code
x = t' * Z_pri + log(p_pri);
psi = LogPartitionFunc(t, Z_pri, p_pri);
p = exp(x - psi);
p(p < 1e-32) = 1e-32; % truncate tiny probabilities
p = p / sum(p);
end