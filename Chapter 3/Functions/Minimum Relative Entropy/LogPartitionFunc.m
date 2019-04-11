function psi = LogPartitionFunc(t, Z_pri, p_pri)
% Compute the sample log-partition function
%  INPUTS
%   t               [vector]   (k_ x 1) Lagrange multipliers
%   Z_pri           [matrix]   (n_ x j_) panel matrix of base view scenarios
%   p_pri           [vector]   (1 x j_) vector of base probabilities
%  OUTPUTS
%   psi             [scalar]   log-partition function at t
%% Code
x = t' * Z_pri + log(p_pri);
psi = LogSumExp(x);
end