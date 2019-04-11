function [M, S] = Log2Lin(Mu, Sigma)
%% Map moments of log-returns to linear returns
%  INPUTS
%   Mu    : [vector] (N x 1)
%   Sigma : [matrix] (N x N)
%  OUTPUTS
%   M     : [vector] (N x 1)
%   S     : [matrix] (N x N)

M = exp(Mu + (1/2) * diag(Sigma)) - 1;
S = exp(Mu + (1/2) * diag(Sigma)) * exp(Mu + (1/2) * diag(Sigma))' .* (exp(Sigma) - 1);

end