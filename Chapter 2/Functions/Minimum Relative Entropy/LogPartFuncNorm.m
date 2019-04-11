function psi = LogPartFuncNorm(mu, sig2)
% Compute log-partition function of a normal distribution
%  INPUTS
%
%  OUTPUTS
%
psi = 0.5 * (mu' * (sig2 \ mu) - logdet(inv(sig2))); % use fast log det
end

function [v] = logdet(x)
% Fast logarithm-determinant of large matrix

%% code
try
    v = 2 * sum(log(diag(chol(x))));
catch  %#ok<CTCH>
    [dummy, u, p] = lu(x);  %#ok<ASGLU>
    du = diag(u);
    c = det(p) * prod(sign(du));
    v = log(c) + sum(log(abs(du)));
    v = real(v);
end

end