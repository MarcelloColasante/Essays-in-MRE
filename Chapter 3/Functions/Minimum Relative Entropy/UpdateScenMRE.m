function X = UpdateScenMRE(t, z_view, logf_pri, x_0, j_)
% Compute the HMC scenarios for an exponential family distribution under 
% generic canonical coordinates, sufficent statistics and base
%  INPUTS
%   t               [vector]   (k_ x 1)  canonical coordinates (Lagrangian multipliers)
%   z_view          [function] sufficient statistics (view function)
%   logf_pri        [function] base log-pdf (up to normalization constant)
%   x_0             [vector]   (n_ x 1) initial guess (MAP)
%   j_              [scalar]   number of scenarios
%  OUTPUTS
%   X               [matrix]   (n_ x j_)  panel matrix of HMC scenarios
%% Code
logpdf = @(x) ExpFamilyLogPdf(x, t, z_view, logf_pri); % exponential family log-pdf (up to normalization constant)
smp = hmcSampler(logpdf,x_0);
smp = tuneSampler(smp);
X = drawSamples(smp,'NumSamples',j_,'StartPoint',x_0); % HMC scenarios
X = X';
end