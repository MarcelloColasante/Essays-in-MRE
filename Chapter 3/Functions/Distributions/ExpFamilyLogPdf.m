function [lpdf, glpdf] = ExpFamilyLogPdf(x, theta_view, z_view, logf_pri)
% Compute the log-pdf and respective gradient for an exponential family 
% distribution under generic canonical coordinates, sufficent statistics
% and base
%  INPUTS
%   x               [vector] (n_ x 1) argument
%   theta_view      [vector]  (k_ x 1)  canonical coordinates (optimal Lagrangian multipliers)
%   z_view          [function] sufficient statistics (view function)
%   logf_pri        [function] base log-pdf (up to normalization constant)
%  OUTPUTS
%   lpdf            [scalar]   log-pdf at x
%   glpdf           [vector]   (n_ x 1) gradient of log-pdf at x
%% Code
[view, jac_view] = z_view(x);
[lpdf_2, glpdf_2] = logf_pri(x);

lpdf_1 = theta_view'*view;
lpdf = lpdf_1 + lpdf_2; % log-pdf

glpdf_1 = jac_view' * theta_view;
glpdf = glpdf_1 + glpdf_2; % gradient

end