function [z_view, jac_z_view] = ExpViewFunc(x, z_mu)
% Compute value and respective Jacobian of a z_view function for linear 
% combination of 
%  INPUTS
%   x               [vector] (n_ x 1) argument
%   z_mu            [vector]  (k_ x n_)  pick matrix
%  OUTPUTS
%   z_view          [scalar]   view function at x
%   jac_z_view      [vector]   (k_ x n_) Jacobian matrix of view function at x
%% Code
z_view = z_mu * x; % value

if nargout > 1
jac_z_view = z_mu; % Jacobian
end
end