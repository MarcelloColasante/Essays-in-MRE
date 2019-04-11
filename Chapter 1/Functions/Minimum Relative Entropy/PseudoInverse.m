function  z_dag = PseudoInverse(z, sig2)
% Compute the pseudo-inverse of z weighted via inv(sig2)
%  INPUTS
%   z          [matrix]   k_ x n_ pick matrix
%   sig2       [matrix]   n_ x n_ weighting matrix
%  OUTPUTS
%   z_dag      [matrix]   n_ x k_ pseudo inverse
%%
z_dag = (z * sig2 * z') \ (z * sig2);
end