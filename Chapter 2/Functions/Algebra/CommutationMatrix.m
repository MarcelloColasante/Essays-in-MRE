function km = CommutationMatrix(n_, k_)
% Compute the k_*n_ x k_*n_ commutation matrix
%  INPUTS
%
%  OUTPUTS
%

e = sparse(1 : n_, 1 : n_, 1); % e = eye(n_);
a = sparse(1 : k_, 1 : k_, 1); % a = eye(k_);
km  = sparse([], [], [], k_*n_, k_*n_); % km = zeros(k_*n_, k_*n_);

for k = 1 : k_   
    km = km + kronecker(kronecker(a(:, k), e), a(:, k)');
end 
end