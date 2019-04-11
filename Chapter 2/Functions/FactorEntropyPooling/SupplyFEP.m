function [matrix] = SupplyFEP(n_, k_, Supplied)
% Compute input matrices for derivatives of entropy

%% code
if nargin < 3 || isempty(Supplied); Supplied = []; end
if ~isfield(Supplied, 'Grad') || isempty(Supplied.Grad); Supplied.Grad.on = false; Supplied.Grad.false = false; end
if ~isfield(Supplied.Grad, 'on') || isempty(Supplied.Grad.on); Supplied.Grad.on = false; end
if ~isfield(Supplied.Grad, 'false') || isempty(Supplied.Grad.false); Supplied.Grad.false = false; end
if ~isfield(Supplied, 'Hess') || isempty(Supplied.Hess); Supplied.Hess.on = false; Supplied.Hess.false = false; end
if ~isfield(Supplied.Hess, 'on') || isempty(Supplied.Hess.on); Supplied.Hess.on = false; end
if ~isfield(Supplied.Hess, 'false') || isempty(Supplied.Hess.false); Supplied.Hess.false = false; end


matrix = [];

if Supplied.Grad.on  && ~Supplied.Grad.false
    Supplied.Hess.on = false;
    
    e = sparse(1 : n_, 1 : n_, 1);    
    matrix.hm = diag(e(:));
    
elseif Supplied.Hess.on && ~Supplied.Hess.false
    
    e = sparse(1 : n_, 1 : n_, 1);
    a = sparse(1 : k_, 1 : k_, 1); 
    
    matrix.hm = diag(e(:));
    matrix.hm1 = sparse([], [], [], n_^2, n_);
    matrix.hm2 = sparse([], [], [], n_, n_^2);
    matrix.km  = sparse([], [], [], k_*n_, k_*n_);
    matrix.km1 = sparse([], [], [], k_*n_, k_*n_^2);
    
    for k = 1 : k_   
        matrix.km = matrix.km + kronecker(kronecker(a(:, k), e), a(:, k)');
    end 
    
    for n = 1 : n_    
        matrix.hm1 = matrix.hm1 + kronecker(e(:, n), diag(e(:, n)));
        matrix.hm2 = matrix.hm2 + kronecker(e(:, n)', diag(e(:, n)));
        matrix.km1 = matrix.km1 + kronecker(kronecker(e(:, n)', a), diag(e(:, n)));
    end
    
end

end




