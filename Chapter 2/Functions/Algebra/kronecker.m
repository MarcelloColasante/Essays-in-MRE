function x = kronecker(a,b)
%% Fast kronecker product
%  INPUTS 
%   a   : [matrix] i_ x j_ matrix
%   b   : [matrix] k_ x l_ matrix

%  OUTPUTS
%   x   : [matrix] i_*k_ x j_*l_ block matrix in which the (i,j)-th block is defined as a(i,j) * b

%   Author: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%   Version: 06/02/2011
%   Original name : kron

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[i_, j_] = size(a);
[k_, l_] = size(b);

if ~issparse(a) && ~issparse(b)
    
    % Both matrices are dense.
    a = reshape(a,[1 i_ 1 j_]);
    b = reshape(b,[k_ 1 l_ 1]);
    x = reshape(bsxfun(@times,a,b),[i_*k_ j_*l_]);
    
else
    
    % One of the matrices is sparse.
    [ia,ja,sa] = find(a);
    [ib,jb,sb] = find(b);
    ix = bsxfun(@plus,k_*(ia(:)-1).',ib(:));
    jx = bsxfun(@plus,l_*(ja(:)-1).',jb(:));
    
    % The @and operator is slightly faster for logicals.
    if islogical(sa) && islogical(sb)
        x = sparse(ix,jx,bsxfun(@and,sb(:),sa(:).'),i_*k_,j_*l_);
    else
        x = sparse(ix,jx,double(sb(:))*double(sa(:).'),i_*k_,j_*l_);
    end
    
end