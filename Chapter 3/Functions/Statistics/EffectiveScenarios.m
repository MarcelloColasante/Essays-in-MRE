function ens = EffectiveScenarios(p, Type)
% This function computes the Effective Number of Scenarios of Flexible 
% Probabilities via different types of functions
%  INPUTS
%   p       : [vector] (1 x j_) vector of Flexible Probabilities
%   Type    : [struct] type of function: 'ExpEntropy', 'GenExpEntropy' 
%  OUTPUTS
%   ens     : [scalar] Effective Number of Scenarios
% NOTE: 
% The exponential of the entropy is set as default, otherwise
% Specify Type.ExpEntropy.on = true to use the exponential of the entropy
% or
% Specify Type.GenExpEntropy.on = true and supply the scalar 
% Type.ExpEntropy.g to use the generalized exponential of the entropy

if nargin < 2 || isempty(Type); Type.ExpEntropy.on = true; end
if ~isfield(Type, 'ExpEntropy') || isempty(Type.ExpEntropy); Type.ExpEntropy.on = false; end
if ~isfield(Type, 'GenExpEntropy') || isempty(Type.GenExpEntropy); Type.GenExpEntropy.on = false; end
%% Code

if Type.ExpEntropy.on
    p(p==0)=10^-250; %avoid log(0) in ens computation
    ens = exp(-p * log(p'));
elseif Type.GenExpEntropy.on
    ens = sum(p .^ Type.GenExpEntropy.g) ^ (-1 / (Type.GenExpEntropy.g - 1));
end