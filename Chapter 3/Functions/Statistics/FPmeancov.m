function [m, s2]=FPmeancov(x,p)
% This function computes the mean and covariance matrix of a Flexible Probabilities distribution 
% INPUT
% x    :[matrix] (i_ x t_) scenarios
% p    :[vector] ( 1 x t_) Flexible Probabilities
% OUTPUT
% m    :[vector] (i_ x 1)  mean
% s2   :[matrix] (i_ x i_) covariance matrix
%% Code
if size(p,2)==1; p=p'; end

[i_,t_]=size(x);
m = x*p'; % mean
X_cent = x - repmat(m,1,t_);
s2 =(X_cent.*repmat(p,i_,1))*X_cent'; % covariance matrix
s2 = (s2 + s2')/2; % ensure true symmetric outcome
end