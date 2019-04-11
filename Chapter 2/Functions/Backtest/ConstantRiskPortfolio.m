function [out] = ConstantRiskPortfolio(mup, Sigmap, sigTarget, wbound, h0, h1N, prices, pttc)
%% Find constant risk portfolio (max expected return given a level of risk)

cvx_clear; echo off;

% Exp and Sigma in prices
N = length(mup);
Sigmap = (Sigmap + Sigmap') * 0.5;
tSigmaph1N = (Sigmap * h1N)';

n = 5 * N; % problem dimension
costs = prices * pttc;
cp = costs;
cm = costs;
p  = prices;
A  = chol(Sigmap, 'upper');
zN = zeros(N, 1);

tic;
%x = [h; hp; hm; dhp; dhm];
cvx_begin quiet
    variable x(n)
    h   = x(1:N);
    hp  = x((N+1):2*N);
    hm  = x((2*N)+1:3*N);
    dhp = x((3*N)+1:4*N);
    dhm = x((4*N)+1:5*N);
    maximize( mup' * h - cp' * dhp - cm' * dhm )
    subject to
        p' * hp == p' * hm;
        dhp - dhm == h - h0;
        h - hp + hm == zN;
        %hp + hm  <= ((hp + hm)' * p) * wbound ./p;
        hp  >= zN;
        hp  <= ((hp + hm)' * p) * wbound ./p;
        hm  >= zN;
        hm  <= ((hp + hm)' * p) * wbound ./p;
        dhp >= zN;
        dhm >= zN;
        norm(A * h, 2) <= sigTarget;
cvx_end
cpu = toc;
    
% Convert solution into h   
hstar  = x(1:N);
check1 = hstar' * Sigmap * hstar - sigTarget^2;
check2 = tSigmaph1N * hstar;

% Transaction costs
if all(h0 == 0)
    tc = 0;
else
    tc = abs(hstar - h0)' * costs;
end

out = [];
out.h   = hstar;
out.tc  = tc;
out.check = [];
out.check.ck1 = check1;
out.check.ck2 = check2;
out.check.ck3 = cvx_status;
out.cpu = cpu;

end