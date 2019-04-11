function [out] = ConstantRiskPortfolioEqualWeight(Sigmap, sigTarget, h0, prices, pttc, longShort)
%% Find constant risk portfolio for equal weight portfolio

N = size(Sigmap, 1);
w = ones(N, 1) / N;
h = w ./ prices;
if longShort
    h(N/2+1:end) = -h(N/2+1:end);
end

FUN = @(x) DiffVariances(x, h, Sigmap, sigTarget);
tic;
[lstar, dummy, exitflag] = fzero(FUN, 1);
cpu = toc;
if exitflag < 1
    disp('Non convergence');
end
hstar = abs(lstar) * h;
check1 = hstar' * Sigmap * hstar - sigTarget^2;

% Transaction costs
costs = prices * pttc;
if isempty(h0)
    tc = 0;
else
    tc = abs(hstar - h0)' * costs;
end

out = [];
out.h   = hstar;
out.tc  = tc;
out.cpu = cpu;
out.check = [];
out.check.ck1 = check1;

end

function [out] = DiffVariances(x, h, Sigmap, sigTarget)

h_ = x * h;
out = (h_' * Sigmap * h_) - sigTarget^2;

end