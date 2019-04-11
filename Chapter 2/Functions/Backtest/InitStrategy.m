function [strat] = InitStrategy(time, N)

nBacktest = length(time) - 1;

strat = [];
strat.time   = time;
strat.pnl    = NaN(nBacktest, 1);
strat.pnl(1) = 0;
strat.s2     = NaN(nBacktest, 1);
strat.h_t    = zeros(N, 1);
strat.h      = NaN(N, nBacktest);
strat.mu     = NaN(N, nBacktest);
strat.sig    = NaN(N, nBacktest);
strat.rk     = NaN(N, nBacktest);
strat.check  = cell(nBacktest, 1);
strat.cpu    = NaN(nBacktest, 1);
strat.cpu2   = NaN(nBacktest, 1);

end