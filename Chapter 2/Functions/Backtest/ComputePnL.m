function [pnl] = ComputePnL(h, pnl_price, tc)
% Compute the PnL of a portfolio

pnl = pnl_price' * h - tc;

end