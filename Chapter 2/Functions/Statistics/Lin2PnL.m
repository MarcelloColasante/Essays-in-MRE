function [mup_t, sigma2p_t] = Lin2PnL(price_t, mu_t, sigma2_t)

mup_t = price_t .* mu_t;
sigma2p_t = diag(price_t) * sigma2_t * diag(price_t);
sigma2p_t = (sigma2p_t + sigma2p_t')/2; % ensure matrix is symmetric
end