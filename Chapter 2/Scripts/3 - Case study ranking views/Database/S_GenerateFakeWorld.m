clc; clear all; close all;

% Generate Black-Scoles prices
n_ = 30; % market dimension
t_ = 783; % time series length
lambda = log(2)/52; % HFP mean and covariance half-life

load db_DJIA.mat;

rets_its = diff(log(price)); % compound returns

% Mean and covariance matrix (compound returns)
[mu_C, sigma2_C] = EwmaFP(rets_its', lambda);

% Mean and covariance matrix (linear returns)
[mu_R, sigma2_R] = Log2Lin(mu_C, sigma2_C);

% Random walk dynamics (log-prices)
x = NaN(t_, n_);
x(1, :) = log(price(1, :));
epsi = mvnrnd(mu_C', sigma2_C, t_); % normal shocks

for t = 2 : t_
    x(t, :) = x(t-1, :) + epsi(t, :);
end

price = exp(x);
%time = [1 : t_]';

save db_FakeWorld.mat price time mu_R sigma2_R