function [p, z_JK] = PvalueJK(mu, sig2, t_)
% P-value of the difference of Sharpe ratios, using the approach by Jobson and Korkie (1981) and Memmel (2003).

a = sig2(1,1) * sig2(2,2) - sqrt(sig2(1,1) * sig2(2,2)) * sig2(1,2);
b = sig2(1,1) * mu(2)^2 + sig2(2,2) * mu(1)^2;
c = sig2(1,2) * (mu(1) * mu(2))/(sqrt(sig2(1,1) * sig2(2,2)));

theta = (2 * a + 0.5 * b - c)/ t_;

z_JK = (sqrt(sig2(1,1)) * mu(2) - sqrt(sig2(2,2)) * mu(1)) / sqrt(theta);

p = 2 * (1 - normcdf(abs(z_JK)));
end