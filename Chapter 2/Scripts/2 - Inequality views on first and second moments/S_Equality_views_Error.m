% This script compares the numerical and anlaytical implementations of MRE
% under equality views on linear combinations of first two moments

clc; clear; close all;

%% Input parameters
n_ = 30; % market dimension
k_mu = 3; % number of views on expectations
k_sig = 2; % number of views on covariances

j_ = 100; % number of draws

for j = 1: j_
mu_pri = zeros(n_, 1); % base expectations
sig_pri = 1./randn(n_, n_);
[~, sig2_pri] = cov2corr(sig_pri * sig_pri'); % base covariances

% Equality views on expecations
z_mu = randn(k_mu, n_);
eta_mu_view = randn(k_mu, 1);  

% Equality views on variances
z_sig = z_mu(end-1 : end, :);
sig_view = 1./randn(k_sig, k_sig);
[~, sig2_view] = cov2corr(sig_view * sig_view');
eta_sig2_view = sig2_view + eta_mu_view(end-1:end) * eta_mu_view(end-1:end)'; 

%% Compute optimal Lagrange multipliers, updated expectation and covariance
km = CommutationMatrix(n_, n_);
[theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMREEq12MomentsN(z_mu, z_sig, eta_mu_view,  eta_sig2_view, mu_pri, sig2_pri, km);
[theta_mu_view_true, theta_sig2_view_true, mu_pos_true, sig2_pos_true] = FitMREExpCovN(z_mu, z_sig, eta_mu_view, sig2_view, mu_pri, sig2_pri);

%% Convergence check
err_theta_mu_view(j) = norm(theta_mu_view - theta_mu_view_true);
err_theta_sig2_view(j) = norm(theta_sig2_view - theta_sig2_view_true, 'fro');
err_mu_pos(j) = norm(mu_pos - mu_pos_true);
err_sig2_pos(j) = norm(sig2_pos - sig2_pos_true, 'fro');
end

%% Figures
% settings
nbins = round(10*log(j_));

figure(); set(gcf,'color','w');
subplot(2,2,1);
[counts,centers] = hist(err_theta_mu_view, nbins);
hm=bar(centers,counts);
set(hm, 'EdgeColor',  'none');
xlabel('Error on \theta for exp', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)
box off
subplot(2,2,2);
[counts,centers] = hist(err_theta_sig2_view, nbins);
hm=bar(centers,counts);
set(hm, 'EdgeColor',  'none');
xlabel('Error on \theta for cov', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)
box off
subplot(2,2,3);
[counts,centers] = hist(err_mu_pos, nbins);
hm=bar(centers,counts);
set(hm, 'EdgeColor',  'none');
xlabel('Error on exp', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)
box off
subplot(2,2,4);
[counts,centers] = hist(err_sig2_pos, nbins);
hm=bar(centers,counts);
set(hm, 'EdgeColor',  'none');
xlabel('Error on cov', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 24)
box off







