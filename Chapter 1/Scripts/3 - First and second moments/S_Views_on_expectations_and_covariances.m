% This script show a simple implementations of the MRE approach under 
% normal base and views on linear combinations of expectations and
% covariances

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29; 0.33]; % base expectation
sig2_pri = [0.18	0.11	0.13
            0.11	0.23	0.16
            0.13	0.16	0.23]; % base covariance

z_mu = [1 -1 0]; % pick matrix for expectations
theta_mu_view_true = 1.73; % true posterior Lagrange multipliers for expectations

z_sig = [0 1 -1]; % pick matrix for second moments
theta_sig2_view_true = 3.04; % true posterior Lagrange multipliers for second moments

%% Compute true updated canonical coordinates
[theta_mu_pri, theta_sig2_pri] = Norm2Theta(mu_pri, sig2_pri);
theta_mu_pos_true = theta_mu_pri + z_mu' * theta_mu_view_true;
theta_sig2_pos_true = theta_sig2_pri + z_sig' * theta_sig2_view_true * z_sig;

%% Compute true updated parameters
[mu_pos_true, sig2_pos_true] = Theta2Norm(theta_mu_pos_true, theta_sig2_pos_true);
mu_view = z_mu * mu_pos_true;
sig2_view = z_sig * sig2_pos_true * z_sig'; 

%% Compute optimal Lagrange multipliers, updated expectation and covariance
[theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMREExpCovN(z_mu, z_sig, mu_view, sig2_view, mu_pri, sig2_pri);

%% Check
norm([theta_mu_view; theta_sig2_view(:)] - [theta_mu_view_true; theta_sig2_view_true(:)])
norm(mu_pos - mu_pos_true)
norm(sig2_pos - sig2_pos_true, 'fro')

