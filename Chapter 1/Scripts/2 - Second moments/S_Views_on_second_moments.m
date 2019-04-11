% This script show a simple implementations of the MRE approach under 
% normal base and views on linear combinations of second non-central moments

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29; 0.33]; % base expectation
sig2_pri = [0.18	0.11	0.13
            0.11	0.23	0.16
            0.13	0.16	0.23]; % base covariance

theta_sig2_view_true = [-1.83  -2.82
                        -2.82  -3.13]; % true optimal Lagrange multipliers       
        
z_sig = [1	0 -1
         0	1  1]; % pick matrix 

%% Compute true updated canonical coordinates
[theta_mu_pri, theta_sig2_pri] = Norm2Theta(mu_pri, sig2_pri);
theta_mu_pos_true = theta_mu_pri;
theta_sig2_pos_true = theta_sig2_pri + z_sig' * theta_sig2_view_true * z_sig;

%% Compute true updated parameters
[mu_pos_true, sig2_pos_true] = Theta2Norm(theta_mu_pos_true, theta_sig2_pos_true);
eta_sig2_view = z_sig * (sig2_pos_true + mu_pos_true * mu_pos_true') * z_sig';  

%% Compute optimal Lagrange multipliers, updated expectation and covariance
threshold= 1e-7;
[theta_sig2_view, mu_pos, sig2_pos] = FitMRE2MomentsN(z_sig, eta_sig2_view, mu_pri, sig2_pri, threshold);

%% Check
norm(theta_sig2_view - theta_sig2_view_true, 'fro')
norm(mu_pos - mu_pos_true)
norm(sig2_pos - sig2_pos_true, 'fro')