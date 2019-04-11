% This script show a simple implementations of the MRE approach under 
% normal base and views on linear combinations of first moments

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29; 0.33]; % base expectations
sig2_pri = [0.18	0.11	0.13
            0.11	0.23	0.16
            0.13	0.16	0.23]; % base covariance
        
theta_mu_view_true = [5.71; 0.38]; % true optimal Lagrange multipliers
     
z_mu = [1	-1	0
        0	1	-1]; % pick matrix 

%% Compute true updated canonical coordinates
[theta_mu_pri, theta_sig2_pri] = Norm2Theta(mu_pri, sig2_pri);
theta_mu_pos_true = theta_mu_pri + z_mu' * theta_mu_view_true;
theta_sig2_pos_true = theta_sig2_pri;

%% Compute true updated parameters
[mu_pos_true, sig2_pos_true] = Theta2Norm(theta_mu_pos_true, theta_sig2_pos_true); 
eta_mu_view = z_mu * mu_pos_true;  

%% Compute optimal Lagrange multipliers
theta_mu_view = (z_mu * sig2_pri * z_mu') \ (eta_mu_view - z_mu * mu_pri);

%% Compute updated expectation and covariance
z_mu_dag = PseudoInverse(z_mu, sig2_pri);
mu_pos = mu_pri + z_mu_dag' * (eta_mu_view - z_mu * mu_pri);

sig2_pos = sig2_pri;

%% Check
norm(theta_mu_view - theta_mu_view_true)
norm(mu_pos - mu_pos_true)
norm(sig2_pos - sig2_pos_true, 'fro')