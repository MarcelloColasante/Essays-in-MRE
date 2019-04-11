% This script show a simple implementations of the MRE approach under 
% normal base and views on same linear combinations of first and second non-central moments

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29; 0.33]; % base expectation
sig2_pri = [0.18	0.11	0.13
            0.11	0.23	0.16
            0.13	0.16	0.23]; % base covariance

z = [1 -1 0
     0 1 -1]; % pick matrix for expectations
theta_mu_view_true = [5.71; 0.38]; % true posterior Lagrange multipliers for expectations

theta_sig2_view_true = [-1.83  -2.82
                        -2.82  -3.13]; % true optimal Lagrange multipliers for second moments

%% Compute true updated canonical coordinates
[theta_mu_pri, theta_sig2_pri] = Norm2Theta(mu_pri, sig2_pri);
theta_mu_pos_true = theta_mu_pri + z' * theta_mu_view_true;
theta_sig2_pos_true = theta_sig2_pri + z' * theta_sig2_view_true * z;

%% Compute true updated parameters
[mu_pos_true, sig2_pos_true] = Theta2Norm(theta_mu_pos_true, theta_sig2_pos_true);
eta_mu_view = z * mu_pos_true;
eta_sig2_view = z * (sig2_pos_true + mu_pos_true * mu_pos_true') * z'; 

%% Compute optimal Lagrange multipliers
sig2_view = eta_sig2_view - eta_mu_view * eta_mu_view';
theta_mu_view = (sig2_view \ eta_mu_view - z * sig2_pri * z' \ (z * mu_pri));

theta_sig2_view = 0.5 * (inv(z * sig2_pri * z') - inv(sig2_view));

%% Compute updated expectation and covariance
z_dag = PseudoInverse(z, sig2_pri);
mu_pos = mu_pri + z_dag' * (eta_mu_view - z * mu_pri);

sig2_pos = sig2_pri + z_dag' * (sig2_view - z * sig2_pri * z') * z_dag;

%% Check
norm([theta_mu_view; theta_sig2_view(:)] - [theta_mu_view_true; theta_sig2_view_true(:)])
norm(mu_pos - mu_pos_true)
norm(sig2_pos - sig2_pos_true, 'fro')

