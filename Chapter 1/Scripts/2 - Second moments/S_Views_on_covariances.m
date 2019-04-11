% This script show a simple implementations of the MRE approach under 
% normal base and views on linear combinations of covariances
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
k_ = size(z_sig, 1);

%% Compute true updated canonical coordinates
[theta_mu_pri, theta_sig2_pri] = Norm2Theta(mu_pri, sig2_pri);
theta_mu_pos_true = theta_mu_pri;
theta_sig2_pos_true = theta_sig2_pri + z_sig' * theta_sig2_view_true * z_sig;

%% Compute true updated parameters
[mu_pos_true, sig2_pos_true] = Theta2Norm(theta_mu_pos_true, theta_sig2_pos_true);
sig2_view = z_sig * sig2_pos_true * z_sig';  

%% Compute optimal Lagrange multipliers
theta_sig2_view = 0.5 * (inv(z_sig * sig2_pri * z_sig') - inv(sig2_view));

%% Compute updated expectation and covariance
z_sig_dag = PseudoInverse(z_sig, sig2_pri);
mu_pos = mu_pri + z_sig_dag' * ((sig2_view /(z_sig * sig2_pri * z_sig')) - eye(k_)) * z_sig * mu_pri;

sig2_pos = sig2_pri + z_sig_dag' * (sig2_view - z_sig * sig2_pri * z_sig') * z_sig_dag;

%% Check
norm(theta_sig2_view - theta_sig2_view_true, 'fro')
norm(mu_pos - mu_pos_true)
norm(sig2_pos - sig2_pos_true, 'fro')