% This script show a simple implementations of the MRE approach under 
% normal base and (in)equality views on linear combinations of first moments

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29; 0.33]; % base expectations
sig2_pri = [0.18	0.11	0.13
            0.11	0.23	0.16
            0.13	0.16	0.23]; % base covariance
eta = -0.01;       
     
z_mu_ineq = [1	-1	 0
             0	 1	-1]; % pick matrix (inequality) 
eta_mu_ineq = eta * ones(2, 1);    
    
z_mu_eq = [0, 0, 1]; % pick matrix (equality)
eta_mu_eq = 0.3;

z_mu = [z_mu_ineq; z_mu_eq];
eta_mu_view = [eta_mu_ineq; eta_mu_eq];

%% Compute optimal Lagrange multipliers, updated expectation and covariance
[theta_mu_view, mu_pos, sig2_pos] = FitMREIneq1MomentsN(z_mu_ineq, z_mu_eq, eta_mu_view, mu_pri, sig2_pri);

%% Views check
z_mu * mu_pos < eta_mu_view

