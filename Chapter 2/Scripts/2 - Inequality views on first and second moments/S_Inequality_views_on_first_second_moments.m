% This script show a simple implementations of the MRE approach under 
% normal base and (in)equality views on linear combinations of first and 
% second moments

clc; clear; close all;

%% Input parameters
n_ = 6;
k_mu_ineq = 3;
k_mu_eq = 2;
k_sig_ineq = k_mu_eq;

mu_pri = [-1.14; 0.10; 0.72; 2.58; -0.66; 0.18]; % base expectations
sig2_pri = [1.0000    0.2453    0.2286   -0.6345    0.0546   -0.5702
           0.2453    1.0000    0.3304    0.4226   -0.6301    0.2885
           0.2286    0.3304    1.0000   -0.2960   -0.7910    0.0330
          -0.6345    0.4226   -0.2960    1.0000   -0.2038    0.8045
           0.0546   -0.6301   -0.7910   -0.2038    1.0000   -0.3057
          -0.5702    0.2885    0.0330    0.8045   -0.3057    1.0000]; % base covariance

% Inequality views on expecations
z_mu_ineq = [-1.5771    0.0335    0.3502   -0.2620   -0.8314   -0.5336
              0.5080   -1.3337   -0.2991   -1.7502   -0.9792   -2.0026
              0.2820    1.1275    0.0229   -0.2857   -1.1564    0.9642]; % pick matrix
eta_mu_ineq = [0.5201; -0.0200; -0.0348]; % features 

% Equality views on expecations
z_mu_eq = [-0.7982   -0.1332    1.3514   -0.5890   -0.8479    2.5260
            1.0187   -0.7145   -0.2248   -0.2938   -1.1201    1.6555]; % pick matrix
eta_mu_eq = [0.3075; -1.2571];  % features


z_mu = [z_mu_ineq; z_mu_eq];
eta_mu_view = [eta_mu_ineq; eta_mu_eq];

% Inequality views on covariances
z_sig_ineq = z_mu_eq; % pick matrix
sig2_view = [1.0000   -0.5720
           -0.5720    1.0000];
eta_sig2_view = sig2_view + eta_mu_eq * eta_mu_eq'; % features

%% Compute optimal Lagrange multipliers, updated expectation and covariance
km = CommutationMatrix(n_, n_);
[theta_mu_view, theta_sig2_view, mu_pos, sig2_pos] = FitMREIneq12MomentsN(z_mu_ineq, z_mu_eq, z_sig_ineq, eta_mu_view,  eta_sig2_view, mu_pri, sig2_pri, km);

%% Views check
z_mu_ineq * mu_pos
eta_mu_ineq
norm(z_mu_eq * mu_pos - eta_mu_eq)
z_sig_ineq * (sig2_pos + mu_pos * mu_pos') * z_sig_ineq' 
eta_sig2_view
