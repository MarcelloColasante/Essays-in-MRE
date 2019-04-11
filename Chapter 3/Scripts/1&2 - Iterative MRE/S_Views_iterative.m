% This script show a simple implementations of the non-parametric and
% iterative MRE approach for a normal base distribution with views on 
% linear combinations of expectations

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29]; % base expectations
sig2_pri = [0.18	0.11	
            0.11	0.23]; % base covariance
n_ = length(mu_pri);

z_mu = [1	-1]; % pick matrix 
eta_mu = 1.02; % features 

j_ = 100000; % number of simulations

%% Compute true optimal Lagrange multipliers
theta_mu_view_true = (z_mu * sig2_pri * z_mu') \ (eta_mu - z_mu * mu_pri);

%% Compute true updated features
mu_pos_true = mu_pri + sig2_pri * z_mu' * theta_mu_view_true;
sig2_pos_true = sig2_pri;
X_pos_true = mvnrnd(mu_pos_true, sig2_pos_true, j_); 
X_pos_true = X_pos_true'; % updated simulations

%% Estimate optimal Lagrange multipliers
inv_sig2_pri = inv(sig2_pri);
logf_pri = @(x) NormLogPdf(x, mu_pri, inv_sig2_pri); % base distribution

p_pri = ones(1,j_)/j_; % base probabilities

z_view = @(x) ExpViewFunc(x, z_mu); % view function

X_pri = mvnrnd(mu_pri, sig2_pri, j_);
X_pri = X_pri'; % base simulations

threshold = 1e-2;
[X_pos, p_pos, theta_mu_view, mh_pos, mgrad_pos, ens_pos] = IterMRE(z_view, [], eta_mu, logf_pri, X_pri, p_pri, j_, threshold);

%% Compute non-parametric features
theta_mu_view_np = theta_mu_view(1); % non-parametric estimate
[mu_pos_np, sig2_pos_np] = FPmeancov(X_pos(:, :, 1), p_pos(1, :));
X_pos_np = SampleScenProbDistribution(X_pos(:, :, 1), p_pos(1, :), j_); % updated simulations
intenisty_np = norm(theta_mu_view_np)
ens_np = ens_pos(1)

%% Compute one-step iterarive features
theta_mu_view_iter = theta_mu_view(2); % one-step iterative estimate
[mu_pos_iter, sig2_pos_iter] = FPmeancov(X_pos(:,:, 2), p_pos(2, :));
intenisty_iter = norm(theta_mu_view_iter - theta_mu_view_np)
ens_iter = ens_pos(2)

%% Non-parametric Check
norm(theta_mu_view_np - theta_mu_view_true)
norm(mu_pos_np - mu_pos_true)
norm(sig2_pos_np - sig2_pos_true, 'fro')

%% One-step iterative Check
norm(theta_mu_view_iter - theta_mu_view_true)
norm(mu_pos_iter - mu_pos_true)
norm(sig2_pos_iter - sig2_pos_true, 'fro')

%% Figures
% color setting
grey_range = [0 0.01 0.8];
orange = [1 0.4 0];
green = [0.1 0.8 0];
dark = [0.2 0.2 0.2];
blue = [0 0.4 1];

% contour settings
r = 1;
x1 = r * min(X_pos_np(1,:)):.01: r * max(X_pos_np(1,:)); %// x axis
x2 = r * min(X_pos_np(2,:)):.01: r * max(X_pos_np(2,:)); %// y axis

[X1, X2] = meshgrid(x1, x2); 
X3 = mvnpdf([X1(:) X2(:)], mu_pos_true', sig2_pos_true); 
X3 = reshape(X3, length(x2),length(x1)); 

%% Non-parametric MRE scatter
figure(); set(gcf,'color','w');
[CM, C] = ColorCodedFP(p_pos(1, :), [], [], grey_range, 0, 18, [17 5]);
colormap(CM);
scatter(X_pos(1,:, 1), X_pos(2,:, 1), 3, C, '*'); hold on;
[C,h] = contour(x1,x2, X3); hold on
h.LineWidth = 2;
h.LineColor = green;
xlabel('X_1',  'FontSize', 24);
ylabel('X_2',  'FontSize', 24);
set(gca, 'FontSize', 16)
legend('NP MRE','True');

%% Iterative MRE scatter
figure(); set(gcf,'color','w');
[CM, C] = ColorCodedFP(p_pos(2, :), [], [], grey_range, 0, 18, [17 5]);
colormap(CM);
scatter(X_pos(1,:, 2), X_pos(2,:, 2), 3, C, '*'); hold on;
[C,h] = contour(x1,x2, X3); hold on
h.LineWidth = 2;
h.LineColor = green;
xlabel('X_1',  'FontSize', 24);
ylabel('X_2',  'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
legend('One-step MRE','True');




