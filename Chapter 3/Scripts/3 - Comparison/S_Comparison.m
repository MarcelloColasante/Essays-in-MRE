% This script show a comparison of the estimators and loss ensuing from the
% non-parametric and iterative MRE approach for a normal base distribution 
% with views on linear combinations of expectations

clc; clear; close all;

%% Input parameters
mu_pri = [0.26; 0.29]; % base expectations
sig2_pri = [0.18	0.11	
            0.11	0.23]; % base covariance
n_ = length(mu_pri);

z_mu = [1	-1]; % pick matrix 
eta_mu = 1.02; % features 

j_ = 100000; % number of simulations
r_ = 100; % number of iterations

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

threshold = 1e-2;
theta_mu_view_np = NaN(1, r_);
theta_mu_view_iter = NaN(1, r_);
for r = 1: r_
r

X_pri = mvnrnd(mu_pri, sig2_pri, j_);
X_pri = X_pri'; % base simulations

[X_pos, p_pos, theta_mu_view, mh_pos, mgrad_pos, ens_pos] = IterMRE(z_view, [], eta_mu, logf_pri, X_pri, p_pri, j_, threshold);

ens_pos
theta_mu_view_np(r) = theta_mu_view(1);
theta_mu_view_iter(r) = theta_mu_view(end);
end

%% Compute non-parametric error, bias, inefficiency and loss
e_theta_np = mean(theta_mu_view_np);
bias_np = abs(e_theta_np - theta_mu_view_true)
inef_np = std(theta_mu_view_np)
er_np = bias_np^2 + inef_np^2
Loss_np = (theta_mu_view_np -theta_mu_view_true).^2;

%% Compute non-parametric error, bias, inefficiency and loss
e_theta_iter = mean(theta_mu_view_iter);
bias_iter = abs(e_theta_iter - theta_mu_view_true)
inef_iter = std(theta_mu_view_iter)
er_iter = bias_iter^2 + inef_iter^2
Loss_iter = (theta_mu_view_iter -theta_mu_view_true).^2;

%% Figures
% color setting
colhist = [.8 .8 .8];
orange = [1 0.4 0];
green = [0.1 0.8 0];
dark = [0.2 0.2 0.2];
blue = [0 0.4 1];

%% Esimators distributions
% Compute histograms
p = ones(1, r_) / length(r_);

option.n_bins = round(10*log(j_));
[Theta_hist_, Theta_x_] = HistogramFP(theta_mu_view_np, p, option); % Non-parametric estimator
[Theta_hist_pos, Theta_x_pos] = HistogramFP(theta_mu_view_iter, p, option); % Iterative estimator

option.n_bins = round(15*log(j_));
[L_Theta_hist_,L_Theta_x_] = HistogramFP(Loss_np, p, option);
[L_Theta_hist_pos,L_Theta_x_pos] = HistogramFP(Loss_iter, p, option);

% histograms of estimators
x_min = min([quantile(theta_mu_view_np,0.0001) quantile(theta_mu_view_iter,0.0001)]);
x_max = max([quantile(theta_mu_view_np,0.9999) quantile(theta_mu_view_iter,0.9999)]);

g = figure(); set(gcf,'color','w');
set(g,'position',[20 20 800 300])
set(g,'color','w','units','normalized','outerposition',[0.15 0.25 0.75 0.55]);
% Iterative 
subplot(2,1,1);
hm=bar(Theta_x_pos,Theta_hist_pos);
set(hm, 'FaceColor', colhist, 'EdgeColor',  'none');
line([e_theta_iter, e_theta_iter],[0,0],'color',orange,'Marker','o','MarkerSize', 6,'MarkerFaceColor',orange)
line([theta_mu_view_true,e_theta_iter],[0,0],'color',orange,'LineWidth',6)
line([e_theta_iter-inef_iter,e_theta_iter+inef_iter],[max(Theta_hist_pos)*0.02,max(Theta_hist_pos)*0.02],'color',blue,'LineWidth',4)
line([theta_mu_view_true,theta_mu_view_true],[0,0],'color',green,'Marker','o','MarkerSize', 6,'MarkerFaceColor',green)
set(gca,'Xlim',[x_min x_max]);
xlabel('Iterative semi-parametric', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
box off
title('ESTIMATORS DISTRIBUTION', 'FontSize', 24);
% Non-parametric
subplot(2,1,2);
hm=bar(Theta_x_,Theta_hist_);
set(hm, 'FaceColor', colhist, 'EdgeColor',  'none');
line([e_theta_np, e_theta_np],[0,0],'color',orange,'Marker','o','MarkerSize', 6,'MarkerFaceColor',orange)
BIAS = line([theta_mu_view_true,e_theta_np],[0,0],'color',orange,'LineWidth',6);
INEF = line([e_theta_np-inef_np,e_theta_np+inef_np],[max(Theta_hist_)*0.02,max(Theta_hist_)*0.02],'color',blue,'LineWidth',4);
TRUE = line([theta_mu_view_true,theta_mu_view_true],[0,0],'color',green,'Marker','o','MarkerSize', 6,'MarkerFaceColor',green)
set(gca,'Xlim',[x_min x_max]);
xlabel('Non-parametric', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
box off
l=legend([BIAS,INEF, TRUE],'bias' ,'ineff.', 'True');
set(l,'position',[0.87 0.78 0.1 0.07],'box','off', 'FontSize', 24);

%% Losses distributions
% histograms of square losses
x_min = min([-max(Loss_np)*0.0005 -max(Loss_iter)*0.0005]);
x_max = max([quantile(Loss_np,0.70) quantile(Loss_iter,0.70)]);

h = figure(); set(gcf,'color','w');
set(h,'position',[20 20 800 300])
set(h,'color','w','units','normalized','outerposition',[0.2 0.2 0.75 0.55]);
% Iterative 
subplot(2,1,1);
hLm = bar(L_Theta_x_pos,L_Theta_hist_pos);
set(hLm, 'FaceColor', colhist, 'EdgeColor',  'none');
line([0,bias_iter^2],[0.002,0.002],'color',orange,'LineWidth',5)
line([bias_iter^2,er_iter],[0.002,0.002],'color',blue,'LineWidth',5)
line([0,er_iter],[max(L_Theta_hist_pos)*0.0275,max(L_Theta_hist_pos)*0.0275],'color',dark,'LineWidth',5)
line([0,0],[0,0],'color',green,'Marker','o','MarkerSize',6,'MarkerFaceColor',green)
set(gca,'Xlim',[x_min x_max]);
set(gca,'Xlim',[-max(Loss_iter)*0.005 quantile(Loss_iter,0.999)]);
xlabel('Iterative semi-parametric', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
box off
title('LOSS DISTRIBUTION', 'FontSize', 24);
% Non-parametric
subplot(2,1,2);
hLm = bar(L_Theta_x_,L_Theta_hist_);
set(hLm, 'FaceColor', colhist, 'EdgeColor',  'none');
BIAS = line([0,bias_np^2],[0.002,0.002],'color',orange,'LineWidth',5);
INEF = line([bias_np^2,er_np],[0.002,0.002],'color',blue,'LineWidth',5);
ERROR = line([0,er_np],[max(L_Theta_hist_)*0.0275,max(L_Theta_hist_)*0.0275],'color',dark,'LineWidth',5);
line([0,0],[0,0],'color',green,'Marker','o','MarkerSize',6,'MarkerFaceColor',green)
set(gca,'Xlim',[x_min x_max]);
set(gca,'Xlim',[-max(Loss_np)*0.005 quantile(Loss_np,0.999)]);
xlabel('Non-parametric', 'FontSize', 24);
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
box off
l = legend([ERROR, BIAS,INEF],'error', 'bias^2' ,'ineff.^2');
set(l,'position',[0.87 0.78 0.1 0.11],'box','off', 'FontSize', 24);


