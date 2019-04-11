% This script performs the parallel backtest a systematic trading strategy, 
% processing ranking momentum signals in the Dow Jones equity market via MRE 
% approach and % standard approach by Grinold and Khan (1999), FEP approach 
% by Meucci, A., Ardia, D., and Keel, S. (2011), using different values of
% the decay parameter for the historical estimation

clear; clc; close all;

run InitLibrary.m

%% Input parameters
hl = 52 : 26 : 208; % half-life for rolling values

%% Initialize list
inList = {};
for i1 = 1 : length(hl)
        in = [];
        in.hl = hl(i1);
        inList{end+1} = in; %#ok<SAGROW>
end

disp(length(inList));

MaxCore = 4;
outList = cell(length(inList), 1);
if MaxCore == 1
    for i = 1 : length(inList)
        outList{i} = PerformBacktest(inList{i});
    end
end
if MaxCore > 1;
    try matlabpool(MaxCore); catch; end; %#ok<CTCH>
    parfor i = 1 : length(inList)
        outList{i} = PerformBacktest(inList{i});
    end
    try matlabpool close; catch; end;  %#ok<CTCH>
end

save ParallelBacktest.mat outList;

tmp = [outList{:}];
hl = unique([tmp.hl]);

%% Setup time span
start_date = datenum('01.01.2006', 'dd.mm.yyyy');
time = outList{1}.EW.time;
pos  = find(time >= start_date);
time = time(pos);

%% Compute PnL's, Sharpe ratios and p-values
n = length(outList);
cpnl_CO_ = NaN(length(time), n);
cpnl_FEP_ = NaN(length(time), n);
cpnl_MRE_ = NaN(length(time), n);
label = {};

for i = 1 : n    
    pnl_EW      = outList{i}.EW.pnl(pos);
    pnl_NS      = outList{i}.NS.pnl(pos);
    pnl_Com     = outList{i}.CO.pnl(pos);
    pnl_FEP     = outList{i}.FEP.pnl(pos);
    pnl_MRE      = outList{i}.MRE.pnl(pos);
    cpnl_CO_(:, i) = cumsum(pnl_Com);
    cpnl_FEP_(:, i) = cumsum(pnl_FEP);
    cpnl_MRE_(:, i) = cumsum(pnl_MRE);
    
    pnl = [pnl_EW(2:end), pnl_MRE(2:end), pnl_Com(2:end), pnl_FEP(2:end), pnl_NS(2:end)];
    m_pnl = mean(pnl);
    s2_pnl = cov(pnl);
    sr_pnl(i, :) = m_pnl ./ sqrt(diag(s2_pnl))';

    t_ = length(pnl_EW(2:end));
    Index = [1, 2];
    p_MRE(i) = PvalueJK(m_pnl(Index), s2_pnl(Index, Index), t_);
    Index = [1, 3];
    p_CO(i) = PvalueJK(m_pnl(Index), s2_pnl(Index, Index), t_);
    Index = [1, 4];
    p_FEP(i) = PvalueJK(m_pnl(Index), s2_pnl(Index, Index), t_);
    Index = [1, 5];
    p_NS(i) = PvalueJK(m_pnl(Index), s2_pnl(Index, Index), t_);
    
end

%% Filter data
idx = true(n, 1);
cpnl_CO_ = cpnl_CO_(:, idx);
cpnl_FEP_ = cpnl_FEP_(:, idx);
cpnl_MRE_ = cpnl_MRE_(:, idx);

%% Figures

%Fan plots
figure(); hold on;
subplot(1, 2, 1);
ax1 = PlotFan(time, cpnl_CO_, 'b');
ax2 = PlotFan(time, cpnl_MRE_, 'r');
grid on; datetick('x', 12); axis tight; 
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
%title('Fan plot'); 
%legend([ax1 ax2], {'Common', 'MRE'}, 'Location','northwest');

subplot(1, 2, 2);
hold on;
ax1 = PlotFan(time, cpnl_FEP_, 'm');
ax2 = PlotFan(time, cpnl_MRE_, 'r');
grid on; datetick('x', 12); axis tight; 
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
title('Fan plot'); 
legend([ax1 ax2], {'FEP', 'MRE'}, 'Location','northwest');

% Sharpe ratios
figure(); 
subplot(2, 1, 1); hold on;
plot(hl, sr_pnl(:, 1)', 'go-', 'linewidth', 3, 'markersize', 2);
plot(hl, sr_pnl(:, 2)', 'ro-', 'linewidth', 3, 'markersize', 2);
plot(hl, sr_pnl(:, 3)', 'bx-', 'linewidth', 3, 'markersize', 2);
plot(hl, sr_pnl(:, 4)', 'mx-', 'linewidth', 3, 'markersize', 2);
plot(hl, sr_pnl(:, 5)', 'kx-', 'linewidth', 3, 'markersize', 2);
grid on; axis tight;
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
title('Sharpe ratios'); 
legend('Equally weighted', 'MRE', 'Common', 'FEP', 'No views', 'Location', 'NorthWest');

% P-values
subplot(2, 1, 2); hold on;
plot(hl, p_MRE, 'ro-', 'linewidth', 3, 'markersize', 2);
plot(hl, p_CO, 'bx-', 'linewidth', 3, 'markersize', 2);
plot(hl, p_FEP, 'mx-', 'linewidth', 3, 'markersize', 2);
plot(hl, p_NS, 'kx-', 'linewidth', 3, 'markersize', 2);
grid on; axis tight; 
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
title('P-value');




