function [out] = PerformBacktest(in)
% This function perferm the backtest in parallel for a range of values of
% the decay parameter for the historical estimation

%% Input parameters
n_ = 30; % market dimension

lambda_st = log(2)/8; % momentum signal short-term half-life
lambda_lt = log(2)/52; % momentum signal long-term half-life
lambda = log(2) / in.hl;
volTarget = 1e4; % volatility target in $
wbound = .5e6; % constraint on concentration
pttc = 0.0005; % transaction costs, as fraction of market value

load db_DJIA.mat;

start_date = datenum('01.01.2006', 'dd.mm.yyyy'); % backtest dates
idx = time >= start_date;
bt_time = time(idx);
nBacktest = length(bt_time) - 1;

%% Initialize strategies
run InitLibrary.m
EW = InitStrategy(bt_time, n_);
NS = InitStrategy(bt_time, n_);
CO = InitStrategy(bt_time, n_);
MRE = InitStrategy(bt_time, n_);
FEP = InitStrategy(bt_time, n_);

%%% Set user supplied derivatives for FEP
k_ = 1;
Supplied = [];
Supplied.Hess.on = true; 
Supplied.matrix = SupplyFEP(n_, k_, Supplied); 

for t = 1 : nBacktest
    disp(['Iteration ', num2str(t)]);
    
    % In-Sample prices and returns (expanding window)
    pos_t = find(time == bt_time(t), 1, 'first');
    price_its = price(1:pos_t, 1:n_);
    price_t = price_its(end, :)';
    rets_its = diff(price_its) ./ price_its(1:end-1, :); % linear returns
    
    % Out-of-sample prices PnL
    pos_nxt = find(time == bt_time(t+1), 1, 'first');
    price_nxt = price(pos_nxt, 1:n_)';
    pnl_price = price_nxt - price_t;              
        
    % Mean and covariance matrix estimation (returns)
    [mu_t, sigma2_t] = EwmaFP(rets_its', lambda);   
    
    % Mean and covariance matrix estimation (prices)
    mup_t = price_t .* mu_t;
    sigma2p_t = diag(price_t) * sigma2_t * diag(price_t);
    sigma2p_t = (sigma2p_t + sigma2p_t')/2; % ensure matrix is symmetric
    sigp_t = sqrt(diag(sigma2p_t));
            
    % ==> Equally-weighted portfolio
    EW_ptf = ConstantRiskPortfolioEqualWeight(sigma2p_t, volTarget, EW.h_t, price_t, pttc, false);
    EW.h_t = EW_ptf.h;
    EW.h(:, t) = EW.h_t;
    EW.pnl(t+1) = ComputePnL(EW.h_t, pnl_price, EW_ptf.tc);
    EW.s2(t+1) = ComputeExAnteRisk(EW.h_t, sigma2p_t);
    EW.cpu(t) = EW_ptf.cpu; 
    
    % ==> No signal approach      
    NS_ptf = ConstantRiskPortfolio(mup_t, sigma2p_t, volTarget, wbound, NS.h_t, EW.h_t, price_t, pttc);
    NS.h_t = NS_ptf.h;
    NS.h(:, t) = NS.h_t;
    NS.pnl(t+1) = ComputePnL(NS.h_t, pnl_price, NS_ptf.tc);
    NS.s2(t+1) = ComputeExAnteRisk(NS.h_t, sigma2p_t);
    NS.check{t} = NS_ptf.check;
    NS.cpu(t) = NS_ptf.cpu;
            
    % Signal used in Common and MRE approaches  
    score_t = -EwmaFP(rets_its', lambda_st) ./ sqrt(EwmaFP(rets_its'.^2, lambda_lt));
        
    % Signal (which is a ranking)
    [~, rk] = sort(score_t, 'ascend');
    [~, rk_Signal_t] = sort(rk, 'ascend');
    CO.rk(:, t) = rk_Signal_t;
    MRE.rk(:, t) = rk_Signal_t;
    
    % ==> Common approach 
    % Convert signal into expected returns
    mup_Com_t = sigp_t .* (rk_Signal_t - 0.5 * (n_+1)) * (2 / (n_-1)); % factor ensure max sharpe is 1
    CO.mu(:, t) = mup_Com_t;
    CO.sig(:, t) = sigp_t;        
    
    % Constant $ vol allocation
    CO_ptf = ConstantRiskPortfolio(mup_Com_t, sigma2p_t, volTarget, wbound, CO.h_t, EW.h_t, price_t, pttc);
    CO.h_t = CO_ptf.h;
    CO.h(:, t) = CO.h_t;
    CO.pnl(t+1) = ComputePnL(CO.h_t, pnl_price, CO_ptf.tc);
    CO.s2(t+1) = ComputeExAnteRisk(CO.h_t, sigma2p_t);
    CO.check{t} = CO_ptf.check;
    CO.cpu(t) = CO_ptf.cpu;
      
    % ==> FEP approach 
    % Base (in prices space)
    Base = [];
    Base.mu = mup_t;
    Base.sigma2 = sigma2p_t; 
    Base.k_ = k_;
    Guess = Base;
        
    % Views 
    Views = [];
    Views.Signal.on = true;
    Views.Signal.a = zeros(n_-1, n_);
    for i = 1 : (n_-1)
        idx1 = rk_Signal_t == i;
        idx2 = rk_Signal_t == i + 1;
        Views.Signal.a(i,idx1) = 1;
        Views.Signal.a(i,idx2) = -1;
    end
    Views.Signal.b = -(2/(n_-1)) * ones(n_-1, 1);
    Views.Signal.EqType = ones(n_-1, 1); % <= inequality
    % Summability
    pos1 = find(rk_Signal_t == 1);
    pos2 = find(rk_Signal_t == n_);
    Views.Signal.a(n_, 1:n_) = 0;
    Views.Signal.a(n_, pos1) = 1;
    Views.Signal.a(n_, pos2) = 1;
    Views.Signal.b(n_) = 0;
    Views.Signal.EqType(n_) = 0;
    % Rescaling   
    Views.Signal.a(n_+1, 1:n_) = 0;
    Views.Signal.a(n_+1, pos2) = 1; 
    Views.Signal.b(n_+1) = 1;
    Views.Signal.EqType(n_+1) = 0;
    tic;
    
    Update_FEP = FactorEntropyPooling(Base, Views, [], Guess, Supplied);
    
    FEP.mu(:, t) = Update_FEP.mu;
    FEP.sig(:, t) = sqrt(diag(Update_FEP.sigma2));
    FEP.cpu2(t) = toc;
     
    % Constant $ vol allocation
    FEP_ptf = ConstantRiskPortfolio(Update_FEP.mu, Update_FEP.sigma2, volTarget, wbound, FEP.h_t, EW.h_t, price_t, pttc);    
    FEP.h_t = FEP_ptf.h;
    FEP.h(:, t) = FEP.h_t;
    FEP.pnl(t+1) = ComputePnL(FEP.h_t, pnl_price, FEP_ptf.tc);
    FEP.s2(t+1) = ComputeExAnteRisk(FEP.h_t, Update_FEP.sigma2); 
    FEP.check{t} = FEP_ptf.check;
    FEP.cpu(t) = FEP_ptf.cpu;
    
    % ==> MRE approach 
    % Update base covariance
    sig_vol_view = sqrt(diag(Update_FEP.sigma2));
    [~, c2] = cov2corr(Base.sigma2);
    Base.sigma2 = diag(sig_vol_view) * c2 * diag(sig_vol_view); % update volatilities

             
    % Ranking of Sharpe ratios
    z_mu_ineq = zeros(n_-1, n_);
    for i = 1 : (n_-1)
        idx1 = rk_Signal_t == i;
        idx2 = rk_Signal_t == i + 1;
        z_mu_ineq(i,idx1) = 1/sig_vol_view(idx1);
        z_mu_ineq(i,idx2) = -1/sig_vol_view(idx2);
    end
    
    eta_mu_ineq = -(2/(n_-1)) * ones(n_-1, 1); 
    
    % Summability and rescaling
    z_mu_eq = zeros(2, n_);
    pos1 = find(rk_Signal_t == 1);
    pos2 = find(rk_Signal_t == n_);
    z_mu_eq(1, pos1) = 1/sig_vol_view(pos1);
    z_mu_eq(1, pos2) = 1/sig_vol_view(pos2);
    z_mu_eq(2, pos2) = 1/sig_vol_view(pos2); 
    
    eta_mu_eq = [0; 1];
   
    eta_mu_view = [eta_mu_ineq; eta_mu_eq];
    tic; 
    
    [~, Update_MRE.mu, Update_MRE.sigma2] = FitMREIneq1MomentsN(z_mu_ineq, z_mu_eq, eta_mu_view, Base.mu, Base.sigma2);   

    MRE.mu(:, t) = Update_MRE.mu;
    MRE.sig(:, t) = sqrt(diag(Update_MRE.sigma2));
    MRE.cpu2(t) = toc;
    
    % Constant $ vol allocation
    MRE_ptf = ConstantRiskPortfolio(Update_MRE.mu, Update_MRE.sigma2, volTarget, wbound, MRE.h_t, EW.h_t, price_t, pttc);    
    MRE.h_t = MRE_ptf.h;
    MRE.h(:, t) = MRE.h_t;
    MRE.pnl(t+1) = ComputePnL(MRE.h_t, pnl_price, MRE_ptf.tc);
    MRE.s2(t+1) = ComputeExAnteRisk(MRE.h_t, Update_MRE.sigma2); 
    MRE.check{t} = MRE_ptf.check;
    MRE.cpu(t) = MRE_ptf.cpu;
end

out = [];
out.hl = in.hl;
out.EW = EW;
out.NS = NS;
out.CO = CO;
out.FEP = FEP;
out.MRE = MRE;
end