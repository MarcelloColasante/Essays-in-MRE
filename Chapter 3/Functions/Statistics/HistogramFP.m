function [f,xi] = HistogramFP(Epsi,p,option)
% This function computes the histogram of the data series contained in
% Epsi, according to the Flexible Probabilities p associated with each
% observation.
%  INPUTS
%  Epsi    :[vector](1 x t_) for the 2d-hist or (2 x t_) for the 3d-hist; data series
%  p       :[vector](1 x t_) flexible probabilities associated to the data series Epsi
%  option  :[struct] input specifying the bins' properties; fields:
%            - .n_bins : number of bins or alternatively the centers of bins
%            - .tau : horizon
%              .k_ : number of bins
%  OUTPUTS
%  f   :[row vector] vector containing the heights of each bin
%  xi  :[row vector] vector containing the center of each bin

%% Code
[n_,~] = size(Epsi);

names = fieldnames(option);
switch names{1}
    case 'n_bins'
    n_bins = option.n_bins;    
    switch n_
        case 1 % 'n_bins', n_=1
            
        if length(n_bins) == 1% compute bins' centers
            [~,xi] = hist(Epsi,n_bins);
        elseif length(n_bins) > 1
            xi = n_bins;
        end
        h = xi(2)-xi(1);
        f = nan(1,length(xi));
        for k = 1:length(xi)% compute histogram
            Index = (Epsi >= xi(k)-h/2) & (Epsi < xi(k)+h/2);
            f(k) = sum(p(Index));
        end
        f = f/h;
        
        case 2 % 'n_bins', n_=2
        
        if isequal(size(n_bins),[2 1]) || isequal(size(n_bins),[1 2])% compute bins' centers
            [~,xi] = hist3(Epsi',n_bins);
        else
            xi{1} = n_bins(1,:);
            xi{2} = n_bins(2,:);
        end
        h1 = xi{1}(2)-xi{1}(1);
        h2 = xi{2}(2)-xi{2}(1);
        f = nan(length(xi{1}),length(xi{2}));% compute histogram
        for k1 = 1:length(xi{1})
            for k2 = 1:length(xi{2})
                Index = (Epsi(1,:)>=xi{1}(k1)-h1/2) & (Epsi(1,:)<xi{1}(k1)+h1/2) & (Epsi(2,:)>=xi{2}(k2)-h2/2) & (Epsi(2,:)<xi{2}(k2)+h2/2);
                f(k1,k2) = sum(p(Index));
            end
        end
        f = f/(h1*h2);
        
    end
    case 'tau'
    tau = option.tau;
    k_ = option.k_;
    a = -norminv(10^(-15), 0, sqrt(tau));% compute bins' centers
    h = 2*a/k_;
    xi = (-a+h : h : a);
    f = nan(1,length(xi));
    for k = 1:length(xi)% compute histogram
        Index = (Epsi >= xi(k)-h/2) & (Epsi < xi(k)+h/2);
        f(k) = sum(p(Index));
    end
    f(k_) = max(1-sum(f(1:end-1)),0);
    f = f/h;
    
end

