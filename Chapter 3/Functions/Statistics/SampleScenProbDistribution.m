function X_sample = SampleScenProbDistribution(x,p,j_)
% This function generates a sample from the scenario-probability distribution
% defined by scenarios x and probabilities p
% INPUT
% x  [matrix] (n_ x t_) scenarios defining the scenario-probability distribution of the random variable X
% p  [vector] (1 x t_) probabilities corresponding to the scenarios in x 
% j_ [scalar] number of scenarios to be generated
% OUTPUT
% X  [matrix] (n_ x t_) sample from the scenario-probability distribution (x,p)
%
% For details on the exercise, see here .
%% Code
%empirical cdf
empirical_cdf=[0 cumsum(p)];

%create random matrix
rand_uniform=rand(1,j_);

% scenarios
[~,ind] = histc(rand_uniform,empirical_cdf); 
X_sample = x(:,ind);
end

