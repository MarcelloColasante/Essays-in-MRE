function [Posterior] = FactorEntropyPooling(Prior, Views, Z, Guess, Supplied)
% Compute Factor Entropy Pooling posterior as described in
%   Meucci, Ardia, Colasante, Keel (2013) "Portfolio Construction via Factor Entropy Pooling"
%	  last revision and code available at http://symmys.com/node/160
%   Code by the authors (version December 2013)
%
%  The following views can be defined (and combined)
%  - (In)equality on signal/noise ratio                       ==> A * (E(x) ./ Sd(X))            = (or <= or >=) b
%  - (In)equality view on expectation                         ==> E(Ax)                          = (or <= or >=) b
%  - (In)equality view on standard deviation                  ==> Sd(Ax)                         = (or <= or >=) b
%  - (In)equality view on covariance                          ==> Cov(Ax,Dx)                     = (or <= or >=) b
%  - Equality equilibrium                                     ==> E(x) - g * Cov(X)* w           = 0
% NB
%  - Enable views by settings .on = true and disable views by setting .on = false
%  - All constraints (except Views on tangent portfolio) accept both equality (EqType = 0) and
%    inequality (EqType = 1 for <= and EqType = -1 for >=);
%  - Views on covariance should be pre-processed to flag/map non-sensical constraints. Moreover, for views on
%    covariance, b must the the lower part of the covariance matrix (without diagonal part). EqType must be a
%    vector of the same size
%  - Gradient and Hessian of views currently active only for views on signal/noise ratio
%  - The following option for derivatives can be defined (and not combined)
%  - Default:               : gradient of target       
%  - Supplied.NoDeriv = true: none                 
%  - Supplied.Grad    = true: gradient of target & constraints                          
%  - Supplied.Hess    = true: Hessian of target & constraints
%  - Enable numerical derivatives by settings:
%  - Supplied.Grad.false = true: numerical gradient       
%  - Supplied.Hess.false = true: numerical Hessian

%% code
if nargin < 2 || isempty(Views); Views = []; end
if nargin < 3 || isempty(Z); Z = []; end
if nargin < 4 || isempty(Guess); Guess = Prior; end
if ~isfield(Views, 'Exp') || isempty(Views.Exp); Views.Exp.on = false; end
if ~isfield(Views, 'Sd') || isempty(Views.Sd); Views.Sd.on = false; end
if ~isfield(Views, 'Cov') || isempty(Views.Cov); Views.Cov.on = false; end
if ~isfield(Views, 'Equilibrium') || isempty(Views.Equilibrium); Views.Equilibrium.on = false; end
if ~isfield(Views, 'Signal') || isempty(Views.Signal); Views.Signal.on = false; end
if ~isfield(Views, 'Generic') || isempty(Views.Generic); Views.Generic.on = false; end
if nargin < 5 || isempty(Supplied); Supplied = []; end
if ~isfield(Supplied, 'Grad') || isempty(Supplied.Grad); Supplied.Grad.on = false; Supplied.Grad.false = false; end
if ~isfield(Supplied.Grad, 'on') || isempty(Supplied.Grad.on); Supplied.Grad.on = false; end
if ~isfield(Supplied.Grad, 'false') || isempty(Supplied.Grad.false); Supplied.Grad.false = false; end
if ~isfield(Supplied, 'Hess') || isempty(Supplied.Hess); Supplied.Hess.on = false; Supplied.Hess.false = false; end
if ~isfield(Supplied.Hess, 'on') || isempty(Supplied.Hess.on); Supplied.Hess.on = false; end
if ~isfield(Supplied.Hess, 'false') || isempty(Supplied.Hess.false); Supplied.Hess.false = false; end
if ~isfield(Supplied, 'NoDeriv') || isempty(Supplied.NoDeriv); Supplied.NoDeriv = false; end
if ~isfield(Supplied, 'matrix') || isempty(Supplied.matrix); Supplied.matrix = []; end


k_ = Prior.k_;
mu_pri = Prior.mu(:);
n_ = length(mu_pri);
sigma2_pri = Prior.sigma2;
invsigma2_pri = sigma2_pri \ eye(n_);

% starting value
mu_0 = Guess.mu;
sigma2_0 = Guess.sigma2;
theta_0 = InitialGuess(mu_0, sigma2_0, k_);

% objective (relative entropy) ...
f = @(theta) REnormLRD(theta, mu_pri, invsigma2_pri, n_, k_, Supplied);
g = [];

if all(~[Views.Exp.on, Views.Sd.on, Views.Cov.on, Views.Equilibrium.on, Views.Signal.on, Views.Generic.on])
    options = optimset('GradObj', 'on', 'Display', 'off', ...
            'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');
    if Supplied.Hess.on
        options = optimset('GradObj', 'on', 'Hessian', 'on', 'Display', 'off', ...
            'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');
    end
    
    [theta_, mveppar_, exitflag] = fminunc(f, theta_0, options);
    if exitflag < 1
    error('MATLAB:FactorEntropyPooling', 'Optimization did not converge');
    end
    
    % reshape posterior structure
    [mu_pos, sigma2_pos] = theta2param(theta_, n_, k_);

    Posterior = Prior;
    Posterior.mu = mu_pos(:);
    Posterior.sigma2 = sigma2_pos;
    Posterior.mveppar = mveppar_;
    return;
end

if Supplied.NoDeriv
    if Views.Exp.on || Views.Sd.on || Views.Cov.on || Views.Equilibrium.on || Views.Signal.on || Views.Generic.on
        g = @(theta) ViewsConstraint(theta, Views, n_, k_, Z);
    end
    % optimization without derivatives
    options = optimset('Algorithm', 'interior-point', 'Display', 'off', ...
        'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');
else

    if nargin < 5 || isempty(Supplied)
        % ... and contraints (views/stress-tests)
        if Views.Exp.on || Views.Sd.on || Views.Cov.on || Views.Equilibrium.on || Views.Signal.on || Views.Generic.on
            g = @(theta) ViewsConstraint(theta, Views, n_, k_, Z);
        end
        % optimization with analytical gradient of object function
        options = optimset('Algorithm', 'interior-point', 'GradObj', 'on', 'Display', 'off', ...
            'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');   
    else    
        % ... and contraints (views/stress-tests)
        if Views.Exp.on || Views.Sd.on || Views.Cov.on || Views.Equilibrium.on || Views.Signal.on || Views.Generic.on
            g = @(theta) ViewsConstraint(theta, Views, n_, k_, Z);
        end
        % optimization with analytical gradient of object function
        options = optimset('Algorithm', 'interior-point', 'GradObj', 'on', 'Display', 'off', ...
            'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');   

        if Views.Signal.on && ~Views.Exp.on && ~Views.Sd.on && ~Views.Cov.on && ~Views.Equilibrium.on
            if Supplied.Grad.on % compute gradient of constraints
               Supplied.Hess.on = false; %
               if nargin < 6 || isempty(Supplied.matrix)
                  Supplied.matrix = SupplyFEP(n_, k_, Supplied);
               end
               g = @(theta) ViewsConstraint(theta, Views, n_, k_, Z, Supplied);
               options = optimset('Algorithm', 'interior-point','GradObj', 'on','GradConstr','on','Display', 'off', ...
                    'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off');
            elseif Supplied.Hess.on % compute hessian of target and constraints
                if nargin < 6 || isempty(Supplied.matrix)
                   Supplied.matrix = SupplyFEP(n_, k_, Supplied);
                end
                g = @(theta) ViewsConstraint(theta, Views, n_, k_, Z, Supplied); 
                options = optimset('Algorithm', 'interior-point','GradObj', 'on','GradConstr','on','Display', 'off', ...
                    'MaxFunEvals', 1000000, 'MaxIter', 1000000, 'DerivativeCheck', 'off', ...
                    'Hessian','user-Supplied','HessFcn',@(theta, lambda) HessLagrange(theta, lambda, mu_pri, invsigma2_pri, Views, n_, k_, Supplied));
            end
        end

    end
end

[theta_, mveppar_, exitflag] = fmincon(f, theta_0, [], [], [], [], [], [], g, options);
if exitflag < 1
    error('MATLAB:FactorEntropyPooling', 'Optimization did not converge');
end

% reshape posterior structure
[mu_pos, sigma2_pos] = theta2param(theta_, n_, k_);

Posterior = Prior;
Posterior.mu = mu_pos(:);
Posterior.sigma2 = sigma2_pos;
Posterior.mveppar = mveppar_;

end

function [theta_0] = InitialGuess(mu, sigma2, k_)
% Initial guess for parameter

%% code
try
    warning('off'); %#ok<WNOFF>
    [b_0, d2_0] = factoran(sigma2, k_, 'Xtype', 'covariance', 'rotate', 'none');
    warning('on'); %#ok<WNON>
    dsig = diag(sqrt(diag(sigma2)));
    b_0 = dsig * b_0;
    d2_0 = dsig.^2 * d2_0;
    theta_0 = [mu; reshape(b_0, [], 1); sqrt(d2_0)];
catch %#ok<CTCH>
    n_ = length(mu);
    theta_0 = rand(n_ * (k_ + 2), 1);
end

end


function [c, ceq,gradc,gradceq] = ViewsConstraint(theta, Views, n_, k_, Z, Supplied)
% Constraints = views/stress-tests
% Compute the constraint function in the following cases:
%
% (In)equality view on expectation
% (In)equality view on standard deviation
% (In)equality view on covariance
%     equality view on tangent portfolio
% (In)equality view on signal
% (In)equality generic view
%
% Compute the analytical gradient in the following cases:
%
% (In)equality view on signal

%% code
[mu, sigma2, b, d] = theta2param(theta, n_, k_);

c   = [];
ceq = [];
gradc = [];
gradceq = [];

% Views on expectation
if Views.Exp.on
    Exp_ = Views.Exp.a * mu;
    tmp = Exp_ - Views.Exp.b;
    idx = Views.Exp.EqType == 0;
    % == case
    ceq = [ceq; tmp(idx)];
    % <= case
    idx = Views.Exp.EqType == 1;
    c = [c; tmp(idx)];
    % >= case
    idx = Views.Exp.EqType == -1;
    c = [c; -tmp(idx)];
end

% Views on standard deviation
if Views.Sd.on
    % == case
    sigma2_i = Views.Sd.a * sigma2 * Views.Sd.a';
    tmp = sqrt(diag(sigma2_i)) - Views.Sd.b(idx);
    % == case
    idx = Views.Sd.EqType == 0;
    ceq = [ceq; tmp(idx)];    
    % <= case
    idx = Views.Sd.EqType == 1;
    c = [c; tmp(idx)];
    % >= case
    idx = Views.Sd.EqType == -1;
    c = [c; -tmp(idx)];
end

% Views on covariance
if Views.Cov.on
    % == case
    sigma2_i = Views.Cov.a * sigma2 * Views.Cov.a';
    n = size(sigma2_i, 2);
    ids = tril(reshape(1:n^2, n, n));
    ids = ids(ids > 0);
    tmp = sigma2_i(ids) - Views.Cov.b;
    % == case
    idx = Views.Cov.EqType == 0;
    ceq = [ceq; tmp(idx)];
    % <= case
    idx = Views.Cov.EqType == 1;
    c = [c; tmp(idx)];
    % >= case
    idx = Views.Cov.EqType == -1;
    c = [c; -tmp(idx)];
end

% Views on a tangent portfolio
if Views.Equilibrium.on
    tmp = mu - Views.Equilibrium.g * sigma2 * Views.Equilibrium.w;
    ceq = [ceq; tmp];
end

% Views on signal
if Views.Signal.on
    sig = sqrt(diag(sigma2));
    signal = Views.Signal.a * (mu ./ sig);
    tmp = signal - Views.Signal.b;    
    % == case
    idx = Views.Signal.EqType == 0;
    ceq = [ceq; tmp(idx)];
    % <= case
    idx1 = Views.Signal.EqType == 1;
    c = [c; tmp(idx1)];    
    % >= case
    idx2 = Views.Signal.EqType == -1;
    c = [c; -tmp(idx2)];
    
    % compute gradient
    if nargout > 2
        aeq = Views.Signal.a(idx, :);
        aineq = [Views.Signal.a(idx1, :); - Views.Signal.a(idx2, :)];
        % numerical gradient
        if Supplied.Grad.false
            funeq = @(theta_) SigNoConstr(theta_, aeq, n_, k_);
            ngradeq = NumJac(funeq, theta);
            ngradeq = ngradeq';
            gradceq = [gradceq; ngradeq];

            fun = @(theta_) SigNoConstr(theta_, aineq, n_, k_);
            ngrad = NumJac(fun, theta);
            ngrad = ngrad';
            gradc = [gradc; ngrad];
            
        else
            % analytical gradient
            e = sparse(1 : n_, 1 : n_, 1);
            alpha = mu ./ diag(sigma2) .^ (3/2);

            const1 = sqrt(diag(diag(1 ./ sigma2)));
            const2 = kronecker(b', e) * Supplied.matrix.hm;
            const3 = diag(alpha .* d);

            % == case
            gradeq_mu = const1 * aeq';
            gradeq_b = - const2 * kronecker(aeq', alpha);
            gradeq_d = - const3 * aeq';

            gradceq = [gradceq; gradeq_mu; gradeq_b; gradeq_d];

            % <= and >= case
            grad_mu = const1 * aineq';
            grad_b = - const2 *kronecker(aineq', alpha);
            grad_d = - const3 * aineq';

            gradc = [gradc; grad_mu; grad_b; grad_d];
        end
    end
end

% Generic views
if Views.Generic.on
    % affine transformation in the normal case
    X = Z * chol(sigma2, 'lower')';
    X = bsxfun(@plus, X, mu(:)');
    [check, EqType] = Views.Generic.Function(X);
    ceq = [ceq; check(EqType == 0)];
    c = [c; check(EqType == 1)];
    c = [c; -check(EqType == -1)];
end

end

function [constr] = SigNoConstr(theta, a, n_, k_)
% Constraint function on signal/noise ratio

%% code
[mu, sigma2] = theta2param(theta, n_, k_);
constr = a * (mu ./ sqrt(diag(sigma2)));

end

function [hesslag] = HessLagrange(theta, lambda, mu_pri, invsigma2_pri, Views, n_, k_, Supplied)
% Hessian of Lagrangian

%% code
if Supplied.Hess.false
    % compute numerical Hessian of entropy
    target = @(theta_) REnormLRD(theta_, mu_pri, invsigma2_pri, n_, k_);
    grad2 = NumHess(target, theta); 
    
    % compute numerical Hessian of Lagrangian
    idx = Views.Signal.EqType == 0;
    idx1 = Views.Signal.EqType == 1;
    idx2 = Views.Signal.EqType == -1;

    aeq = Views.Signal.a(idx, :);
    aineq = [Views.Signal.a(idx1, :); - Views.Signal.a(idx2, :)];
    meq_ = size(aeq, 1);
    mineq_= size(aineq, 1);
    
    hesslag = grad2;

    for m = 1 : max(mineq_, meq_)
        if m <= mineq_
           fun = @(theta_) SigNoConstr(theta_, aineq(m, :), n_, k_);
           hessc = NumHess(fun, theta);
           hesslag = hesslag + lambda.ineqnonlin(m) * hessc;
        end
        if m <= meq_
           funeq = @(theta_) SigNoConstr(theta_, aeq(m, :), n_, k_);
           hessceq = NumHess(funeq, theta);
           hesslag = hesslag + lambda.eqnonlin(m) * hessceq;
        end        
    end
        
else
    
    [mu, sigma2, b, d] = theta2param(theta, n_, k_);
    % compute inverse of sigma2 using binomial inverse theorem
    delta2 = d.^2;
    diag_  = diag(1 ./ delta2);
    tmp    = (b' * diag_ * b + eye(k_)) \ (b' * diag_);
    invsigma2 = diag_ - (diag_ * b * tmp);

    % compute analytical Hessian of entropy
    e = sparse(1 : n_, 1 : n_, 1);
    a = sparse(1 : k_, 1 : k_, 1); 
    v = (invsigma2_pri - invsigma2);
    diagd = diag(d);
    grad2_mumu = invsigma2_pri;
    grad2_dd = (2 * diagd * invsigma2) .* (invsigma2 * diagd) + diag(diag(v));
    grad2_bd = kronecker(b' * invsigma2, invsigma2 * diagd) * 2 * Supplied.matrix.hm1;
    grad2_bb = kronecker(b' * invsigma2* b, invsigma2) + Supplied.matrix.km * kronecker(invsigma2 * b, b' * invsigma2) + kronecker(a, v);
    grad2 = [grad2_mumu          zeros(n_, n_ * k_)   zeros(n_, n_)
             zeros(n_ * k_, n_)  grad2_bb             grad2_bd
             zeros(n_, n_)      (grad2_bd)'           grad2_dd];

    % compute analytical Hessian of constraints
    idx  = Views.Signal.EqType == 0;
    idx1 = Views.Signal.EqType == 1;
    idx2 = Views.Signal.EqType == -1;

    aeq = Views.Signal.a(idx, :);
    aineq = [Views.Signal.a(idx1, :); - Views.Signal.a(idx2, :)];
    [meq_, ~] = size(aeq);
    [mineq_, ~] = size(aineq);

    const1  = diag(diag(1 ./ sigma2)).^(3/2);
    const2  = kronecker(b', e);
    const3  = const2 * Supplied.matrix.hm;
    const4  = kronecker(aeq', const1);
    const5  = kronecker(aineq', const1);
    const6  = diag(sigma2).^(3/2);
    const7  = diag(sigma2).^(5/2);
    const8  = - kronecker(d', e) * Supplied.matrix.hm;
    const9  = 3 * const3 * kronecker(b, diag(mu ./ const7))- kronecker(a, diag(mu ./ const6));
    const10 = diag((mu .* d) ./ const7);
    const11 = diag(mu .* ((3 * d.^2) ./ const7 - 1 ./ const6));

    % == case
    grad2eq_mumu = zeros(n_, n_);
    grad2eq_bmu  = - const3 * const4;
    grad2eq_dmu  = const8 * const4;
    grad2eq_bb   = Supplied.matrix.km1 * kronecker(aeq', const9);
    grad2eq_bd   = 3 * const3 * kronecker(aeq', const10);
    grad2eq_dd   = Supplied.matrix.hm2 * kronecker(aeq', const11);
    % <= and >= case
    grad2_mumu = zeros(n_, n_);
    grad2_bmu  = - const3 * const5;
    grad2_dmu  = const8 * const5;
    grad2_bb   = Supplied.matrix.km1 * kronecker(aineq', const9);
    grad2_bd   = 3 * const3 * kronecker(aineq', const10);
    grad2_dd   = Supplied.matrix.hm2 * kronecker(aineq', const11);

    % compute Hessian of Lagrangian
    hesslag = grad2;

    for m = 1 : max(mineq_, meq_)

        if m <= mineq_
            hessc = [grad2_mumu                     (grad2_bmu(:,(m-1)*n_+1:m*n_))'     (grad2_dmu(:,(m-1)*n_+1:m*n_))'
                     grad2_bmu(:,(m-1)*n_+1:m* n_)  grad2_bb(:,(m-1)*n_*k_+1:m*n_*k_)   grad2_bd(:,(m-1)*n_+1:m*n_)
                     grad2_dmu(:,(m-1)*n_+1:m*n_)   (grad2_bd(:,(m-1)*n_+1:m*n_))'      grad2_dd(:,(m-1)*n_+1:m*n_)];
            hesslag = hesslag + lambda.ineqnonlin(m) * hessc;

        end
        if m <= meq_
            hessceq = [grad2eq_mumu                    (grad2eq_bmu(:,(m-1)*n_+1:m*n_))'    (grad2eq_dmu(:,(m-1)*n_+1:m*n_))'
                       grad2eq_bmu(:,(m-1)*n_+1:m*n_)  grad2eq_bb(:,(m-1)*n_*k_+1:m*n_*k_)  grad2eq_bd(:,(m-1)*n_+1:m*n_)
                       grad2eq_dmu(:,(m-1)*n_+1:m*n_)  (grad2eq_bd(:,(m-1)*n_+1:m*n_))'     grad2eq_dd(:,(m-1)*n_+1:m*n_)];
            hesslag = hesslag + lambda.eqnonlin(m) * hessceq;

        end


    end
end

end

function [mu, sigma2, b, d] = theta2param(theta, n_, k_)
% Reparametrization from theta matrix

%% code
id = 1 : n_;
mu = reshape(theta(id), [], 1);
id = (n_+1) : n_ + (n_*k_);
b = reshape(theta(id), n_, k_);
id = n_ + (n_*k_) + 1 : n_ * (2+k_);
d = reshape(theta(id), [], 1);

sigma2 = b * b' + diag(d.^2);

end

