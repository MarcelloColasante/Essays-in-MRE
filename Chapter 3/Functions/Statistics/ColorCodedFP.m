function  [Colormap, FPColors] = ColorCodedFP(FP,Min_p,Max_p,GreyRange,Cmin,Cmax,ValueRange)
% This function creates a colormap and a color-specifying vector,
% associating each probability value in FP with a certain grey gradation:
% values with higher probability are associated with darker gradations of
% grey; values with lower probability are instead associated with lighter
% gradations.
%  INPUTS
%  FP                   : [vector](1 x t_) vector of flexible probabilities
%  Min_p      (optional): [scalar] lower threshold: to probabilities <=Min_p is associated the lightest grey
%  Max_p      (optional): [scalar] upper threshold: to probabilities >=Max_p is associated the darkest grey
%  GreyRange  (optional): [column vector] it defines the hues of grey in the Colormap; 
%                       - first entry: darkest gray (default: 0 = black);
%                       - last entry: lightest gray (default: 0.8);
%                       - step: colormap step (default 0.01; if more hues are needed set it to a smaller value)
%  Cmin       (optional): [scalar] value associated to the darkest grey (default 0)
%  Cmax       (optional): [scalar] value associated to the lightest grey (default 20)
%  ValueRange (optional): [vector](1 x 2) range of values associated to hues of grey in the middle (default [20 0])
%  OUTPUTS
%  Colormap             : [matrix] this is the colormap to set before plotting the scatter
%  FPColors             : [vector](t_ x 1) contains the colors to set as input argoment of the function "scatter";
%                         the values in FPColors are linearly mapped to the colors in Colormap.
%% Code

if nargin < 2 || isempty(Min_p); Min_p = quantile(FP',0.01); end
if nargin < 3 || isempty(Max_p); Max_p = quantile(FP',0.99); end
if nargin < 4 || isempty(GreyRange); GreyRange = (0:0.01:0.8)'; end
if nargin < 5 || isempty(Cmin); Cmin = 0; end
if nargin < 6 || isempty(Cmax); Cmax = 20; end
if nargin < 7 || isempty(ValueRange); ValueRange = [20 0]; end

GreyRange = GreyRange(:);
Colormap = [GreyRange GreyRange GreyRange];

%scatter colors
C = NaN(length(FP),1);
for t = 1:length(FP)
    if FP(t) >= Max_p
        C(t) = Cmin;       
    elseif FP(t) <= Min_p
         C(t) = Cmax;
    else C(t) = interp1([Min_p Max_p],ValueRange, FP(t),'linear');
    end
end   

FPColors = C(:);
end

