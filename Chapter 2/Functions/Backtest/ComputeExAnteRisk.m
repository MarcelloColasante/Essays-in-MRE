function [s2] = ComputeExAnteRisk(h, sigma2p)

s2 = h' * sigma2p * h;

end