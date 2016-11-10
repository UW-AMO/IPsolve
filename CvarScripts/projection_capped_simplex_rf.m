function [ x_plus ] = projection_capped_simplex_rf( x, lb, ub, h )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

f = @(lambda) (sum(max(min(x - lambda, ub), lb)) - h);

lambda_opt = fzero(f, 0);
x_plus = max(min(x - lambda_opt, ub), lb);
end

