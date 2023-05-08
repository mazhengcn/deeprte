function [boundary, rand_params] = gen_test_boundary(params, rand_params)

% [var_x_start, var_x_range] = get_rand_params(params.var_x_scope);
% [var_v_start, var_v_range] = get_rand_params(params.var_v_scope);

% rand_params.variance_x = var_x_range * rand([1, 4]) + var_x_start;
% rand_params.variance_v = var_v_range * rand([1, 4]) + var_v_start;

% rand_params.r_ind = randi(params.r_ind_scope, 1, 4);
% rand_params.v_ind = randi(params.v_ind_scope, 1, 4);

yl = params.yl; xl = params.xl; yr = params.yr; xr = params.xr;
hy = params.hy; hx = params.hx;

M = params.M; J = params.J; I = params.I;

ct = params.ct;
st = params.st;

func_psiL = @(x, y)(1 + 0 .* y);
func_psiL_v = @(x)(1 + 0 .* x);
% func_psiL_v = @(x)(exp(- (x - x(v_index(1))).^2/2 / variance_v(1)));
func_psiR = @(x, y)(0 .* y);
func_psiR_v = @(x)(1 + 0 .* x);
func_psiB = @(x, y)(0 .* x);
func_psiB_v = @(x)(1 + 0 .* x);
func_psiT = @(x, y)(0 .* x);
func_psiT_v = @(x)(1 + 0 .* x);

boundary.psiL = zeros(2*M, J+1); boundary.psiR = zeros(2*M, J+1); boundary.psiB = zeros(2*M, I+1); boundary.psiT = zeros(2*M, I+1);

boundary.psiL(:, :) = func_psiL_v([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_psiL_v([st(3 * M + 1:4 * M); st(1:M)]) * func_psiL(xl, yl:hy:yr);
boundary.psiR(:, :) = func_psiR_v(ct(1 * M + 1:3 * M)) .* func_psiR_v(st(1 * M + 1:3 * M)) * func_psiR(xr, yl:hy:yr);
boundary.psiB(:, :) = func_psiB_v(ct(0 * M + 1:2 * M)) .* func_psiB_v(st(0 * M + 1:2 * M)) * func_psiB(xl:hx:xr, yl);
boundary.psiT(:, :) = func_psiT_v(ct(2 * M + 1:4 * M)) .* func_psiT_v(st(2 * M + 1:4 * M)) * func_psiT(yl:hy:yr, yr);



end