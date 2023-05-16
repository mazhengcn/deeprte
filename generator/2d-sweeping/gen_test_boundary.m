function [boundary, bc_params] = gen_test_boundary(params)

if params.test_bc_type == "constant"
    [boundary, bc_params] = bc1_boundary(params);
elseif params.test_bc_type == "sin"
    [boundary, bc_params] = sin_boundary(params);
end



end

function [boundary, bc_params] = bc1_boundary(params)

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

bc_params=[];

end

function [boundary, bc_params] = sin_boundary(params)

yl = params.yl; xl = params.xl; yr = params.yr; xr = params.xr;
hy = params.hy; hx = params.hx;

M = params.M; J = params.J; I = params.I;

ct = params.ct;
st = params.st;

[a_start, a_range] = get_rand_params(params.amplitude_scope);
[k_start, k_range] = get_rand_params(params.wavenumber_scope);

shift = params.amplitude_scope(end);

amplitude = a_range * rand([1, 4]) + a_start;
wavenumber = k_range * rand([1, 4]) + k_start;

func_psiL = @(x, y)(amplitude(1)*sin(wavenumber(1) * y)+shift);
func_psiL_v = @(x)(1 + 0 .* x);
func_psiR = @(x, y)(amplitude(2)*sin(wavenumber(2) * y)+shift);
func_psiR_v = @(x)(1 + 0 .* x);
func_psiB = @(x, y)(amplitude(3)*sin(wavenumber(3) * x)+shift);
func_psiB_v = @(x)(1 + 0 .* x);
func_psiT = @(x, y)(amplitude(4)*sin(wavenumber(4) * x)+shift);
func_psiT_v = @(x)(1 + 0 .* x);

boundary.psiL = zeros(2*M, J+1); boundary.psiR = zeros(2*M, J+1); boundary.psiB = zeros(2*M, I+1); boundary.psiT = zeros(2*M, I+1);

boundary.psiL(:, :) = func_psiL_v([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_psiL_v([st(3 * M + 1:4 * M); st(1:M)]) * func_psiL(xl, yl:hy:yr);
boundary.psiR(:, :) = func_psiR_v(ct(1 * M + 1:3 * M)) .* func_psiR_v(st(1 * M + 1:3 * M)) * func_psiR(xr, yl:hy:yr);
boundary.psiB(:, :) = func_psiB_v(ct(0 * M + 1:2 * M)) .* func_psiB_v(st(0 * M + 1:2 * M)) * func_psiB(xl:hx:xr, yl);
boundary.psiT(:, :) = func_psiT_v(ct(2 * M + 1:4 * M)) .* func_psiT_v(st(2 * M + 1:4 * M)) * func_psiT(yl:hy:yr, yr);

bc_params=[];

end

function [start, range] = get_rand_params(input_scope)
start = input_scope(1);
range = input_scope(2)-input_scope(1);

end