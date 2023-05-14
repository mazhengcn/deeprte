function [scattering_kernel, boundary, sigma, q, varepsilon, rand_params] = gen_inputs(params)

% generate scattering kernel
[scattering_kernel, rand_params.g] = gen_scattering_kernel(params.g_scope, params.N);
% generate boundary
if params.generate_train_data == true
    [boundary, rand_params.bc_params] = gen_train_boundary(params);
else
    [boundary, rand_params.bc_params] = gen_test_boundary(params);
end
% generate sigma, varepsilon and q
[sigma, q, varepsilon, rand_params.sigma_params] = gen_coef(params);

end
% generate scattering kernel
function [sigma, q, varepsilon, sigma_params] = gen_coef(params)
J = params.J; I = params.I;
yl = params.yl; xl = params.xl; yr = params.yr; xr = params.xr;
hy = params.hy; hx = params.hx;

varepsilon = zeros(I, J); sigma_t = zeros(I, J); sigma_a = zeros(I, J); q = zeros(I, J);


[in_sigma_a_start, in_sigma_a_range] = get_rand_params(params.in_sigma_a_scope);
[out_sigma_a_start, out_sigma_a_range] = get_rand_params(params.out_sigma_a_scope);
[in_sigma_t_start, in_sigma_t_range] = get_rand_params(params.in_sigma_t_scope);
[out_sigma_t_start, out_sigma_t_range] = get_rand_params(params.out_sigma_t_scope);

in_sigma_a = in_sigma_a_range * rand() + in_sigma_a_start;
out_sigma_a = out_sigma_a_range * rand() + out_sigma_a_start;
in_sigma_t = in_sigma_t_range * rand() + in_sigma_t_start;
out_sigma_t = out_sigma_t_range * rand() + out_sigma_t_start;

f_sigma_T = @(x, y)(out_sigma_t * (x <= xr) .* (y <= yr) + (in_sigma_t-out_sigma_t) * (0.4 <= x) .* (x <= 0.6) * (0.4 <= y) .* (y <= 0.6));
f_sigma_a = @(x, y)(out_sigma_a * (x <= xr) .* (y <= yr) + (in_sigma_a-out_sigma_a) * (0.4 <= x) .* (x <= 0.6) * (0.4 <= y) .* (y <= 0.6));
f_varepsilon = @(x, y)1 .* (x <= xr) .* (y <= yr);
f_q = @(x, y)(0) .* (x <= xr) .* (y <= yr);

for i = 1:I + 1
    for j = 1:J + 1
        sigma_t(i, j) = f_sigma_T(xl+(i - 1) * hx, yl+(j - 1) * hy);
        sigma_a(i, j) = f_sigma_a(xl+(i - 1) * hx, yl+(j - 1) * hy);
        q(i, j) = f_q((i - 1) * hx, (j - 1) * hy);
        varepsilon(i, j) = f_varepsilon((i - 1) * hx, (j - 1) * hy);
    end
end

sigma.sigma_a = sigma_a;
sigma.sigma_t = sigma_t;

sigma_params.in_sigma_a = in_sigma_a;
sigma_params.out_sigma_a = out_sigma_a;
sigma_params.in_sigma_t = in_sigma_t;
sigma_params.out_sigma_t = out_sigma_t;

end

% generate boundary
function [boundary, bc_params] = gen_train_boundary(params)

[var_x_start, var_x_range] = get_rand_params(params.var_x_scope);
[var_v_start, var_v_range] = get_rand_params(params.var_v_scope);

variance_x = var_x_range * rand([1, 4]) + var_x_start;
variance_v = var_v_range * rand([1, 4]) + var_v_start;

r_ind = randi(params.r_ind_scope, 1, 4);
v_ind = randi(params.v_ind_scope, 1, 4);

yl = params.yl; xl = params.xl; yr = params.yr; xr = params.xr;
hy = params.hy; hx = params.hx;

M = params.M; J = params.J; I = params.I;

ct = params.ct;
st = params.st;

y_l = r_ind(1) * hy + yl;
y_r = yr - r_ind(2) * hy;
x_b = xr - r_ind(3) * hx;
x_t = r_ind(4) * hx + xl;

func_psiL = @(x, y)(exp(- (y - y_l) .^ 2/2 / variance_x(1)) * exp(- (x - xl) .^ 2/2 / variance_x(1)));
func_psiL_v = @(x)(exp(- (x - x(v_ind(1))) .^ 2/2 / variance_v(1)));
func_psiR = @(x, y)(exp(- (y - y_r) .^ 2/2 / variance_x(2)) * exp(- (x - xr) .^ 2/2 / variance_x(2)));
func_psiR_v = @(x)(exp(- (x - x(v_ind(2))) .^ 2/2 / variance_v(2)));
func_psiB = @(x, y)(exp(- (y - yl) .^ 2/2 / variance_x(3)) * exp(- (x - x_b) .^ 2/2 / variance_x(3)));
func_psiB_v = @(x)(exp(- (x - x(v_ind(3))) .^ 2/2 / variance_v(3)));
func_psiT = @(x, y)(exp(- (y - yr) .^ 2/2 / variance_x(4)) * exp(- (x - x_t) .^ 2/2 / variance_x(4)));
func_psiT_v = @(x)(exp(- (x - x(v_ind(4))) .^ 2/2 / variance_v(4)));

sum_x_L = sum(func_psiL(xl, yl + hy:hy:yr - hy)) + sum(func_psiL(xr, yl + hy:hy:yr - hy)) + sum(func_psiL(hx:hx:xr, yl)) + sum(func_psiL(xl:hx:xr, yr));
sum_x_R = sum(func_psiR(xl, yl + hy:hy:yr - hy)) + sum(func_psiR(xr, yl + hy:hy:yr - hy)) + sum(func_psiR(xl:hx:xr, yl)) + sum(func_psiR(xl:hx:xr, yr));
sum_x_B = sum(func_psiB(xl, yl + hy:hy:yr - hy)) + sum(func_psiB(xr, yl + hy:hy:yr - hy)) + sum(func_psiB(xl:hx:xr, yl)) + sum(func_psiB(xl:hx:xr, yr));
sum_x_T = sum(func_psiT(xl, yl + hy:hy:yr - hy)) + sum(func_psiT(xr, yl + hy:hy:yr - hy)) + sum(func_psiT(xl:hx:xr, yl)) + sum(func_psiT(xl:hx:xr, yr));

sum_v_L = sqrt(sum(func_psiL_v([ct]) .* func_psiL_v([st])));
sum_v_R = sqrt(sum(func_psiR_v([ct]) .* func_psiR_v([st])));
sum_v_B = sqrt(sum(func_psiB_v([ct]) .* func_psiB_v([st])));
sum_v_T = sqrt(sum(func_psiT_v([ct]) .* func_psiT_v([st])));

func_psiL = @(x, y)(1/sum_x_L * exp(- (y - y_l) .^ 2/2 / variance_x(1)) * exp(- (x - xl) .^ 2/2 / variance_x(1)));
func_psiL_v = @(x)(1/sum_v_L *exp(- (x - x(v_ind(1))) .^ 2/2 / variance_v(1)));
% func_psiL_v = @(x)(exp(- (x - x(v_index(1))).^2/2 / variance_v(1)));
func_psiR = @(x, y)(1/sum_x_R * exp(- (y - y_r) .^ 2/2 / variance_x(2)) * exp(- (x - xr) .^ 2/2 / variance_x(2)));
func_psiR_v = @(x)(1/sum_v_R *exp(- (x - x(v_ind(2))) .^ 2/2 / variance_v(2)));
func_psiB = @(x, y)(1/sum_x_B * exp(- (y - yl) .^ 2/2 / variance_x(3)) * exp(- (x - x_b) .^ 2/2 / variance_x(3)));
func_psiB_v = @(x)(1/sum_v_B *exp(- (x - x(v_ind(3))) .^ 2/2 / variance_v(3)));
func_psiT = @(x, y)(1/sum_x_T * exp(- (y - yr) .^ 2/2 / variance_x(4)) * exp(- (x - x_t) .^ 2/2 / variance_x(4)));
func_psiT_v = @(x)(1/sum_v_T *exp(- (x - x(v_ind(4))) .^ 2/2 / variance_v(4)));

func_list_x = {func_psiL, func_psiR, func_psiB, func_psiT};
func_list_v = {func_psiL_v, func_psiR_v, func_psiB_v, func_psiT_v};

boundary.psiL = zeros(2*M, J+1); boundary.psiR = zeros(2*M, J+1); boundary.psiB = zeros(2*M, I+1); boundary.psiT = zeros(2*M, I+1);

for i = 1:4
    boundary.psiL(:, :) = boundary.psiL(:, :) + func_list_v{i}([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_list_v{i}([st(3 * M + 1:4 * M); st(1:M)]) * func_list_x{i}(xl, yl:hy:yr);
    boundary.psiR(:, :) = boundary.psiR(:, :) + func_list_v{i}(ct(1 * M + 1:3 * M)) .* func_list_v{i}(st(1 * M + 1:3 * M)) * func_list_x{i}(xr, yl:hy:yr);
    boundary.psiB(:, :) = boundary.psiB(:, :) + func_list_v{i}(ct(0 * M + 1:2 * M)) .* func_list_v{i}(st(0 * M + 1:2 * M)) * func_list_x{i}(xl:hx:xr, yl);
    boundary.psiT(:, :) = boundary.psiT(:, :) + func_list_v{i}(ct(2 * M + 1:4 * M)) .* func_list_v{i}(st(2 * M + 1:4 * M)) * func_list_x{i}(yl:hy:yr, yr);
end

bc_params.v_ind = v_ind;
bc_params.r_ind = r_ind;
bc_params.variance_v = variance_v;
bc_params.variance_x = variance_x;

end

% generate sigma
function [scattering_kernel, g] = gen_scattering_kernel(g_scope, N)

[start, range] = get_rand_params(g_scope);
g = rand()*(range)+start;
scattering_kernel = P2generator(N, g);

end

function [start, range] = get_rand_params(input_scope)
start = input_scope(1);
range = input_scope(2)-input_scope(1);

end
