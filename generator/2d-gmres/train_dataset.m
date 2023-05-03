% clear all
addpath(genpath('./gmres'));
clc

tic
%% 离散设置
output = param_train();

N = output{1}(1); I = output{1}(2); J = output{1}(3); xl = output{1}(4); xr = output{1}(5); yl = output{1}(6); yr = output{1}(7);

N_itr = output{1}(8);

func_psibc = output{2}{1};
func_psibc_v = output{2}{2};
params_var_x = output{2}{3};
params_var_v = output{2}{4};
range_c_ind = output{2}{5};

omega_c = output{3}{1}; params_sigma_T = output{3}{2}; params_sigma_a = output{3}{3}; params_q = output{3}{4};

path = output{4}{1};
hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell
[omega, ct, st, M, theta] = qnwlege2(N);

%% 准备储存空间
list_psiL = zeros(2 * M, J, N_itr); list_psiR = list_psiL;
list_psiB = zeros(2 * M, I, N_itr); list_psiT = list_psiB;

%% 生成随机系数并构建入射函数

list_var = zeros(2, 4, N_itr);
list_yhat = zeros(4, N_itr);
list_v_index = zeros(4, N_itr);
mesh_L_theta = [theta(3 * M + 1:4 * M); theta(1:M)] .* ones(1, J);
mesh_R_theta = theta(1 * M + 1:3 * M) .* ones(1, J);
mesh_B_theta = theta(0 * M + 1:2 * M) .* ones(1, I);
mesh_T_theta = theta(2 * M + 1:4 * M) .* ones(1, I);

for n = 1:N_itr
    variance_x = params_var_x(1) * rand([1, 4]) + params_var_x(2);
    variance_v = params_var_v(1) * rand([1, 4]) + params_var_v(2);
    % variance_x = [0.0012, 0.0085, 0.0168, 0.0099];
    % variance_v = [0.0020, 0.0091, 0.0076, 0.0055];
    % variance_x = 0.02 * zeros(1, 4) + 0.001
    % variance_v = 0.01 * zeros(1, 4) + 0.001
    c_ind = randi(range_c_ind, 1, 4);
    v_index = randi(2 * M, 4, 1);
    % v_index = [4; 11; 4; 6];
    
    list_v_index(:, n) = v_index;
    y_l = (c_ind(1) - 0.5) * hy + yl;
    y_r = yr - (c_ind(2) - 0.5) * hy;
    x_b = xr - (c_ind(3) - 0.5) * hx;
    x_t = (c_ind(4) - 0.5) * hx + xl;
    
    func_psiL = @(x, y)(func_psibc(x, y, xl, y_l, variance_x(1)));
    func_psiL_v = @(v)(func_psibc_v(v, v(list_v_index(1, n)), variance_v(1)));
    % func_psiL_v = @(x)(exp(- (x - x(v_index(1))).^2/2 / variance_v(1)));
    func_psiR = @(x, y)(func_psibc(x, y, xr, y_r, variance_x(2)));
    func_psiR_v = @(v)(func_psibc_v(v, v(list_v_index(2, n)), variance_v(2)));
    func_psiB = @(x, y)(func_psibc(x, y, x_b, yl, variance_x(3)));
    func_psiB_v = @(v)(func_psibc_v(v, v(list_v_index(3, n)), variance_v(3)));
    func_psiT = @(x, y)(func_psibc(x, y, x_t, yr, variance_x(4)));
    func_psiT_v = @(v)(func_psibc_v(v, v(list_v_index(4, n)), variance_v(4)));
    
    sum_x_L = sum(func_psiL(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiL(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiL(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl) + func_psiL(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr));
    sum_x_R = sum(func_psiR(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiR(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiR(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl) + func_psiR(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr));
    sum_x_B = sum(func_psiB(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiB(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiB(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl) + func_psiB(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr));
    sum_x_T = sum(func_psiT(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiT(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy) + func_psiT(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl) + func_psiT(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr));
    
    sum_v_L = sqrt(sum(func_psiL_v([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_psiL_v([st(3 * M + 1:4 * M); st(1:M)])));
    sum_v_R = sqrt(sum(func_psiR_v(ct(1 * M + 1:3 * M)) .* func_psiR_v(st(1 * M + 1:3 * M))));
    sum_v_B = sqrt(sum(func_psiB_v(ct(0 * M + 1:2 * M)) .* func_psiB_v(st(0 * M + 1:2 * M))));
    sum_v_T = sqrt(sum(func_psiT_v(ct(2 * M + 1:4 * M)) .* func_psiT_v(st(2 * M + 1:4 * M))));
    
    func_psiL = @(x, y)(1 / sum_x_L * func_psibc(x, y, xl, y_l, variance_x(1)));
    func_psiL_v = @(v)(1 / sum_v_L * func_psibc_v(v, v(list_v_index(1, n)), variance_v(1)));
    % func_psiL_v = @(x)(exp(- (x - x(v_index(1))).^2/2 / variance_v(1)));
    func_psiR = @(x, y)(1 / sum_x_R * func_psibc(x, y, xr, y_r, variance_x(2)));
    func_psiR_v = @(v)(1 / sum_v_R * func_psibc_v(v, v(list_v_index(2, n)), variance_v(2)));
    func_psiB = @(x, y)(1 / sum_x_B * func_psibc(x, y, x_b, yl, variance_x(3)));
    func_psiB_v = @(v)(1 / sum_v_B * func_psibc_v(v, v(list_v_index(3, n)), variance_v(3)));
    func_psiT = @(x, y)(1 / sum_x_T * func_psibc(x, y, x_t, yr, variance_x(4)));
    func_psiT_v = @(v)(1 / sum_v_T * func_psibc_v(v, v(list_v_index(4, n)), variance_v(4)));
    
    func_list_x = {func_psiL, func_psiR, func_psiB, func_psiT};
    func_list_v = {func_psiL_v, func_psiR_v, func_psiB_v, func_psiT_v};
    
    for i = 1:4
        list_psiL(:, :, n) = list_psiL(:, :, n) + func_list_v{i}([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_list_v{i}([st(3 * M + 1:4 * M); st(1:M)]) * func_list_x{i}(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy);
        list_psiR(:, :, n) = list_psiR(:, :, n) + func_list_v{i}(ct(1 * M + 1:3 * M)) .* func_list_v{i}(st(1 * M + 1:3 * M)) * func_list_x{i}(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy);
        list_psiB(:, :, n) = list_psiB(:, :, n) + func_list_v{i}(ct(0 * M + 1:2 * M)) .* func_list_v{i}(st(0 * M + 1:2 * M)) * func_list_x{i}(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl);
        list_psiT(:, :, n) = list_psiT(:, :, n) + func_list_v{i}(ct(2 * M + 1:4 * M)) .* func_list_v{i}(st(2 * M + 1:4 * M)) * func_list_x{i}(yl + 0.5 * hy:hy:yr - 0.5 * hy, yr);
    end
    
    list_var(1, :, n) = variance_x;
    list_var(2, :, n) = variance_v;
    
    list_yhat(:, n) = c_ind;
end

%% 指定散射截面，源项会发生变化的区域(Omega_C)
Omega_C = @(x, y) (x >= omega_c(1)) .* (x <= omega_c(2)) .* (y >= omega_c(3)) .* (y <= omega_c(4));
[Xc, Yc] = meshgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy);
[row, col] = find(Omega_C(Xc, Yc) > 0);
LC = [row, col]; %Omega_C对应的网格集合

%% Omega_C外不变的散射截面，源项
f_varepsilon = @(x, y)1 .* (x <= xr) .* (y <= yr);
f_sigma_T = @(x, y)(params_sigma_T(1)) .* (x <= xr) .* (y <= yr);
f_sigma_a = @(x, y)(params_sigma_a(1)) .* (x <= xr) .* (y <= yr);
f_q = @(x, y)(params_q(1)) .* (x <= xr) .* (y <= yr);

%% Omega_C内的散射截面，源项
g_varepsilon = cell(N_itr, 1); g_sigma_T = g_varepsilon; g_sigma_a = g_varepsilon; g_q = g_varepsilon;

for n = 1:N_itr
    g_varepsilon{n} = @(x, y)1 * (x <= xr) .* (y <= yr);
    g_sigma_T{n} = @(x, y)(params_sigma_T(2)) .* (x <= xr) .* (y <= yr);
    g_sigma_a{n} = @(x, y)(params_sigma_a(2)) .* (x <= xr) .* (y <= yr);
    g_q{n} = @(x, y)(params_q(2)) .* (x <= xr) .* (y <= yr);
end

T_offline_part1 = toc;
%% 运行主程序
Input = {[N I J xl xr yl yr], {f_sigma_T, f_sigma_a, f_varepsilon, f_q, LC}, {list_psiL, list_psiR, list_psiB, list_psiT, g_sigma_T, g_sigma_a, g_varepsilon, g_q}};
[list_psi_x, list_psi_y, list_alpha, list_Psi, list_Phi, list_varepsilon, list_sigma_T, list_sigma_a, list_q, ...
    T_offline_part2, T_online_each] = run_main(Input);
T_offline = T_offline_part1 + T_offline_part2
%% generate mat file

% Input = {[N I J xl xr yl yr M], {list_psiL, list_psiR, list_psiB, list_psiT}, {omega, list_sigma_a, list_sigma_t, ct, st}, {list_Phi, list_Psi}};

% [psi_label phi rv psi_bc rv_prime omega_prime sigma_a sigma_t r ct st omega x y w_angle] = convert_data(Input);

psi_label = permute(list_Psi, [4 2 3 1]);
phi = permute(list_Phi, [3 1 2]);
psiL = permute(list_psiL, [3 2 1]);
psiR = permute(list_psiR, [3 2 1]);
psiB = permute(list_psiB, [3 2 1]);
psiT = permute(list_psiT, [3 2 1]);
rv = zeros(I, J, 4 * M, 4);
[x, y, vx] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy, ct);
[x, y, vy] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy, st);
rv(:, :, :, 1) = x;
rv(:, :, :, 2) = y;
rv(:, :, :, 3) = vx;
rv(:, :, :, 4) = vy;

psi_bc = cat(2, psiL, psiR, psiB, psiT);
omega = squeeze(omega);
% psil
[x, y, vx_l] = ndgrid(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy, ct(ct > 0));
[x, y, vy_l] = ndgrid(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy, st(ct > 0));
[x, y, omega_l] = ndgrid(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy, omega(ct > 0));
rv_l = zeros(J, 2 * M, 4);
rv_l(:, :, 1) = squeeze(x);
rv_l(:, :, 2) = squeeze(y);
rv_l(:, :, 3) = squeeze(vx_l);
rv_l(:, :, 4) = squeeze(vy_l);

[x, y, vx_r] = ndgrid(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy, ct(ct < 0));
[x, y, vy_r] = ndgrid(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy, st(ct < 0));
[x, y, omega_r] = ndgrid(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy, omega(ct < 0));
rv_r = zeros(J, 2 * M, 4);
rv_r(:, :, 1) = squeeze(x);
rv_r(:, :, 2) = squeeze(y);
rv_r(:, :, 3) = squeeze(vx_r);
rv_r(:, :, 4) = squeeze(vy_r);

[x, y, vx_b] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl, ct(st > 0));
[x, y, vy_b] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl, st(st > 0));
[x, y, omega_b] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl, omega(st > 0));
rv_b = zeros(I, 2 * M, 4);
rv_b(:, :, 1) = squeeze(x);
rv_b(:, :, 2) = squeeze(y);
rv_b(:, :, 3) = squeeze(vx_b);
rv_b(:, :, 4) = squeeze(vy_b);

[x, y, vx_t] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr, ct(st < 0));
[x, y, vy_t] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr, st(st < 0));
[x, y, omega_t] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yr, omega(st < 0));
rv_t = zeros(I, 2 * M, 4);
rv_t(:, :, 1) = squeeze(x);
rv_t(:, :, 2) = squeeze(y);
rv_t(:, :, 3) = squeeze(vx_t);
rv_t(:, :, 4) = squeeze(vy_t);

rv_prime = cat(1, rv_l, rv_r, rv_b, rv_t);
omega_prime = cat(1, squeeze(omega_l), squeeze(omega_r), squeeze(omega_b), squeeze(omega_t));

sigma_a = permute(list_sigma_a, [3 1 2]);
sigma_t = permute(list_sigma_T, [3 1 2]);

[x, y] = ndgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy);
r = zeros(I, J, 2);
r(:, :, 1) = x;
r(:, :, 2) = y;

ct = squeeze(ct);
st = squeeze(st);

x = squeeze(xl + 0.5 * hx:hx:xr - 0.5 * hx);
y = squeeze(yl + 0.5 * hy:hy:yr - 0.5 * hy);
w_angle = omega;

save(path, 'psi_label', 'phi', 'rv', 'psi_bc', 'rv_prime', 'omega_prime', 'sigma_a', 'sigma_t', 'r', 'ct', 'st', 'omega', 'x', 'y', 'w_angle')

rmpath(genpath('./gmres'));
