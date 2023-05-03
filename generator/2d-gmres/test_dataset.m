clear all
addpath(genpath('./gmres'));
clc

tic
%% 离散设置
output = param_test();

N = output{1}(1); I = output{1}(2); J = output{1}(3); xl = output{1}(4); xr = output{1}(5); yl = output{1}(6); yr = output{1}(7);

N_itr = output{1}(8);

func_bc_list_x = output{2}{1};
func_bc_list_v = output{2}{2};
params_x = output{2}{3};
params_v = output{2}{4};
% params_var_v = output{2}{4};
% range_c_ind = output{2}{5};

omega_c = output{3}{1}; params_sigma_T = output{3}{2}; params_sigma_a = output{3}{3}; params_q = output{3}{4};

path = output{4}{1};
hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell
[omega, ct, st, M, theta] = qnwlege2(N);

%% 准备储存空间
list_psiL = zeros(2 * M, J, N_itr); list_psiR = list_psiL;
list_psiB = zeros(2 * M, I, N_itr); list_psiT = list_psiB;

%% 生成随机系数并构建入射函数

mesh_L_theta = [theta(3 * M + 1:4 * M); theta(1:M)] .* ones(1, J);
mesh_R_theta = theta(1 * M + 1:3 * M) .* ones(1, J);
mesh_B_theta = theta(0 * M + 1:2 * M) .* ones(1, I);
mesh_T_theta = theta(2 * M + 1:4 * M) .* ones(1, I);

for n = 1:N_itr
    
    func_psiL = @(x, y)(func_bc_list_x{1}(x, y, params_x(n, 1, :)));
    func_psiL_v = @(v)(func_bc_list_v{1}(v, params_v(n, 1, :)));
    func_psiR = @(x, y)(func_bc_list_x{2}(x, y, params_x(n, 2, :)));
    func_psiR_v = @(v)(func_bc_list_v{2}(v, params_v(n, 2, :)));
    func_psiB = @(x, y)(func_bc_list_x{3}(x, y, params_x(n, 3, :)));
    func_psiB_v = @(v)(func_bc_list_v{3}(v, params_v(n, 3, :)));
    func_psiT = @(x, y)(func_bc_list_x{4}(x, y, params_x(n, 4, :)));
    func_psiT_v = @(v)(func_bc_list_v{4}(v, params_v(n, 4, :)));
    
    list_psiL(:, :, n) = func_psiL_v([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_psiL_v([st(3 * M + 1:4 * M); st(1:M)]) * func_psiL(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy);
    list_psiR(:, :, n) = func_psiR_v(ct(1 * M + 1:3 * M)) .* func_psiR_v(st(1 * M + 1:3 * M)) * func_psiR(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy);
    list_psiB(:, :, n) = func_psiB_v(ct(0 * M + 1:2 * M)) .* func_psiB_v(st(0 * M + 1:2 * M)) * func_psiB(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl);
    list_psiT(:, :, n) = func_psiT_v(ct(2 * M + 1:4 * M)) .* func_psiT_v(st(2 * M + 1:4 * M)) * func_psiT(yl + 0.5 * hy:hy:yr - 0.5 * hy, yr);
    
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
