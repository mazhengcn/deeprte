clear all
clc

tic
%% 离散设置
N = 3; %2*N*(N+1) is the size of quadrature set
xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell
[omega, ct, st, M, theta] = qnwlege2(N);

%% 准备储存空间
N_itr = 500;
list_psiL = zeros(2 * M, J, N_itr); list_psiR = list_psiL;
list_psiB = zeros(2 * M, I, N_itr); list_psiT = list_psiB;

%% 生成随机系数并构建入射函数
list_A = (rand(N_itr, 4) - 0.5) * 20;
list_k = ceil((rand(N_itr, 4)) * 50);
list_b = (rand(N_itr, 2) - 0.5) * 0.02;
list_a = (rand(N_itr, 2) - 0.5) * 2;
list_var = zeros(2, 4, N_itr);
list_yhat = zeros(4, N_itr);
list_v_index = randi(2 * M, 4, N_itr);
mesh_L_theta = [theta(3 * M + 1:4 * M); theta(1:M)] .* ones(1, J);
mesh_R_theta = theta(1 * M + 1:3 * M) .* ones(1, J);
mesh_B_theta = theta(0 * M + 1:2 * M) .* ones(1, I);
mesh_T_theta = theta(2 * M + 1:4 * M) .* ones(1, I);

for n = 1:N_itr
    variance_x = 0.02 * rand([1, 4]) + 0.005;
    variance_v = 0.01 * rand([1, 4]) + 0.005;
    % variance_x = [0.0012, 0.0085, 0.0168, 0.0099];
    % variance_v = [0.0020, 0.0091, 0.0076, 0.0055];
    % variance_x = 0.02 * zeros(1, 4) + 0.001
    % variance_v = 0.01 * zeros(1, 4) + 0.001
    c_ind = randi([1, 40], 1, 4);
    v_index = randi(M, 4, 1);
    % v_index = [4; 11; 4; 6];

    list_v_index(:, n) = v_index;
    y_l = (c_ind(1) - 0.5) * hy + yl;
    y_r = yr - (c_ind(2) - 0.5) * hy;
    x_b = xr - (c_ind(3) - 0.5) * hx;
    x_t = (c_ind(4) - 0.5) * hx + xl;

    func_psiL = @(x, y)(list_A(n, 1) * sin(list_k(n, 1) * pi * y) + 10);
    func_psiL_v = @(x)(1 + 0 .* x);
    func_psiR = @(x, y)(list_A(n, 2) * sin(list_k(n, 2) * pi * y) + 10);
    func_psiR_v = @(x)(1 + 0 .* x);
    func_psiB = @(x, y)(list_A(n, 3) * sin(list_k(n, 3) * pi * x) + 10);
    func_psiB_v = @(x)(1 + 0 .* x);
    func_psiT = @(x, y)(list_A(n, 4) * sin(list_k(n, 4) * pi * x) + 10);
    func_psiT_v = @(x)(1 + 0 .* x);

    func_list_x = {func_psiL, func_psiR, func_psiB, func_psiT};
    func_list_v = {func_psiL_v, func_psiR_v, func_psiB_v, func_psiT_v};

    list_psiL(:, :, n) = func_psiL_v([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_psiL_v([st(3 * M + 1:4 * M); st(1:M)]) * func_psiL(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy);
    list_psiR(:, :, n) = func_psiR_v(ct(1 * M + 1:3 * M)) .* func_psiR_v(st(1 * M + 1:3 * M)) * func_psiR(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy);
    list_psiB(:, :, n) = func_psiB_v(ct(0 * M + 1:2 * M)) .* func_psiB_v(st(0 * M + 1:2 * M)) * func_psiB(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl);
    list_psiT(:, :, n) = func_psiT_v(ct(2 * M + 1:4 * M)) .* func_psiT_v(st(2 * M + 1:4 * M)) * func_psiT(yl + 0.5 * hy:hy:yr - 0.5 * hy, yr);

    list_var(1, :, n) = variance_x;
    list_var(2, :, n) = variance_v;

    list_yhat(:, n) = c_ind;
end

%% 指定散射截面，源项会发生变化的区域(Omega_C)
Omega_C = @(x, y) (x >= 0.4) .* (x <= 0.6) .* (y >= 0.4) .* (y <= 0.6);
[Xc, Yc] = meshgrid(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy);
[row, col] = find(Omega_C(Xc, Yc) > 0);
LC = [row, col]; %Omega_C对应的网格集合

%% Omega_C外不变的散射截面，源项
f_varepsilon = @(x, y)1 .* (x <= xr) .* (y <= yr);
f_sigma_T = @(x, y)(10) .* (x <= xr) .* (y <= yr);
f_sigma_a = @(x, y)(5) .* (x <= xr) .* (y <= yr);
f_q = @(x, y)(0) .* (x <= xr) .* (y <= yr);

%% Omega_C内的散射截面，源项
g_varepsilon = cell(N_itr, 1); g_sigma_T = g_varepsilon; g_sigma_a = g_varepsilon; g_q = g_varepsilon;

for n = 1:N_itr
    g_varepsilon{n} = @(x, y)1 * (x <= xr) .* (y <= yr);
    g_sigma_T{n} = @(x, y)(5) .* (x <= xr) .* (y <= yr);
    g_sigma_a{n} = @(x, y)(2) .* (x <= xr) .* (y <= yr);
    g_q{n} = @(x, y)(0) .* (x <= xr) .* (y <= yr);
end

T_offline_part1 = toc;
%% 运行主程序
Input = {[N I J xl xr yl yr], {f_sigma_T, f_sigma_a, f_varepsilon, f_q, LC}, {list_psiL, list_psiR, list_psiB, list_psiT, g_sigma_T, g_sigma_a, g_varepsilon, g_q}};
[list_psi_x, list_psi_y, list_alpha, list_Psi, list_Phi, list_varepsilon, list_sigma_T, list_sigma_a, list_q, ...
        T_offline_part2, T_online_each] = run_main(Input);
T_offline = T_offline_part1 + T_offline_part2
%% generate mat file

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

save test_sin.mat psi_label phi rv psi_bc rv_prime omega_prime sigma_a sigma_t r ct st omega x y w_angle
