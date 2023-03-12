clear all
clc

tic
%% discretization setting
N = 3; %2*N*(N+1) is the size of quadrature set
xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell
[omega, ct, st, M, theta, ~] = qnwlege2(N);

N_itr = 10;
list_psiL = zeros(2 * M, J + 1, N_itr); list_psiR = list_psiL;
list_psiB = zeros(2 * M, I + 1, N_itr); list_psiT = list_psiB;
%% 生成随机系数并构建入射函数
list_b = (rand(N_itr, 2) - 0.5) * 0.02;
list_a = (rand(N_itr, 2) - 0.5) * 2;
list_g = floor(rand(1, N_itr) * 20) * 0.0125;
list_v_index = randi(2 * M, 4, N_itr);

% g = 0.75; %anisotropic coefficient
% K = P2generator(N, g); %Kernel matrix

%% Omega_C内的散射截面，源项
f_varepsilon = cell(N_itr, 1); f_sigma_T = f_varepsilon; f_sigma_a = f_varepsilon; f_q = f_varepsilon;

%% run main
list_Psi = zeros(4 * M, I + 1, J + 1, N_itr);
list_Phi = zeros(I + 1, J + 1, N_itr);
list_sigma_a = zeros(I + 1, J + 1, N_itr);
list_sigma_T = zeros(I + 1, J + 1, N_itr);

scattering_kernel = zeros(N_itr, 4 * M, 4 * M);

%% cross sections, external source term and boundary conditions
for n = 1:N_itr

    g = list_g(n); %anisotropic coefficient
    K = P2generator(N, g); %Kernel matrix
    scattering_kernel(n, :, :) = K;
    variance_x = 0.02 * rand([1, 4]) + 0.005;
    variance_v = 0.01 * rand([1, 4]) + 0.005;
    c_ind = randi([2, 40], 1, 4);
    v_index = randi(2 * M, 4, 1);

    list_v_index(:, n) = v_index;
    y_l = c_ind(1) * hy + yl;
    y_r = yr - c_ind(2) * hy;
    x_b = xr - c_ind(3) * hx;
    x_t = c_ind(4) * hx + xl;

    func_psiL = @(x, y)(exp(- (y - y_l) .^ 2/2 / variance_x(1)) * exp(- (x - xl) .^ 2/2 / variance_x(1)));
    func_psiL_v = @(x)(exp(- (x - x(list_v_index(1, n))) .^ 2/2 / variance_v(1)));
    % func_psiL_v = @(x)(exp(- (x - x(v_index(1))).^2/2 / variance_v(1)));
    func_psiR = @(x, y)(exp(- (y - y_r) .^ 2/2 / variance_x(2)) * exp(- (x - xr) .^ 2/2 / variance_x(2)));
    func_psiR_v = @(x)(exp(- (x - x(list_v_index(2, n))) .^ 2/2 / variance_v(2)));
    func_psiB = @(x, y)(exp(- (y - yl) .^ 2/2 / variance_x(3)) * exp(- (x - x_b) .^ 2/2 / variance_x(3)));
    func_psiB_v = @(x)(exp(- (x - x(list_v_index(3, n))) .^ 2/2 / variance_v(3)));
    func_psiT = @(x, y)(exp(- (y - yr) .^ 2/2 / variance_x(4)) * exp(- (x - x_t) .^ 2/2 / variance_x(4)));
    func_psiT_v = @(x)(exp(- (x - x(list_v_index(4, n))) .^ 2/2 / variance_v(4)));

    sum_x_L = sum(func_psiL(xl, yl + hy:hy:yr - hy)) + sum(func_psiL(xr, yl + hy:hy:yr - hy)) + sum(func_psiL(hx:hx:xr, yl)) + sum(func_psiL(xl:hx:xr, yr));
    sum_x_R = sum(func_psiR(xl, yl + hy:hy:yr - hy)) + sum(func_psiR(xr, yl + hy:hy:yr - hy)) + sum(func_psiR(xl:hx:xr, yl)) + sum(func_psiR(xl:hx:xr, yr));
    sum_x_B = sum(func_psiB(xl, yl + hy:hy:yr - hy)) + sum(func_psiB(xr, yl + hy:hy:yr - hy)) + sum(func_psiB(xl:hx:xr, yl)) + sum(func_psiB(xl:hx:xr, yr));
    sum_x_T = sum(func_psiT(xl, yl + hy:hy:yr - hy)) + sum(func_psiT(xr, yl + hy:hy:yr - hy)) + sum(func_psiT(xl:hx:xr, yl)) + sum(func_psiT(xl:hx:xr, yr));

    sum_v_L = sqrt(sum(func_psiL_v(ct) .* func_psiL_v(st)));
    sum_v_R = sqrt(sum(func_psiR_v(ct) .* func_psiR_v(st)));
    sum_v_B = sqrt(sum(func_psiB_v(ct) .* func_psiB_v(st)));
    sum_v_T = sqrt(sum(func_psiT_v(ct) .* func_psiT_v(st)));

    func_psiL = @(x, y)(5 / sum_x_L * exp(- (y - y_l) .^ 2/2 / variance_x(1)) * exp(- (x - xl) .^ 2/2 / variance_x(1)));
    func_psiL_v = @(x)(1 / sum_v_L * exp(- (x - x(list_v_index(1, n))) .^ 2/2 / variance_v(1)));
    func_psiR = @(x, y)(5 / sum_x_R * exp(- (y - y_r) .^ 2/2 / variance_x(2)) * exp(- (x - xr) .^ 2/2 / variance_x(2)));
    func_psiR_v = @(x)(1 / sum_v_R * exp(- (x - x(list_v_index(2, n))) .^ 2/2 / variance_v(2)));
    func_psiB = @(x, y)(5 / sum_x_B * exp(- (y - yl) .^ 2/2 / variance_x(3)) * exp(- (x - x_b) .^ 2/2 / variance_x(3)));
    func_psiB_v = @(x)(1 / sum_v_B * exp(- (x - x(list_v_index(3, n))) .^ 2/2 / variance_v(3)));
    func_psiT = @(x, y)(5 / sum_x_T * exp(- (y - yr) .^ 2/2 / variance_x(4)) * exp(- (x - x_t) .^ 2/2 / variance_x(4)));
    func_psiT_v = @(x)(1 / sum_v_T * exp(- (x - x(list_v_index(4, n))) .^ 2/2 / variance_v(4)));

    func_list_x = {func_psiL, func_psiR, func_psiB, func_psiT};
    func_list_v = {func_psiL_v, func_psiR_v, func_psiB_v, func_psiT_v};

    for i = 1:4
        list_psiL(:, :, n) = list_psiL(:, :, n) + func_list_v{i}([ct(3 * M + 1:4 * M); ct(1:M)]) .* func_list_v{i}([st(3 * M + 1:4 * M); st(1:M)]) * func_list_x{i}(xl, yl:hy:yr);
        list_psiR(:, :, n) = list_psiR(:, :, n) + func_list_v{i}(ct(1 * M + 1:3 * M)) .* func_list_v{i}(st(1 * M + 1:3 * M)) * func_list_x{i}(xr, yl:hy:yr);
        list_psiB(:, :, n) = list_psiB(:, :, n) + func_list_v{i}(ct(0 * M + 1:2 * M)) .* func_list_v{i}(st(0 * M + 1:2 * M)) * func_list_x{i}(xl:hx:xr, yl);
        list_psiT(:, :, n) = list_psiT(:, :, n) + func_list_v{i}(ct(2 * M + 1:4 * M)) .* func_list_v{i}(st(2 * M + 1:4 * M)) * func_list_x{i}(yl:hy:yr, yr);
    end

    f_varepsilon{n} = @(x, y)1 .* (x <= xr) .* (y <= yr);
    f_sigma_T{n} = @(x, y)(10 * (x <= xr) .* (y <= yr) - (5) * (0.4 <= x) .* (x <= 0.6) * (0.4 <= y) .* (y <= 0.6));
    f_sigma_a{n} = @(x, y)(5 * (x <= xr) .* (y <= yr) - (3) * (0.4 <= x) .* (x <= 0.6) * (0.4 <= y) .* (y <= 0.6));
    f_q{n} = @(x, y)(0) .* (x <= xr) .* (y <= yr);

    psiL = list_psiL(:, 2:40, n);
    psiR = list_psiR(:, 2:40, n); % i=I+1, j=2:J, m=M+1:4*M
    psiB = list_psiB(:, 2:40, n); % i=2:I, j=1,   m=1:2*M
    psiT = list_psiT(:, 2:40, n); % i=2:I, j=J+1, m=2*M+1:4*M

    psiLB = list_psiL(M + 1:2 * M, 1, n); % i=1,   j=1,   m=1:M
    psiLT = list_psiL(1:M, I + 1, n); % i=1,   j=J+1, m=3*M+1:4*M
    psiRB = list_psiR(1:M, 1, n); % i=I+1, j=1,   m=M+1:2*M
    psiRT = list_psiR(M + 1:2 * M, 1, n); % i=I+1, j=J+1, m=2*M+1:3*M

    [T, maxerrPsi, maxerrPhi, phi_final, psi_final, sigma_T, sigma_a] = run_main(K, N, I, J, xl, xr, yl, yr, f_sigma_T{n}, f_sigma_a{n}, f_varepsilon{n}, f_q{n}, psiL, psiR, psiB, psiT, psiLB, psiLT, psiRB, psiRT);

    list_Psi(:, :, :, n) = psi_final;
    list_Phi(:, :, n) = phi_final;
    list_sigma_a(:, :, n) = sigma_a;
    list_sigma_T(:, :, n) = sigma_T;
end

psi_label = permute(list_Psi, [4 2 3 1]);
phi = permute(list_Phi, [3 1 2]);
psiL = permute(list_psiL, [3 2 1]);
psiR = permute(list_psiR, [3 2 1]);
psiB = permute(list_psiB, [3 2 1]);
psiT = permute(list_psiT, [3 2 1]);
rv = zeros(I + 1, J + 1, 4 * M, 4);
[x, y, vx] = ndgrid(xl:hx:xr, yl:hy:yr, ct);
[x, y, vy] = ndgrid(xl:hx:xr, yl:hy:yr, st);
rv(:, :, :, 1) = x;
rv(:, :, :, 2) = y;
rv(:, :, :, 3) = vx;
rv(:, :, :, 4) = vy;

psi_bc = cat(2, psiL, psiR, psiB, psiT);
omega = squeeze(omega);
% psil
[x, y, vx_l] = ndgrid(xl, yl:hy:yr, [ct(3 * M + 1:4 * M); ct(1:M)]);
[x, y, vy_l] = ndgrid(xl, yl:hy:yr, [st(3 * M + 1:4 * M); st(1:M)]);
[x, y, omega_l] = ndgrid(xl, yl:hy:yr, [omega(3 * M + 1:4 * M); omega(1:M)]);
rv_l = zeros(J + 1, 2 * M, 4);
rv_l(:, :, 1) = squeeze(x);
rv_l(:, :, 2) = squeeze(y);
rv_l(:, :, 3) = squeeze(vx_l);
rv_l(:, :, 4) = squeeze(vy_l);

[x, y, vx_r] = ndgrid(xr, yl:hy:yr, ct(ct < 0));
[x, y, vy_r] = ndgrid(xr, yl:hy:yr, st(ct < 0));
[x, y, omega_r] = ndgrid(xr, yl:hy:yr, omega(ct < 0));
rv_r = zeros(J + 1, 2 * M, 4);
rv_r(:, :, 1) = squeeze(x);
rv_r(:, :, 2) = squeeze(y);
rv_r(:, :, 3) = squeeze(vx_r);
rv_r(:, :, 4) = squeeze(vy_r);

[x, y, vx_b] = ndgrid(xl:hx:xr, yl, ct(st > 0));
[x, y, vy_b] = ndgrid(xl:hx:xr, yl, st(st > 0));
[x, y, omega_b] = ndgrid(xl:hx:xr, yl, omega(st > 0));
rv_b = zeros(I + 1, 2 * M, 4);
rv_b(:, :, 1) = squeeze(x);
rv_b(:, :, 2) = squeeze(y);
rv_b(:, :, 3) = squeeze(vx_b);
rv_b(:, :, 4) = squeeze(vy_b);

[x, y, vx_t] = ndgrid(xl:hx:xr, yr, ct(st < 0));
[x, y, vy_t] = ndgrid(xl:hx:xr, yr, st(st < 0));
[x, y, omega_t] = ndgrid(xl:hx:xr, yr, omega(st < 0));
rv_t = zeros(I + 1, 2 * M, 4);
rv_t(:, :, 1) = squeeze(x);
rv_t(:, :, 2) = squeeze(y);
rv_t(:, :, 3) = squeeze(vx_t);
rv_t(:, :, 4) = squeeze(vy_t);

rv_prime = cat(1, rv_l, rv_r, rv_b, rv_t);
omega_prime = cat(1, squeeze(omega_l), squeeze(omega_r), squeeze(omega_b), squeeze(omega_t));

sigma_a = permute(list_sigma_a, [3 1 2]);
sigma_t = permute(list_sigma_T, [3 1 2]);

[x, y] = ndgrid(xl:hx:xr, yl:hy:yr);
r = zeros(I + 1, J + 1, 2);
r(:, :, 1) = x;
r(:, :, 2) = y;

ct = squeeze(ct);
st = squeeze(st);

x = squeeze(xl:hx:xr);
y = squeeze(yl:hy:yr);
w_angle = omega;

save test_random_kernel_0311.mat psi_label phi psi_bc rv_prime omega_prime sigma_a sigma_t ct st x y w_angle scattering_kernel
