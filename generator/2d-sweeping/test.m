clear all
clc

tic
%% discretization setting
N = 3; %2*N*(N+1) is the size of quadrature set
xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell
[omega, ct, st, M, theta, ~] = qnwlege2(N);

% list_psiL = zeros(2 * M, J, N_itr); list_psiR = list_psiL;
% list_psiB = zeros(2 * M, I, N_itr); list_psiT = list_psiB;
%% cross sections, external source term and boundary conditions
f_varepsilon = @(x, y)1 .* (x <= xr) .* (y <= yr);
f_sigma_T = @(x, y)(10) .* (x <= xr) .* (y <= yr);
f_sigma_a = @(x, y)(5) .* (x <= xr) .* (y <= yr);
f_q = @(x, y)(0) .* (x <= xr) .* (y <= yr);
% y_l = 0.5; variance = 1/100;
% func_psiL = @(y)(exp(- (y - y_l).^2/2 / variance));
% func_psiL = @(y)(sin(pi * y));
% psiL = ones(2 * M, 1) * func_psiL(yl + hy:hy:yr - hy);
psiL = ones(2 * M, J - 1); % i=I+1, j=2:J, m=M+1:4*M
psiR = zeros(2 * M, J - 1); % i=I+1, j=2:J, m=M+1:4*M
psiB = zeros(2 * M, I - 1); % i=2:I, j=1,   m=1:2*M
psiT = zeros(2 * M, I - 1); % i=2:I, j=J+1, m=2*M+1:4*M
psiLB = zeros(M, 1); % i=1,   j=1,   m=1:M
psiLT = zeros(M, 1); % i=1,   j=J+1, m=3*M+1:4*M
psiRB = zeros(M, 1); % i=I+1, j=1,   m=M+1:2*M
psiRT = zeros(M, 1); % i=I+1, j=J+1, m=2*M+1:3*M

%% scattering kernel
% K=ones(4*M);

g = 0; %anisotropic coefficient
K = P2generator(N, g); %Kernel matrix
% K = 1 + zeros(24, 24); %Kernel matrix; %Kernel matrix
%% run main
[T, maxerrPsi, maxerrPhi, phi_final, psi_final, ~, ~] = run_main(K, N, I, J, xl, xr, yl, yr, f_sigma_T, f_sigma_a, f_varepsilon, f_q, psiL, psiR, psiB, psiT, psiLB, psiLT, psiRB, psiRT);

psi_final = permute(psi_final, [2 3 1]);
save test.mat phi_final, psi_final
