function output = param_test()
N = 2; %2*N*(N+1) is the size of quadrature set
xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; % IxJ: the number of cells

N_itr = 2;

path = 'test_sin.mat';

func_bc_psiL = @(x, y, arg)(arg(1) * sin(arg(2) * pi * y) + 10);
func_bc_psiL_v = @(x, arg_v)(1 + 0 .* x);
func_bc_psiR = @(x, y, arg)(arg(1) * sin(arg(2) * pi * y) + 10);
func_bc_psiR_v = @(x, arg_v)(1 + 0 .* x);
func_bc_psiB = @(x, y, arg)(arg(1) * sin(arg(2) * pi * x) + 10);
func_bc_psiB_v = @(x, arg_v)(1 + 0 .* x);
func_bc_psiT = @(x, y, arg)(arg(1) * sin(arg(2) * pi * x) + 10);
func_bc_psiT_v = @(x, arg_v)(1 + 0 .* x);

list_A = (rand(N_itr, 4) - 0.5) * 20;
list_k = ceil((rand(N_itr, 4)) * 50);

arg_x = zeros(N_itr, 4, 2);
arg_v = zeros(N_itr, 4, 2);
arg_x(:, :, 1) = list_A;
arg_x(:, :, 2) = list_k;

func_bc_list_x = {func_bc_psiL, func_bc_psiR, func_bc_psiB, func_bc_psiT};
func_bc_list_v = {func_bc_psiL_v, func_bc_psiR_v, func_bc_psiB_v, func_bc_psiT_v};

params_func = {func_bc_list_x, func_bc_list_v, arg_x, arg_v};

omega_c = [0.4, 0.6, 0.4, 0.6]; % [l,r,b,t]
params_sigma_T = [10, 5]; %[outside, inside]
params_sigma_a = [5, 2]; %[outside, inside]
params_q = [0, 0]; %[outside, inside]

output = {[N I J xl xr yl yr N_itr], params_func, {omega_c, params_sigma_T, params_sigma_a, params_q}, {path}};
end
