function output = param_train()
N = 2; %2*N*(N+1) is the size of quadrature set
xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; % IxJ: the number of cells

N_itr = 2;

path = 'train_delta_1.mat';
bc_type == "delta_xv";

if bc_type == "delta_xv"
    func_psibc = @(x, y, x_c, y_c, var_x)(exp(- (y - y_c).^2/2 / var_x) * exp(- (x - x_c).^2/2 / var_x));
    func_psibc_v = @(v, v_c, var_v)(exp(- (v - v_c)).^2/2 / var_v);
    params_var_v = [0.02, 0.005];
    params_var_x = [0.01, 0.005];
    range_c_ind = [1, 40];
    params_func = {func_psibc, func_psibc_v, params_var_v, params_var_x, range_c_ind};
    
elseif bc_type == "delta_x"
    func_psibc = @(x, y, x_c, y_c, var_x)(exp(- (y - y_c).^2/2 / var_x) * exp(- (x - x_c).^2/2 / var_x));
    func_psibc_v = @(v)(1 + 0 .* v);
    params_var_v = [0.02, 0.005];
    params_var_x = [0.01, 0.005];
    range_c_ind = [1, 40];
    params_func = {func_psibc, func_psibc_v, params_var_v, params_var_x, range_c_ind};
end

omega_c = [0.4, 0.6, 0.4, 0.6]; % [l,r,b,t]
params_sigma_T = [10, 5]; %[outside, inside]
params_sigma_a = [5, 2]; %[outside, inside]
params_q = [0, 0]; %[outside, inside]

output = {[N I J xl xr yl yr N_itr], params_func, {omega_c, params_sigma_T, params_sigma_a, params_q}, {path}};
end
