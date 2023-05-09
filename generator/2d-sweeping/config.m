function config = config()

config.N = 3; %2*N*(N+1) is the size of quadrature set
config.xl = 0;
config.xr = 1;
config.yl = 0;
config.yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
config.I = 40;
config.J = 40;

% general settings
config.generate_train_data = true;
config.save_path = '/workspaces/deeprte/generator/train_kernel_g0.4-0.6.mat';
rng('shuffle');
config.rng = rng;

% number of iteration
config.N_itr = 2;

% sigma region
config.regionx_sigma_a = [0.4, 0.6];
config.regionx_sigma_t = [0.4, 0.6];
config.regiony_sigma_a = [0.4, 0.6];
config.regiony_sigma_t = [0.4, 0.6];

config.in_sigma_a_scope = [2, 4];
config.in_sigma_t_scope = [5, 7];

config.out_sigma_a_scope = [5, 5];
config.out_sigma_t_scope = [10, 10];

% scattering kernel
config.g_scope = [0.4, 0.6];

% boundary config
config.var_x_scope = [0.005, 0.02];
config.var_v_scope = [0.005, 0.01];

config.r_ind_scope = [2, config.I];
config.v_ind_scope = [1, config.N*(config.N+1)];

end
