function config = config()

config.N = 3; %2*N*(N+1) is the size of quadrature set
config.xl = 0;
config.xr = 1;
config.yl = 0;
config.yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
config.I = 40;
config.J = 40;

% number of iteration
config.N_itr = 2;

config.generate_train_data = false;

% sigma region
config.regionx_sigma_a = [0.4, 0.6];
config.regionx_sigma_t = [0.4, 0.6];
config.regiony_sigma_a = [0.4, 0.6];
config.regiony_sigma_t = [0.4, 0.6];

config.in_sigma_a_scope = [1, 3];
config.in_sigma_t_scope = [5, 6];

config.out_sigma_a_scope = [5, 5];
config.out_sigma_t_scope = [10, 10];

% scattering kernel
config.g_scope = [0.0, 0.2];

% boundary config
config.var_x_scope = [0.02, 0.025];
config.var_v_scope = [0.01, 0.015];

config.r_ind_scope = [2, config.I];
config.v_ind_scope = [1, config.N*(config.N+1)];


end