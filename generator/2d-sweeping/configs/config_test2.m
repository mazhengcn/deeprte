function config = config()

config.N = 3; %2*N*(N+1) is the size of quadrature set
config.xl = 0;
config.xr = 1;
config.yl = 0;
config.yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
config.I = 40;
config.J = 40;

% general settings
config.save_dir = '/workspaces/deeprte/generator/data/raw_data/train/g0.1';
config.rng_seed = 143907;

% number of iteration
config.N_itr = 100;

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
config.g_scope = [0.7, 0.9];

% boundary config
config.generate_train_data = false;
config.test_bc_type = "sin_rv";

config.amplitude_r_scope = [-5, 5];
config.wavenumber_r_scope = [-10, 10];

config.amplitude_v_scope = [-1, 1];
config.wavenumber_v_scope = [-6, 6];

% config.amplitude_x_scope = [-5, 5];
% config.wavenumber_x_scope = [-10, 10];

end
