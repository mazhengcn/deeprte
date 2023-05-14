config = config();
params = get_params(config);

rng(params.rng_seed);

N_itr = params.N_itr;
M = params.M; J = params.J; I = params.I;

list_psiL = zeros(2 * M, J + 1, N_itr); list_psiR = list_psiL;
list_psiB = zeros(2 * M, I + 1, N_itr); list_psiT = list_psiB;

list_Psi = zeros(4 * M, I + 1, J + 1, N_itr);
list_Phi = zeros(I + 1, J + 1, N_itr);
list_sigma_a = zeros(I + 1, J + 1, N_itr);
list_sigma_t = zeros(I + 1, J + 1, N_itr);

list_scattering_kernel = zeros(N_itr, 4 * M, 4 * M);

list_rand_params = cell(1, N_itr);

for n = 1:N_itr
    
    [scattering_kernel, boundary, sigma, q, varepsilon, rand_params] = gen_inputs(params);
    
    [T, maxerrPsi, maxerrPhi, phi, psi] = sweeping_solver(scattering_kernel, boundary, sigma, q, varepsilon, params);
    
    list_psiL(:,:,n) = boundary.psiL;
    list_psiR(:,:,n) = boundary.psiR;
    list_psiB(:,:,n) = boundary.psiB;
    list_psiT(:,:,n) = boundary.psiT;
    
    list_Psi(:, :, :, n) = psi;
    list_Phi(:, :, n) = phi;
    list_sigma_a(:, :, n) = sigma.sigma_a;
    list_sigma_t(:, :, n) = sigma.sigma_t;
    
    list_scattering_kernel(n, :, :) = scattering_kernel;
    
    list_rand_params{n} = rand_params;
    
end

boundary.psiL = permute(list_psiL, [3 2 1]);
boundary.psiR = permute(list_psiR, [3 2 1]);
boundary.psiB = permute(list_psiB, [3 2 1]);
boundary.psiT = permute(list_psiT, [3 2 1]);

sigma_a = permute(list_sigma_a, [3 1 2]);
sigma_t = permute(list_sigma_t, [3 1 2]);

scattering_kernel = list_scattering_kernel;

psi_label = permute(list_Psi, [4 2 3 1]);
phi = permute(list_Phi, [3 1 2]);

[psi_bc, rv_prime, omega_prime] = postprocess(params, boundary);

w_angle = squeeze(params.omega);
x = squeeze(params.xl:params.hx:params.xr);
y = squeeze(params.yl:params.hy:params.yr);
ct = squeeze(params.ct);
st = squeeze(params.st);

rand_params = list_rand_params;

split_path = strsplit(config.save_dir,'/');
file_name = append(split_path{end}, '.mat');
if ~exist(params.save_dir,'dir')
    mkdir(params.save_dir);
end
save(append(params.save_dir, '/', file_name), 'psi_label', 'phi', 'psi_bc', 'rv_prime', 'omega_prime', 'sigma_a', 'sigma_t', 'ct', 'st', 'x', 'y', 'w_angle', 'scattering_kernel', 'rand_params', 'config', '-nocompression')
copyfile('config.m', params.save_dir)