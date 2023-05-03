% convert data and generate grid points

function [psi_label phi rv psi_bc rv_prime omega_prime sigma_a sigma_t r ct st omega x y w_angle] = convert_data(Input)

% Input = {[N I J xl xr yl yr M], {list_psiL, list_psiR, list_psiB, list_psiT},{omega, list_sigma_a, list_sigma_t, ct, st},{list_Phi, list_Psi}};

N = Input{1}(1); I = Input{1}(2); J = Input{1}(3); xl = Input{1}(4); xr = Input{1}(5); yl = Input{1}(6); yr = Input{1}(7); M = Input{1}(8);

list_psiL = Input{2}{1}; list_psiR = Input{2}{2}; list_psiB = Input{2}{3}; list_psiT = Input{2}{4};

omega = Input{3}{1}; list_sigma_a = Input{3}{2}; list_sigma_t = Input{3}{3}; ct = Input{3}{4}; st = Input{3}{5};

list_Phi = Input{4}{1}; list_Psi = Input{4}{2};

hx = (xr - xl) / I; hy = (yr - yl) / J;

    function rv_bc = gen_rvbc(x, y, ct, st, omega, len_xy, len_v)
        [x, y, vx_bc] = ndgrid(x, y, ct);
        [x, y, vy_bc] = ndgrid(x, y, st);
        [x, y, omega_bc] = ndgrid(x, y, omega);
        rv_bc = zeros(len_xy, len_v, 4);
        rv_bc(:, :, 1) = squeeze(x);
        rv_bc(:, :, 2) = squeeze(y);
        rv_bc(:, :, 3) = squeeze(vx_bc);
        rv_bc(:, :, 4) = squeeze(vy_bc);
    end

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
rv_l = gen_rvbc(xl, yl + 0.5 * hy:hy:yr - 0.5 * hy, ct(ct > 0), st(ct > 0), omega(ct > 0), J, 2 * M);

rv_r = gen_rvbc(xr, yl + 0.5 * hy:hy:yr - 0.5 * hy, ct(ct < 0), st(ct < 0), omega(ct < 0), J, 2 * M);

rv_b = gen_rvbc(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl, ct(st > 0), st(st > 0), omega(st > 0), I, 2 * M);

rv_t = gen_rvbc(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl, ct(st < 0), st(st < 0), omega(st < 0), I, 2 * M);

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
end
