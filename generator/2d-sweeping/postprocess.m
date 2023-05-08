function [psi_bc, rv_prime, omega_prime] = postprocess(params, boundary)

M = params.M; I = params.I;
yl = params.yl; xl = params.xl; yr = params.yr; xr = params.xr;
hy = params.hy; hx = params.hx;
ct = params.ct;
st = params.st;
omega = squeeze(params.omega);

psi_bc = cat(2, boundary.psiL, boundary.psiR, boundary.psiB, boundary.psiT);
% psil
[rv_l, omega_l] = get_bc_coords(xl, yl:hy:yr, [ct(3 * M + 1:4 * M); ct(1:M)], [st(3 * M + 1:4 * M); st(1:M)], [omega(3 * M + 1:4 * M); omega(1:M)]);
[rv_r, omega_r] = get_bc_coords(xr, yl:hy:yr, ct(ct < 0), st(ct < 0), omega(ct < 0));
[rv_b, omega_b] = get_bc_coords(xl:hx:xr, yl, ct(st > 0), st(st > 0), omega(st > 0));
[rv_t, omega_t] = get_bc_coords(xl:hx:xr, yr, ct(st < 0), st(st < 0), omega(st < 0));

rv_prime = cat(1, rv_l, rv_r, rv_b, rv_t);
omega_prime = cat(1, squeeze(omega_l), squeeze(omega_r), squeeze(omega_b), squeeze(omega_t));
omega_prime = omega_prime / I;

end

function [coords, omega_bc] = get_bc_coords(x, y, ct, st, omega)

[~, ~, vx_bc] = ndgrid(x, y, ct);
[~, ~, vy_bc] = ndgrid(x, y, st);
[x_bc, y_bc, omega_bc] = ndgrid(x, y, omega);
% coords = zeros(J + 1, 2 * M, 4);
coords(:, :, 1) = squeeze(x_bc);
coords(:, :, 2) = squeeze(y_bc);
coords(:, :, 3) = squeeze(vx_bc);
coords(:, :, 4) = squeeze(vy_bc);

end