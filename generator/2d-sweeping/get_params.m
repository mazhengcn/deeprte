function params = get_params(config)

params = config;

params.hx = (params.xr - params.xl) / params.I;
params.hy = (params.yr - params.yl) / params.J; % IxJ: the number of cells, hxxhy: size of cell
[params.omega, params.ct, params.st, params.M, params.theta, ~] = qnwlege2(params.N);

end