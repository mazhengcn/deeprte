clear all
load train_delta_bc.mat

phi = list_Phi;
size_phi = size(phi);

N_itr = size_phi(3);

xl = 0; xr = 1; yl = 0; yr = 1; %[xl,xr]x[yl,yr] is the the computational domain
I = 40;
J = I; hx = (xr - xl) / I; hy = (yr - yl) / J; % IxJ: the number of cells, hxxhy: size of cell

for i = 1:N_itr
    c_ind = list_yhat(1, i);
    x_var = list_var(1, 1, i);
    v_var = list_var(2, 1, i);
    Phic = phi(:, :, i);
    min_value = min(min(Phic));
    surf(xl + 0.5 * hx:hx:xr - 0.5 * hx, yl + 0.5 * hy:hy:yr - 0.5 * hy, Phic);
    view(60, 30)
    xlabel('x')
    ylabel('y')
    zlabel('\phi(x,y)')
    strtitle = strcat('minphi=', num2str(min_value));
    title(strtitle)
    axis([0 1 0 1 0 1.1 * max(max(Phic))])
    strpath = strcat("/cluster/home/xuzhiqin_02/matlab/22-08-15/2d_delta/fig1/", 'c_ind=', num2str(c_ind), '_x_var', num2str(x_var), '_v_var', num2str(v_var), '.jpg');
    saveas(gcf, strpath)
    % saveas(gcf, ['/cluster/home/xuzhiqin_02/matlab/22-08-15/2d_delta/fig/', 'c_ind=', num2str(c_ind), '_x_var', num2str(x_var), '_v_var', num2str(v_var), '.jpg')];
end
