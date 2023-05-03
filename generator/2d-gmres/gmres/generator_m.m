function []=generator_m(N,xl,xr,yl,yr,I,J,N_itr,list_psiL,list_psiR,list_psiB,list_psiT)
%% 离散设置
tic;
hx=(xr-xl)/I; hy=(yr-yl)/J; % IxJ: the number of cells, hxxhy: size of cell
[omega,ct,st,M,theta]=qnwlege2(N);

%% 生成随机系数并构建入射函数
list_A=(rand(N_itr,4)-0.5)*20;
list_k=ceil((rand(N_itr,4))*50);
list_b=(rand(N_itr,2)-0.5)*0.02;
list_a=(rand(N_itr,2)-0.5)*0.4;
list_var = zeros(2,N_itr);
list_yhat = zeros(2,N_itr);

mesh_L_theta = [theta(3*M+1:4*M); theta(1:M)].*ones(1,J);
mesh_R_theta = theta(1*M+1:3*M).*ones(1,J);
mesh_B_theta = theta(0*M+1:2*M).*ones(1,I);
mesh_T_theta = theta(2*M+1:4*M).*ones(1,I);

for n = 1:N_itr
    variance=0.002*rand(1)+0.0001;
    temp=randperm(21);
    j_l=temp(1)+9;
    y_l=(j_l-0.5)*hy+yl;
    list_var(1,n)=variance;
    list_var(2,n)=variance;
    list_yhat(1,n)=y_l;
end
%% 指定散射截面，源项会发生变化的区域(Omega_C)
Omega_C=@(x,y) (x>=0.4).*(x<=0.6).*(y>=0.4).*(y<=0.6);
[Xc,Yc]=meshgrid(xl+0.5*hx:hx:xr-0.5*hx,yl+0.5*hy:hy:yr-0.5*hy);
[row,col]=find(Omega_C(Xc,Yc)>0);
LC=[row,col]; %Omega_C对应的网格集合

%% Omega_C外不变的散射截面，源项
f_varepsilon=@(x,y)1.*(x<=xr).*(y<=yr);
f_sigma_T=@(x,y)(5).*(x<=xr).*(y<=yr);
f_sigma_a=@(x,y)(1).*(x<=xr).*(y<=yr);
f_q=@(x,y)(0).*(x<=xr).*(y<=yr);

%% Omega_C内的散射截面，源项
g_varepsilon=cell(N_itr,1); g_sigma_T=g_varepsilon; g_sigma_a=g_varepsilon; g_q=g_varepsilon;
for n=1:N_itr
    g_varepsilon{n}=@(x,y)1*(x<=xr).*(y<=yr);
    g_sigma_T{n}=@(x,y)(1+list_a(n,1)).*(x<=xr).*(y<=yr);
    g_sigma_a{n}=@(x,y)(0.5+list_a(n,2)).*(x<=xr).*(y<=yr);
    g_q{n}=@(x,y)(0).*(x<=xr).*(y<=yr);
end

T_offline_part1=toc;
%% 运行主程序
Input={[N I J xl xr yl yr],{f_sigma_T,f_sigma_a,f_varepsilon,f_q,LC},{list_psiL,list_psiR,list_psiB,list_psiT,g_sigma_T,g_sigma_a,g_varepsilon,g_q}};
[list_psi_x,list_psi_y,list_alpha,list_Psi,list_Phi,list_varepsilon,list_sigma_T,list_sigma_a,list_q,...
    T_offline_part2,T_online_each] = run_main(Input);
T_offline=T_offline_part1+T_offline_part2
%% generate mat file
save test.mat list_A list_var list_yhat list_k list_psi_x list_psi_y list_alpha list_Phi list_Psi list_varepsilon list_sigma_T list_sigma_a list_q list_psiL list_psiR list_psiB list_psiT ct st omega theta M I J
save time.mat T_offline T_online_each
end
