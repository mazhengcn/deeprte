function [T, maxerrPsi, maxerrPhi, phi, psi, sigma] = sweeping_solver(scattering_kernel, boundary, sigma, q, varepsilon, params)

M = params.M; J = params.J; I = params.I;
hy = params.hy; hx = params.hx;
ct = params.ct; st = params.st;
omega = params.omega;

%% prepare disrectizaion
sigma_t = sigma.sigma_t; sigma_a = sigma.sigma_a;
Sigma_t = sigma_t ./ varepsilon;
Sigma_S = Sigma_t - varepsilon .* sigma_a;
Q = varepsilon .* q;

Psi_new = zeros(4 * M, I + 1, J + 1);
Phi_new = zeros(4 * M, I + 1, J + 1);

N_iter_max = 500;

n = 1; maxerrPsi = zeros(N_iter_max, 1); maxerrPhi = zeros(N_iter_max, 1);
maxerrPsi(1) = Inf; maxerrPhi(1) = Inf;

psiL = boundary.psiL(:, 2:40);
psiR = boundary.psiR(:, 2:40); % i=I+1, j=2:J, m=M+1:4*M
psiB = boundary.psiB(:, 2:40); % i=2:I, j=1,   m=1:2*M
psiT = boundary.psiT(:, 2:40); % i=2:I, j=J+1, m=2*M+1:4*M

psiLB = boundary.psiL(M + 1:2 * M, 1); % i=1,   j=1,   m=1:M
psiLT = boundary.psiL(1:M, I + 1); % i=1,   j=J+1, m=3*M+1:4*M
psiRB = boundary.psiR(1:M, 1); % i=I+1, j=1,   m=M+1:2*M
psiRT = boundary.psiR(M + 1:2 * M, 1); % i=I+1, j=J+1, m=2*M+1:3*M

while n < N_iter_max && maxerrPsi(n) > 1e-5
    Psi_old = Psi_new;
    Phi_old = Phi_new;
    %% Impose boundary conditions
    Psi_new = zeros(4 * M, I + 1, J + 1);
    Psi_new(1:M, 1, 2:J) = psiL(M + 1:end, :);
    Psi_new(3 * M + 1:end, 1, 2:J) = psiL(1:M, :);
    Psi_new(M + 1:3 * M, end, 2:J) = psiR;
    Psi_new(1:2 * M, 2:I, 1) = psiB;
    Psi_new(2 * M + 1:end, 2:I, end) = psiT;
    Psi_new(1:M, 1, 1) = psiLB;
    Psi_new(3 * M + 1:end, 1, end) = psiLT;
    Psi_new(M + 1:2 * M, end, 1) = psiRB;
    Psi_new(2 * M + 1:3 * M, end, end) = psiRT;
    %% run one step
    for i = 2:I + 1
        
        for j = 2:J + 1
            Psi_new(1:M, i, j) = (ct(1:M) .* Psi_new(1:M, i - 1, j) / hx + st(1:M) .* Psi_new(1:M, i, j - 1) / hy + Sigma_S(i, j) * Phi_old(1:M, i, j) + Q(i, j)) ...
                ./ (ct(1:M) / hx + st(1:M) / hy + Sigma_t(i, j));
        end
        
    end
    
    for i = I:-1:1
        
        for j = 2:J + 1
            Psi_new(M + 1:2 * M, i, j) = (-ct(M + 1:2 * M) .* Psi_new(M + 1:2 * M, i + 1, j) / hx + st(M + 1:2 * M) .* Psi_new(M + 1:2 * M, i, j - 1) / hy + Sigma_S(i, j) * Phi_old(M + 1:2 * M, i, j) + Q(i, j)) ...
                ./ (-ct(M + 1:2 * M) / hx + st(M + 1:2 * M) / hy + Sigma_t(i, j));
        end
        
    end
    
    for i = I:-1:1
        
        for j = J:-1:1
            Psi_new(2 * M + 1:3 * M, i, j) = (-ct(2 * M + 1:3 * M) .* Psi_new(2 * M + 1:3 * M, i + 1, j) / hx - st(2 * M + 1:3 * M) .* Psi_new(2 * M + 1:3 * M, i, j + 1) / hy + Sigma_S(i, j) * Phi_old(2 * M + 1:3 * M, i, j) + Q(i, j)) ...
                ./ (-ct(2 * M + 1:3 * M) / hx - st(2 * M + 1:3 * M) / hy + Sigma_t(i, j));
        end
        
    end
    
    for i = 2:I + 1
        
        for j = J:-1:1
            Psi_new(3 * M + 1:end, i, j) = (ct(3 * M + 1:end) .* Psi_new(3 * M + 1:end, i - 1, j) / hx - st(3 * M + 1:end) .* Psi_new(3 * M + 1:end, i, j + 1) / hy + Sigma_S(i, j) * Phi_old(3 * M + 1:end, i, j) + Q(i, j)) ...
                ./ (ct(3 * M + 1:end) / hx - st(3 * M + 1:end) / hy + Sigma_t(i, j));
        end
        
    end
    
    for i = 1:I + 1
        
        for j = 1:J + 1
            Phi_new(:, i, j) = scattering_kernel * (omega .* Psi_new(:, i, j));
        end
        
    end
    
    %% compute difference between two steps
    maxerrPsi(n + 1) = max(max(max(abs(Psi_new - Psi_old))));
    maxerrPhi(n + 1) = max(max(max(abs(Phi_new - Phi_old))));
    n = n + 1;
end

T = toc;
maxerrPsi = maxerrPsi(1:n); maxerrPhi = maxerrPhi(1:n);
%% plot (scalar flux)
Phi_final = zeros(I + 1, J + 1);

for i = 1:I + 1
    
    for j = 1:J + 1
        Phi_final(i, j) = omega' * Psi_new(:, i, j);
    end
    
end

phi = Phi_final;
psi = Psi_new;

% x = squeeze(params.xl:params.hx:params.xr);
% y = squeeze(params.yl:params.hy:params.yr);
% surf(x, y, Phi_final);
% view(60, 30)
% xlabel('x')
% ylabel('y')
% zlabel('\phi(x,y)')
% axis([0 1 0 1 0 1.1 * max(max(Phi_final))])
% shading interp
% saveas(gcf, 'save_2.jpg')
end
