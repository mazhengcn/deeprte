function [T, maxerrPsi, maxerrPhi, phi_final, psi_final, sigma_T, sigma_a] = run_main(K, N, I, J, xl, xr, yl, yr, f_sigma_T, f_sigma_a, f_varepsilon, f_q, psiL, psiR, psiB, psiT, psiLB, psiLT, psiRB, psiRT)
    %% prepare disrectizaion
    [omega, ct, st, M, ~, ~] = qnwlege2(N); % generate quadrature set
    hx = (xr - xl) / I; hy = (yr - yl) / J;
    varepsilon = zeros(I, J); sigma_T = zeros(I, J); sigma_a = zeros(I, J); q = zeros(I, J);

    for i = 1:I + 1

        for j = 1:J + 1
            varepsilon(i, j) = f_varepsilon((i - 1) * hx, (j - 1) * hy);
            sigma_T(i, j) = f_sigma_T((i - 1) * hx, (j - 1) * hy);
            sigma_a(i, j) = f_sigma_a((i - 1) * hx, (j - 1) * hy);
            q(i, j) = f_q((i - 1) * hx, (j - 1) * hy);
        end

    end

    Sigma_T = sigma_T ./ varepsilon;
    Sigma_S = Sigma_T - varepsilon .* sigma_a;
    Q = varepsilon .* q;
    Psi_new = zeros(4 * M, I + 1, J + 1);
    Phi_new = zeros(4 * M, I + 1, J + 1);
    N_iter_max = 500;
    n = 1; maxerrPsi = zeros(N_iter_max, 1); maxerrPhi = zeros(N_iter_max, 1);
    maxerrPsi(1) = Inf; maxerrPhi(1) = Inf;

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
                    ./ (ct(1:M) / hx + st(1:M) / hy + Sigma_T(i, j));
            end

        end

        for i = I:-1:1

            for j = 2:J + 1
                Psi_new(M + 1:2 * M, i, j) = (-ct(M + 1:2 * M) .* Psi_new(M + 1:2 * M, i + 1, j) / hx + st(M + 1:2 * M) .* Psi_new(M + 1:2 * M, i, j - 1) / hy + Sigma_S(i, j) * Phi_old(M + 1:2 * M, i, j) + Q(i, j)) ...
                    ./ (-ct(M + 1:2 * M) / hx + st(M + 1:2 * M) / hy + Sigma_T(i, j));
            end

        end

        for i = I:-1:1

            for j = J:-1:1
                Psi_new(2 * M + 1:3 * M, i, j) = (-ct(2 * M + 1:3 * M) .* Psi_new(2 * M + 1:3 * M, i + 1, j) / hx - st(2 * M + 1:3 * M) .* Psi_new(2 * M + 1:3 * M, i, j + 1) / hy + Sigma_S(i, j) * Phi_old(2 * M + 1:3 * M, i, j) + Q(i, j)) ...
                    ./ (-ct(2 * M + 1:3 * M) / hx - st(2 * M + 1:3 * M) / hy + Sigma_T(i, j));
            end

        end

        for i = 2:I + 1

            for j = J:-1:1
                Psi_new(3 * M + 1:end, i, j) = (ct(3 * M + 1:end) .* Psi_new(3 * M + 1:end, i - 1, j) / hx - st(3 * M + 1:end) .* Psi_new(3 * M + 1:end, i, j + 1) / hy + Sigma_S(i, j) * Phi_old(3 * M + 1:end, i, j) + Q(i, j)) ...
                    ./ (ct(3 * M + 1:end) / hx - st(3 * M + 1:end) / hy + Sigma_T(i, j));
            end

        end

        for i = 1:I + 1

            for j = 1:J + 1
                Phi_new(:, i, j) = K * (omega .* Psi_new(:, i, j));
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

    phi_final = Phi_final;
    psi_final = Psi_new;

    surf(xl:hx:xr, yl:hy:yr, Phi_final);
    view(60, 30)
    xlabel('x')
    ylabel('y')
    zlabel('\phi(x,y)')
    axis([0 1 0 1 0 1.1 * max(max(Phi_final))])
    shading interp
    saveas(gcf, 'save_1.jpg')
end
