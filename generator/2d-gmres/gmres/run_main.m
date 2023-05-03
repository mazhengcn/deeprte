function [list_psi_x, list_psi_y, list_alpha, list_Psi, list_Phi, list_varepsilon, list_sigma_T, list_sigma_a, list_q, ...
    T_offline_part2, T_online_each] = run_main(Input)
%% nested functions
    function [ve, va] = eigensx(i, j)
        [ve, va] = eig(diag(ct.^ - 1) * ((sigma_T(i, j) - varepsilon(i, j)^2 * sigma_a(i, j)) * ones(4 * M) * diag(omega) - sigma_T(i, j) * eye(4 * M)));
        va = diag(va);
        % calculate eigenpairs of B
        if sum((abs(imag(va)) > 0)) > 0
            
            if sigma_a(i) < 10^ - 12
                
                if sum((abs(imag(va)) > 0)) == 2
                    va(abs(imag(va)) > 0) = 0;
                    % when sigma_a=0, 0 is a double eigenvalue of B with some eigenvectors we do not care about
                else
                    error('complex eigenvalues!');
                end
                
            else
                error('complex eigenvalues!');
            end
            
        end
        
        %Be careful! In some case, may generate complex eigenvalues
        [va, order] = sort(va); % sort eigenvalues
        ve = ve(:, order); % sort eigenvector matrix to be consistent with eigenvalue vector.
        ve = ve * diag(1 ./ (max(abs(ve)))); % rescale eigenvector such that omega'*eigenvector=1
    end

    function [ve, va] = eigensy(i, j)
        [ve, va] = eig(diag(st.^ - 1) * ((sigma_T(i, j) - varepsilon(i, j)^2 * sigma_a(i, j)) * ones(4 * M) * diag(omega) - sigma_T(i, j) * eye(4 * M)));
        va = diag(va);
        % calculate eigenpairs of B
        if sum((abs(imag(va)) > 0)) > 0
            
            if sigma_a(i) < 10^ - 12
                
                if sum((abs(imag(va)) > 0)) == 2
                    va(abs(imag(va)) > 0) = 0;
                    % when sigma_a=0, 0 is a double eigenvalue of B with some eigenvectors we do not care about
                else
                    error('complex eigenvalues!');
                end
                
            else
                error('complex eigenvalues!');
            end
            
        end
        
        %Be careful! In some case, may generate complex eigenvalues
        [va, order] = sort(va); % sort eigenvalues
        ve = ve(:, order); % sort eigenvector matrix to be consistent with eigenvalue vector.
        ve = ve * diag(1 ./ (max(abs(ve)))); % rescale eigenvector such that omega'*eigenvector=1
    end

    function matrix = fsm(x, y, i, j)
        matrix = zeros(4 * M, 8 * M);
        [Vx, Dx] = eigensx(i, j); x_0 = (i - 0.5) * hx + xl;
        [Vy, Dy] = eigensy(i, j); y_0 = (j - 0.5) * hy + yl;
        
        if sigma_a(i, j) > 10^ - 8
            matrix(:, 1:2 * M) = Vx(:, 1:2 * M) * diag(exp((Dx(1:2 * M) * (x - x_0 + 0.5 * hx)) / varepsilon(i, j)));
            matrix(:, 2 * M + 1:4 * M) = Vx(:, 2 * M + 1:4 * M) * diag(exp((Dx(2 * M + 1:4 * M) * (x - x_0 - 0.5 * hx)) / varepsilon(i, j)));
            matrix(:, 4 * M + 1:6 * M) = Vy(:, 1:2 * M) * diag(exp((Dy(1:2 * M) * (y - y_0 + 0.5 * hy)) / varepsilon(i, j)));
            matrix(:, 6 * M + 1:8 * M) = Vy(:, 2 * M + 1:4 * M) * diag(exp((Dy(2 * M + 1:4 * M) * (y - y_0 - 0.5 * hy)) / varepsilon(i, j)));
            % generate fundamental solution matrix when sigma_a\neq 0
        else
            matrix(:, 1:2 * M - 1) = Vx(:, 1:2 * M - 1) * diag(exp((Dx(1:2 * M - 1) * (x - x_0 + 0.5 * hx)) / varepsilon(i, j)));
            matrix(:, 2 * M) = ones(4 * M, 1);
            matrix(:, 2 * M + 1) = x - varepsilon(i, j) * ct / sigma_T(i, j);
            matrix(:, 2 * M + 2:4 * M) = Vx(:, 2 * M + 2:4 * M) * diag(exp((Dx(2 * M + 2:4 * M) * (x - x_0 - 0.5 * hx)) / varepsilon(i, j)));
            matrix(:, 4 * M + 1:6 * M - 1) = Vy(:, 1:2 * M - 1) * diag(exp((Dy(1:2 * M - 1) * (y - y_0 + 0.5 * hy)) / varepsilon(i, j)));
            matrix(:, 6 * M) = y - varepsilon(i, j) * st / sigma_T(i, j);
            matrix(:, 6 * M + 1) = x * y - varepsilon(i, j) * x * st / sigma_T(i, j) - varepsilon(i, j) * y * ct / sigma_T(i, j) + 2 * varepsilon(i, j)^2 * (ct .* st) / sigma_T(i, j)^2;
            matrix(:, 6 * M + 2:8 * M) = Vy(:, 2 * M + 2:4 * M) * diag(exp((Dy(2 * M + 2:4 * M) * (y - y_0 - 0.5 * hy)) / varepsilon(i, j)));
            % generate fundamental solution matrix when sigma_a= 0
        end
        
    end

    function Psi0 = Psi0(x, y, i, j)
        %special solution vector at (i,j)th cell with position (x,y)
        if sigma_a(i, j) > 10^ - 8
            Psi0 = q(i, j) * ones(4 * M, 1) / sigma_a(i, j);
        else
            qt = q(i, j); % suppose source term is isotropic
            Psi0 = -3/8 * sigma_T(i, j) * qt * (x^2 + y^2) + 3/4 * varepsilon(i, j) * qt * (ct * x + st * y);
            Psi0 = Psi0 - 3 * varepsilon(i, j)^2 * qt / 2 / sigma_T(i, j) * (ct .* ct + st .* st) + varepsilon(i, j)^2 / sigma_T(i, j) * qt;
        end
        
    end

%% Prepare parameters and matrices
tic
N = Input{1}(1); I = Input{1}(2); J = Input{1}(3); xl = Input{1}(4); xr = Input{1}(5); yl = Input{1}(6); yr = Input{1}(7);
f_sigma_T = Input{2}{1}; f_sigma_a = Input{2}{2}; f_varepsilon = Input{2}{3}; f_q = Input{2}{4}; LC = Input{2}{5};
% ����offline���ֵ�����

[omega, ct, st, M] = qnwlege2(N); % generate quadrature set
hx = (xr - xl) / I; hy = (yr - yl) / J;
varepsilon = zeros(I, J); sigma_T = zeros(I, J); sigma_a = zeros(I, J); q = zeros(I, J);
fsml = zeros(4 * M, 8 * M, I, J); fsmr = fsml; fsmb = fsml; fsmt = fsml; fsmc = fsml;
Psi0l = zeros(4 * M, I, J); Psi0r = Psi0l; Psi0b = Psi0l; Psi0t = Psi0l; Psi0c = Psi0l;

for i = 1:I
    
    for j = 1:J
        varepsilon(i, j) = integral2(f_varepsilon, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        sigma_T(i, j) = integral2(f_sigma_T, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        sigma_a(i, j) = integral2(f_sigma_a, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        q(i, j) = integral2(f_q, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        fsml(:, :, i, j) = fsm((i - 1) * hx + xl, (j - 0.5) * hy + yl, i, j);
        fsmr(:, :, i, j) = fsm(i * hx + xl, (j - 0.5) * hy + yl, i, j);
        fsmb(:, :, i, j) = fsm((i - 0.5) * hx + xl, (j - 1) * hy + yl, i, j);
        fsmt(:, :, i, j) = fsm((i - 0.5) * hx + xl, j * hy + yl, i, j);
        fsmc(:, :, i, j) = fsm((i - 0.5) * hx + xl, (j - 0.5) * hy + yl, i, j);
        %fundamental solution matrices at the left/right/bottom/top edge midpoint of each cell
        Psi0l(:, i, j) = Psi0((i - 1) * hx + xl, (j - 0.5) * hy + yl, i, j);
        Psi0r(:, i, j) = Psi0(i * hx + xl, (j - 0.5) * hy + yl, i, j);
        Psi0b(:, i, j) = Psi0((i - 0.5) * hx + xl, (j - 1) * hy + yl, i, j);
        Psi0t(:, i, j) = Psi0((i - 0.5) * hx + xl, j * hy + yl, i, j);
        Psi0c(:, i, j) = Psi0((i - 0.5) * hx + xl, (j - 0.5) * hy + yl, i, j);
        %special solution vectors at the left/right/bottom/top edge midpoint of each cell
    end
    
end

%% Offline stage
[L, S, l] = layer_generate(LC, I, J);
MB = cell(S, 1); MD = MB; Mp = MB; Mm = MB; MBo = MB;
bD = MB; bp_temp = MB; bp = MB; bm_temp = MB; bm = MB; bBo = MB;
Xp = MB; Zp = MB; Xm = MB; Zm = MB;
% tX=MB; tW=MB; tYbDs=MB;
% LZ=MB; UZ=MB; PZ=MB;
LM = MB; UM = MB; PM = MB;
LBo = zeros(2 * (I + J), S, 2);
% ��ȷ������λ�õľ���,(m,s,1(1,2,3,4��Ӧlrbt)/2(i/j��Ӧλ��))
for s = 2:S
    lms = 8 * M * l(s);
    lB = sum(((L(:, 3, s) == 0) .* (L(:, 1, s) == 1) + (L(:, 4, s) == 0) .* (L(:, 1, s) == I) + (L(:, 5, s) == 0) .* (L(:, 2, s) == 1) + (L(:, 6, s) == 0) .* (L(:, 2, s) == J)) .* (L(:, 1, s) ~= 0));
    lD = sum(((L(:, 3, s) == 0) .* (L(:, 1, s) ~= 1) + (L(:, 4, s) == 0) .* (L(:, 1, s) ~= I) + (L(:, 5, s) == 0) .* (L(:, 2, s) ~= 1) + (L(:, 6, s) == 0) .* (L(:, 2, s) ~= J)) .* (L(:, 1, s) ~= 0));
    Bs = zeros(2 * M * lB, lms); nB = 0;
    Bos = Bs; bBos = zeros(2 * M * lB, 1);
    Ds = zeros(2 * M * lD, lms); bDs = zeros(2 * M * lD, 1); nD = 0;
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == 0
            
            if i > 1
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsml(3 * M + 1:4 * M, :, i, j); fsml(1:M, :, i, j)]; % incident part
                k = L(m, 7, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) =- [fsmr(3 * M + 1:4 * M, :, i - 1, j); fsmr(1:M, :, i - 1, j)]; % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) =- [Psi0l(3 * M + 1:4 * M, i, j); Psi0l(1:M, i, j)] + [Psi0r(3 * M + 1:4 * M, i - 1, j); Psi0r(1:M, i - 1, j)];
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsml(3 * M + 1:4 * M, :, i, j); fsml(1:M, :, i, j)]; % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsml(M + 1:3 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0l(M + 1:3 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [1; j];
            end
            
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == 0
            
            if i < I
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(M + 1:3 * M, :, i, j); % incident part
                k = L(m, 8, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsml(M + 1:3 * M, :, i + 1, j); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0r(M + 1:3 * M, i, j) + Psi0l(M + 1:3 * M, i + 1, j);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(M + 1:3 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsmr(3 * M + 1:4 * M, :, i, j); fsmr(1:M, :, i, j)]; % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = [Psi0r(3 * M + 1:4 * M, i, j); Psi0r(1:M, i, j)];
                nB = nB + 1;
                LBo(nB, s, :) = [2; j];
            end
            
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == 0
            
            if j > 1
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(1:2 * M, :, i, j); % incident part
                k = L(m, 9, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsmt(1:2 * M, :, i, j - 1); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0b(1:2 * M, i, j) + Psi0t(1:2 * M, i, j - 1);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(1:2 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(2 * M + 1:4 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0b(2 * M + 1:4 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [3; i];
            end
            
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == 0
            
            if j < I
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(2 * M + 1:4 * M, :, i, j); % incident part
                k = L(m, 10, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsmb(2 * M + 1:4 * M, :, i, j + 1); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0t(2 * M + 1:4 * M, i, j) + Psi0b(2 * M + 1:4 * M, i, j + 1);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(2 * M + 1:4 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(1:2 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0t(1:2 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [4; i];
            end
            
        end
        
        % check relationship with (i,j-1)
    end
    
    MD{s} = Ds; bD{s} = bDs;
    MB{s} = Bs; MBo{s} = Bos; bBo{s} = bBos;
end

% formulate B_s, \tB_s, D_s, and corresponding vectors for s=2:S
% ������b_s^B��ΪҪ�ñ߽�����ʱҲ��Ҫ���¶�һ��λ��
% ������b_s^tB��Ϊ�����ʱֱ��tB_s*beta_s+b_s^tB

Mp{S} = zeros(0, 8 * M * l(S)); bp_temp{S} = zeros(0, 1);

for s = S:-1:2
    lms = 8 * M * l(s);
    lm = sum(((L(:, 3, s) == -1) + (L(:, 4, s) == -1) + (L(:, 5, s) == -1) + (L(:, 6, s) == -1)) .* (L(:, 1, s) ~= 0));
    Ams = zeros(4 * M * lm, lms); Aps = zeros(4 * M * lm, 8 * M * l(s - 1)); nAm = 0; % prepare storage space and counter for A_s^- and A_{s-1}^+
    bAs = zeros(4 * M * lm, 1);
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == -1
            Ams(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsml(:, :, i, j); % A_s^-
            
            if s > 2
                k = L(m, 7, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmr(:, :, i - 1, j); % A_{s-1}^+
                bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0l(:, i, j) + Psi0r(:, i - 1, j);
            end
            
            nAm = nAm + 1;
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == -1
            Ams(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(:, :, i, j); % A_s^-
            
            if s > 2
                k = L(m, 8, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsml(:, :, i + 1, j); % A_{s-1}^+
                bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0r(:, i, j) + Psi0l(:, i + 1, j);
            end
            
            nAm = nAm + 1;
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == -1
            Ams(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(:, :, i, j); % A_s^-
            
            if s > 2
                k = L(m, 9, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmt(:, :, i, j - 1); % A_{s-1}^+
                bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0b(:, i, j) + Psi0t(:, i, j - 1);
            end
            
            nAm = nAm + 1;
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == -1
            Ams(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(:, :, i, j); % A_s^-
            
            if s > 2
                k = L(m, 10, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmb(:, :, i, j + 1); % A_{s-1}^+
                bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0t(:, i, j) + Psi0b(:, i, j + 1);
            end
            
            nAm = nAm + 1;
        end
        
        % check relationship with (i,j-1)
    end
    
    [Xp{s}, Yp, Wp, Zp{s}] = form_XYWZ(MB{s}, MD{s}, Ams, Mp{s});
    %�����case����Ϊ���еĲ���offlineʵ���϶�Ҫ�����������˲���H�Ĵ�X,Z
    if s > 2
        Mp{s - 1} = Wp * Aps;
        bp_temp{s - 1} = -Yp * bD{s} - Wp * bAs - Zp{s} * bp_temp{s};
    else
        W2p = Wp; %����W_2^+������online����M_1^+��b_1^+
        bp_temp{s - 1} = -Yp * bD{s} - Zp{s} * bp_temp{s};
    end
    
end

%compute M_s^+ bb_s^{0,+}

s = 2; lm = sum(((L(:, 3, s) == -1) + (L(:, 4, s) == -1) + (L(:, 5, s) == -1) + (L(:, 6, s) == -1)) .* (L(:, 1, s) ~= 0));
Mm2 = zeros(2 * M * lm, lms); nm2 = 0;

for m = 1:l(s)
    i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
    
    if L(m, 3, s) == -1
        Mm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsml(3 * M + 1:4 * M, :, i, j); fsml(1:M, :, i, j)]; % 1|2
        nm2 = nm2 + 1;
    end
    
    % check relationship with (i-1,j)
    if L(m, 4, s) == -1
        Mm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(M + 1:3 * M, :, i, j); % 2|1
        nm2 = nm2 + 1;
    end
    
    % check relationship with (i+1,j)
    if L(m, 5, s) == -1
        Mm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(1:2 * M, :, i, j); % 2/1
        nm2 = nm2 + 1;
    end
    
    % check relationship with (i,j-1)
    if L(m, 6, s) == -1
        Mm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(2 * M + 1:4 * M, :, i, j); % 1/2
        nm2 = nm2 + 1;
    end
    
    % check relationship with (i,j+1)
end

%ȷ��M_2^-, b_2^{0,-}ֱ����0����Ϊ��b_2^-ʱ����Ҫ��ȷ��һ��λ��
Mm{2} = Mm2; bm_temp{2} = zeros(2 * M * lm, 1);

for s = 2:S - 1
    lms = 8 * M * l(s);
    lp = sum(((L(:, 3, s) == 1) + (L(:, 4, s) == 1) + (L(:, 5, s) == 1) + (L(:, 6, s) == 1)) .* (L(:, 1, s) ~= 0));
    Aps = zeros(4 * M * lp, lms); Ams = zeros(4 * M * lp, 8 * M * l(s + 1)); nAp = 0; % prepare storage space and counter for A_s^+ and A_{s+1}^-
    bAs = zeros(4 * M * lp, 1);
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == 1
            Aps(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsml(:, :, i, j); % A_s^+
            k = L(m, 7, s); Ams(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmr(:, :, i - 1, j); % A_{s+1}^-
            bAs(nAp * 4 * M + 1:(nAp + 1) * 4 * M) = -Psi0l(:, i, j) + Psi0r(:, i - 1, j);
            nAp = nAp + 1;
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == 1
            Aps(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(:, :, i, j); % A_s^+
            k = L(m, 8, s); Ams(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsml(:, :, i + 1, j); % A_{s+1}^-
            bAs(nAp * 4 * M + 1:(nAp + 1) * 4 * M) = -Psi0r(:, i, j) + Psi0l(:, i + 1, j);
            nAp = nAp + 1;
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == 1
            Aps(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(:, :, i, j); % A_s^+
            k = L(m, 9, s); Ams(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmt(:, :, i, j - 1); % A_{s+1}^-
            bAs(nAp * 4 * M + 1:(nAp + 1) * 4 * M) = -Psi0b(:, i, j) + Psi0t(:, i, j - 1);
            nAp = nAp + 1;
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == 1
            Aps(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(:, :, i, j); % A_s^+
            k = L(m, 10, s); Ams(nAp * 4 * M + 1:(nAp + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmb(:, :, i, j + 1); % A_{s+1}^-
            bAs(nAp * 4 * M + 1:(nAp + 1) * 4 * M) = -Psi0t(:, i, j) + Psi0b(:, i, j + 1);
            nAp = nAp + 1;
        end
        
        % check relationship with (i,j-1)
    end
    
    [Xm{s}, Ym, Wm, Zm{s}] = form_XYWZ(MB{s}, MD{s}, Aps, Mm{s});
    Mm{s + 1} = Wm * Ams;
    bm_temp{s + 1} = -Ym * bD{s} - Wm * bAs - Zm{s} * bm_temp{s};
end

% for s=2:S
%     [tX{s},tY,tW{s},tZ]=form_XYWZ(MB{s},MD{s},[Mm{s};Mp{s}],MBo{s});
%     tYbDs{s}=tY*bD{s};
%     [LZ{s},UZ{s},PZ{s}]=lu(tZ,'vector');
% end
for s = 2:S
    [LM{s}, UM{s}, PM{s}] = lu([MB{s}; MD{s}; Mp{s}; Mm{s}], 'vector');
end

T_offline_part2 = toc;
%% Online stage
num_test = size(Input{3}{1}, 3);
list_psi_x = zeros(4 * M, I, J + 1, num_test);
list_psi_y = zeros(4 * M, I + 1, J, num_test);
list_alpha = zeros(8 * M, I, J, num_test);
list_Phi = zeros(I, J, num_test);
list_Psi = zeros(4 * M, I, J, num_test);
list_varepsilon = zeros(I, J, num_test); list_sigma_T = list_varepsilon; list_sigma_a = list_varepsilon; list_q = list_varepsilon;
list_fsml = zeros(4 * M, 8 * M, I, J, num_test); list_fsmr = list_fsml; list_fsmb = list_fsml; list_fsmt = list_fsml;
T_online_each = zeros(1, num_test);

for n = 1:num_test
    tic
    psiL = Input{3}{1}(:, :, n); psiR = Input{3}{2}(:, :, n); psiB = Input{3}{3}(:, :, n); psiT = Input{3}{4}(:, :, n);
    g_sigma_T = Input{3}{5}{n}; g_sigma_a = Input{3}{6}{n}; g_varepsilon = Input{3}{7}{n}; g_q = Input{3}{8}{n};
    bB = cell(S, 1);
    s = 1;
    lms = 8 * M * l(s);
    lB = sum(((L(:, 3, s) == 0) .* (L(:, 1, s) == 1) + (L(:, 4, s) == 0) .* (L(:, 1, s) == I) + (L(:, 5, s) == 0) .* (L(:, 2, s) == 1) + (L(:, 6, s) == 0) .* (L(:, 2, s) == J)) .* (L(:, 1, s) ~= 0));
    lD = sum(((L(:, 3, s) == 0) .* (L(:, 1, s) ~= 1) + (L(:, 4, s) == 0) .* (L(:, 1, s) ~= I) + (L(:, 5, s) == 0) .* (L(:, 2, s) ~= 1) + (L(:, 6, s) == 0) .* (L(:, 2, s) ~= J)) .* (L(:, 1, s) ~= 0));
    Bs = zeros(2 * M * lB, lms); nB = 0;
    Bos = Bs; bBos = zeros(2 * M * lB, 1);
    Ds = zeros(2 * M * lD, lms); bDs = zeros(2 * M * lD, 1); nD = 0;
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        varepsilon(i, j) = integral2(g_varepsilon, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        sigma_T(i, j) = integral2(g_sigma_T, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        sigma_a(i, j) = integral2(g_sigma_a, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        q(i, j) = integral2(g_q, (i - 1) * hx, i * hx, (j - 1) * hy, j * hy) / hx / hy;
        fsml(:, :, i, j) = fsm((i - 1) * hx + xl, (j - 0.5) * hy + yl, i, j);
        fsmr(:, :, i, j) = fsm(i * hx + xl, (j - 0.5) * hy + yl, i, j);
        fsmb(:, :, i, j) = fsm((i - 0.5) * hx + xl, (j - 1) * hy + yl, i, j);
        fsmt(:, :, i, j) = fsm((i - 0.5) * hx + xl, j * hy + yl, i, j);
        fsmc(:, :, i, j) = fsm((i - 0.5) * hx + xl, (j - 0.5) * hy + yl, i, j);
        %fundamental solution matrices at the left/right/bottom/top edge midpoint of each cell
        Psi0l(:, i, j) = Psi0((i - 1) * hx + xl, (j - 0.5) * hy + yl, i, j);
        Psi0r(:, i, j) = Psi0(i * hx + xl, (j - 0.5) * hy + yl, i, j);
        Psi0b(:, i, j) = Psi0((i - 0.5) * hx + xl, (j - 1) * hy + yl, i, j);
        Psi0t(:, i, j) = Psi0((i - 0.5) * hx + xl, j * hy + yl, i, j);
        Psi0c(:, i, j) = Psi0((i - 0.5) * hx + xl, (j - 0.5) * hy + yl, i, j);
        %special solution vectors at the left/right/bottom/top edge midpoint of each cell
    end
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == 0
            
            if i > 1
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsml(3 * M + 1:4 * M, :, i, j); fsml(1:M, :, i, j)]; % incident part
                k = L(m, 7, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) =- [fsmr(3 * M + 1:4 * M, :, i - 1, j); fsmr(1:M, :, i - 1, j)]; % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) =- [Psi0l(3 * M + 1:4 * M, i, j); Psi0l(1:M, i, j)] + [Psi0r(3 * M + 1:4 * M, i - 1, j); Psi0r(1:M, i - 1, j)];
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsml(3 * M + 1:4 * M, :, i, j); fsml(1:M, :, i, j)]; % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsml(M + 1:3 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0l(M + 1:3 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [1; j];
            end
            
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == 0
            
            if i < I
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(M + 1:3 * M, :, i, j); % incident part
                k = L(m, 8, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsml(M + 1:3 * M, :, i + 1, j); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0r(M + 1:3 * M, i, j) + Psi0l(M + 1:3 * M, i + 1, j);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmr(M + 1:3 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = [fsmr(3 * M + 1:4 * M, :, i, j); fsmr(1:M, :, i, j)]; % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = [Psi0r(3 * M + 1:4 * M, i, j); Psi0r(1:M, i, j)];
                nB = nB + 1;
                LBo(nB, s, :) = [2; j];
            end
            
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == 0
            
            if j > 1
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(1:2 * M, :, i, j); % incident part
                k = L(m, 9, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsmt(1:2 * M, :, i, j - 1); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0b(1:2 * M, i, j) + Psi0t(1:2 * M, i, j - 1);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(1:2 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmb(2 * M + 1:4 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0b(2 * M + 1:4 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [3; i];
            end
            
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == 0
            
            if j < I
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(2 * M + 1:4 * M, :, i, j); % incident part
                k = L(m, 10, s);
                Ds(nD * 2 * M + 1:(nD + 1) * 2 * M, (k - 1) * 8 * M + 1:k * 8 * M) = -fsmb(2 * M + 1:4 * M, :, i, j + 1); % outgoing part
                bDs(nD * 2 * M + 1:(nD + 1) * 2 * M) = -Psi0t(2 * M + 1:4 * M, i, j) + Psi0b(2 * M + 1:4 * M, i, j + 1);
                nD = nD + 1;
            else
                Bs(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(2 * M + 1:4 * M, :, i, j); % incident part
                Bos(nB * 2 * M + 1:(nB + 1) * 2 * M, (m - 1) * 8 * M + 1:m * 8 * M) = fsmt(1:2 * M, :, i, j); % outgoing part
                bBos(nB * 2 * M + 1:(nB + 1) * 2 * M) = Psi0t(1:2 * M, i, j);
                nB = nB + 1;
                LBo(nB, s, :) = [4; i];
            end
            
        end
        
        % check relationship with (i,j-1)
    end
    
    MD{s} = Ds; bD{s} = bDs;
    MB{s} = Bs;
    MBo1 = Bos; bBo{1} = bBos;
    % ȷ��B_1 \tB_1, D_1,
    for s = 1:S
        lB = sum(((L(:, 3, s) == 0) .* (L(:, 1, s) == 1) + (L(:, 4, s) == 0) .* (L(:, 1, s) == I) + (L(:, 5, s) == 0) .* (L(:, 2, s) == 1) + (L(:, 6, s) == 0) .* (L(:, 2, s) == J)) .* (L(:, 1, s) ~= 0));
        bBs = zeros(2 * M * lB, 1); nB = 0;
        
        for m = 1:l(s)
            i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
            
            if L(m, 3, s) == 0 && i == 1
                bBs(nB * 2 * M + 1:(nB + 1) * 2 * M) = psiL(:, j) - [Psi0l(3 * M + 1:4 * M, i, j); Psi0l(1:M, i, j)];
                nB = nB + 1;
            end
            
            if L(m, 4, s) == 0 && i == I
                bBs(nB * 2 * M + 1:(nB + 1) * 2 * M) = psiR(:, j) - Psi0r(M + 1:3 * M, i, j);
                nB = nB + 1;
            end
            
            if L(m, 5, s) == 0 && j == 1
                bBs(nB * 2 * M + 1:(nB + 1) * 2 * M) = psiB(:, i) - Psi0b(1:2 * M, i, j);
                nB = nB + 1;
            end
            
            if L(m, 6, s) == 0 && j == J
                bBs(nB * 2 * M + 1:(nB + 1) * 2 * M) = psiT(:, i) - Psi0t(2 * M + 1:4 * M, i, j);
                nB = nB + 1;
            end
            
        end
        
        bB{s} = bBs;
    end
    
    % ȷ�� b_s^B
    for s = S:-1:2
        
        if s == S
            add_bps = -Xp{s} * bB{s};
        else
            add_bps = -Zp{s} * add_bps - Xp{s} * bB{s};
        end
        
        bp{s - 1} = bp_temp{s - 1} + add_bps;
    end
    
    % compute bb_s^+
    s = 2;
    lm = sum(((L(:, 3, s) == -1) + (L(:, 4, s) == -1) + (L(:, 5, s) == -1) + (L(:, 6, s) == -1)) .* (L(:, 1, s) ~= 0));
    Aps = zeros(4 * M * lm, 8 * M * l(s - 1)); nAm = 0; % prepare storage space and counter for A_s^- and A_{s-1}^+
    bAs = zeros(4 * M * lm, 1);
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == -1
            k = L(m, 7, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmr(:, :, i - 1, j); % A_{s-1}^+
            bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0l(:, i, j) + Psi0r(:, i - 1, j);
            nAm = nAm + 1;
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == -1
            k = L(m, 8, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsml(:, :, i + 1, j); % A_{s-1}^+
            bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0r(:, i, j) + Psi0l(:, i + 1, j);
            nAm = nAm + 1;
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == -1
            k = L(m, 9, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmt(:, :, i, j - 1); % A_{s-1}^+
            bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0b(:, i, j) + Psi0t(:, i, j - 1);
            nAm = nAm + 1;
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == -1
            k = L(m, 10, s); Aps(nAm * 4 * M + 1:(nAm + 1) * 4 * M, (k - 1) * 8 * M + 1:k * 8 * M) = fsmb(:, :, i, j + 1); % A_{s-1}^+
            bAs(nAm * 4 * M + 1:(nAm + 1) * 4 * M) = -Psi0t(:, i, j) + Psi0b(:, i, j + 1);
            nAm = nAm + 1;
        end
        
        % check relationship with (i,j-1)
    end
    
    Mp{1} = W2p * Aps;
    bp{1} = bp{1} - W2p * bAs;
    % ����bb_1^+ M_1^+
    beta = cell(S, 1);
    beta_C = [MB{1}; MD{1}; Mp{1}] \ [bB{1}; bD{1}; bp{1}];
    beta{1} = beta_C;
    % ���Omega_C�ڲ���beta{1}(beta_C)
    s = 2; lm = sum(((L(:, 3, s) == -1) + (L(:, 4, s) == -1) + (L(:, 5, s) == -1) + (L(:, 6, s) == -1)) .* (L(:, 1, s) ~= 0));
    bm2 = zeros(2 * M * lm, 1); nm2 = 0;
    
    for m = 1:l(s)
        i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
        
        if L(m, 3, s) == -1
            k = L(m, 7, s);
            bm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M) = [fsmr(3 * M + 1:4 * M, :, i - 1, j); fsmr(1:M, :, i - 1, j)] * beta_C((k - 1) * 8 * M + 1:k * 8 * M) + [Psi0r(3 * M + 1:4 * M, i - 1, j); Psi0r(1:M, i - 1, j)] - [Psi0l(3 * M + 1:4 * M, i, j); Psi0l(1:M, i, j)]; % 1|2
            nm2 = nm2 + 1;
        end
        
        % check relationship with (i-1,j)
        if L(m, 4, s) == -1
            k = L(m, 8, s);
            bm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M) = fsml(M + 1:3 * M, :, i + 1, j) * beta_C((k - 1) * 8 * M + 1:k * 8 * M) + Psi0l(M + 1:3 * M, i + 1, j) - Psi0r(M + 1:3 * M, i, j); % 2|1
            nm2 = nm2 + 1;
        end
        
        % check relationship with (i+1,j)
        if L(m, 5, s) == -1
            k = L(m, 9, s);
            bm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M) = fsmt(1:2 * M, :, i, j - 1) * beta_C((k - 1) * 8 * M + 1:k * 8 * M) + Psi0t(1:2 * M, i, j - 1) - Psi0b(1:2 * M, i, j); % 2/1
            nm2 = nm2 + 1;
        end
        
        % check relationship with (i,j-1)
        if L(m, 6, s) == -1
            k = L(m, 10, s);
            bm2(nm2 * 2 * M + 1:(nm2 + 1) * 2 * M) = fsmb(2 * M + 1:4 * M, :, i, j + 1) * beta_C((k - 1) * 8 * M + 1:k * 8 * M) + Psi0b(2 * M + 1:4 * M, i, j + 1) - Psi0t(2 * M + 1:4 * M, i, j); % 1/2
            nm2 = nm2 + 1;
        end
        
        % check relationship with (i,j+1)
    end
    
    bm{2} = bm2;
    % ����Omega_C�ĳ���psi_C(b_2^-)
    add_bms = bm2;
    
    for s = 2:S - 1
        add_bms = -Zm{s} * add_bms - Xm{s} * bB{s};
        bm{s + 1} = bm_temp{s + 1} + add_bms;
    end
    
    % ���� bb_s^-
    bBo{1} = bBo{1} + MBo1 * beta_C;
    % ����������s=1
    for s = 2:S
        b = [bB{s}; bD{s}; bp{s}; bm{s}];
        beta{s} = UM{s} \ (LM{s} \ b(PM{s}));
        bBo{s} = bBo{s} + MBo{s} * beta{s};
    end
    
    % ����������s>1
    psioL = zeros(2 * M, J); psioR = psioL;
    psioB = zeros(2 * M, I); psioT = psioB;
    
    for s = 1:S
        lB = size(bBo{s}, 1) / (2 * M);
        
        for m = 1:lB
            
            switch LBo(m, s, 1)
                case 1
                    psioL(:, LBo(m, s, 2)) = bBo{s}((m - 1) * 2 * M + 1:m * 2 * M);
                case 2
                    psioR(:, LBo(m, s, 2)) = bBo{s}((m - 1) * 2 * M + 1:m * 2 * M);
                case 3
                    psioB(:, LBo(m, s, 2)) = bBo{s}((m - 1) * 2 * M + 1:m * 2 * M);
                case 4
                    psioT(:, LBo(m, s, 2)) = bBo{s}((m - 1) * 2 * M + 1:m * 2 * M);
            end
            
        end
        
    end
    
    % ��������
    T_online_each = toc;
    %% gmres
    %     tol=1e-10;
    %     [T_gmres_each(n),alpha_gmres]=gmres_solver(tol,I,J,M,psiL,psiR,psiB,psiT,fsml,fsmr,fsmb,fsmt,Psi0l,Psi0r,Psi0b,Psi0t);
    %% ����alpha
    alpha = zeros(8 * M, I, J);
    
    for s = 1:S
        
        for m = 1:l(s)
            i = L(m, 1, s); j = L(m, 2, s); % get index (i,j)
            alpha(:, i, j) = beta{s}((m - 1) * 8 * M + 1:m * 8 * M);
        end
        
    end
    
    %% ����psix,psiy
    psix = zeros(4 * M, I, J + 1); %��x��ƽ�еıߵ��ص�
    psiy = zeros(4 * M, I + 1, J); %��y��ƽ�еıߵ��ص�
    phix = zeros(I, J + 1);
    phiy = zeros(I + 1, J);
    
    for i = 1:I
        
        for j = 1:J
            psix(:, i, j) = fsmb(:, :, i, j) * alpha(:, i, j) + Psi0b(:, i, j);
            psiy(:, i, j) = fsml(:, :, i, j) * alpha(:, i, j) + Psi0l(:, i, j);
            phix(i, j) = omega' * psix(:, i, j);
            phiy(i, j) = omega' * psiy(:, i, j);
        end
        
    end
    
    for i = 1:I
        psix(:, i, J + 1) = fsmt(:, :, i, j) * alpha(:, i, j) + Psi0t(:, i, j);
        phix(i, j) = omega' * psix(:, i, j);
    end
    
    for j = 1:J
        psiy(:, I + 1, j) = fsmr(:, :, i, j) * alpha(:, i, j) + Psi0r(:, i, j);
        phiy(i, j) = omega' * psiy(:, i, j);
    end
    
    %% simple_check
    % resbl=zeros(2*M,J); resbr=zeros(2*M,J);
    % resbb=zeros(2*M,I); resbt=zeros(2*M,I);
    % for j=1:J
    %     resbl(:,j)=[fsml(3*M+1:4*M,:,1,j);fsml(1:M,:,1,j)]*alpha(:,1,j)+[Psi0l(3*M+1:4*M,1,j);Psi0l(1:M,1,j)]-psiL(:,j);
    %     resbr(:,j)=fsmr(M+1:3*M,:,I,j)*alpha(:,I,j)+Psi0r(M+1:3*M,I,j)-psiR(:,j);
    % end
    % for i=1:I
    %     resbb(:,i)=fsmb(1:2*M,:,i,1)*alpha(:,i,1)+Psi0b(1:2*M,i,1)-psiB(:,i);
    %     resbt(:,i)=fsmt(2*M+1:4*M,:,i,J)*alpha(:,i,J)+Psi0t(2*M+1:4*M,i,J)-psiT(:,i);
    % end
    % resxy=zeros(4*M,I,J-1); resyx=zeros(4*M,I-1,J);
    % for i=1:I
    %     for j=1:J-1
    %         resxy(:,i,j)=(fsmt(:,:,i,j)*alpha(:,i,j)+Psi0t(:,i,j))-(fsmb(:,:,i,j+1)*alpha(:,i,j+1)+Psi0b(:,i,j+1));
    %     end
    % end
    % for i=1:I-1
    %     for j=1:J
    %         resyx(:,i,j)=(fsmr(:,:,i,j)*alpha(:,i,j)+Psi0r(:,i,j))-(fsml(:,:,i+1,j)*alpha(:,i+1,j)+Psi0l(:,i+1,j));
    %     end
    % end
    %% �µ���ͼ��ʽ
    Phic = zeros(I, J);
    Psic = zeros(4 * M, I, J);
    
    for i = 1:I
        
        for j = 1:J
            Psic(:, i, j) = fsmc(:, :, i, j) * alpha(:, i, j) + Psi0c(:, i, j);
            Phic(i, j) = omega' * Psic(:, i, j);
        end
        
    end
    
    %    surf(xl+0.5*hx:hx:xr-0.5*hx,yl+0.5*hy:hy:yr-0.5*hy,Phic);
    %    view(60,30)
    %    xlabel('x')
    %    ylabel('y')
    %    zlabel('\phi(x,y)')
    %    axis([0 1 0 1 0 1.1*max(max(Phic))])
    
    %    saveas(gcf, 'save.jpg')
    
    %% ���
    list_varepsilon(:, :, n) = varepsilon;
    list_sigma_T(:, :, n) = sigma_T;
    list_sigma_a(:, :, n) = sigma_a;
    list_q(:, :, n) = q;
    list_fsml(:, :, :, :, n) = fsml;
    list_fsmr(:, :, :, :, n) = fsmr;
    list_fsmb(:, :, :, :, n) = fsmb;
    list_fsmt(:, :, :, :, n) = fsmt;
    list_psi_x(:, :, :, n) = psix;
    list_psi_y(:, :, :, n) = psiy;
    list_alpha(:, :, :, n) = alpha;
    list_Psi(:, :, :, n) = Psic;
    list_Phi(:, :, n) = Phic;
end

end
