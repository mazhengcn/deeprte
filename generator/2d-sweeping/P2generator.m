function Pmatrix = P2generator(N, g)
%注意，这里没有保可降一维性和可降二维性，但N=3时恰好满足这些条件
[omega0, ct0, st0, M, ~, mut0] = qnwlege2(N);
omega = [omega0; omega0] / 2;
ct = [ct0; ct0];
st = [st0; st0];
mut = [mut0; -mut0];

G = @(x)0.5 * (1 - g ^ 2) ./ ((1 + g ^ 2 - 2 * g * x) .^ (1.5));
pt = zeros((8 * M) ^ 2, 1);
Hm = zeros(8 * M, (8 * M) ^ 2); Hn = Hm; Hc = Hm; Hs = Hm; Hmut = Hm;
dm = ones(8 * M, 1); dn = dm; dc = g * ct; ds = g * st; dmut = g * mut;

for n = 1:8 * M
    
    for m = 1:8 * M
        k = (n - 1) * 8 * M + m;
        pt(k) = G(ct(m) * ct(n) + st(m) * st(n) + mut(m) * mut(n));
        Hm(m, k) = omega(n);
        Hn(n, k) = omega(m);
        Hc(n, k) = omega(m) * ct(m);
        Hs(n, k) = omega(m) * st(m);
        Hmut(n, k) = omega(m) * mut(m);
    end
    
end

% H1=zeros((4*M)^2,(8*M)^2); H2=H1;
% d1=zeros((4*M)^2,1); d2=d1;
% for n=1:4*M
%     for m=1:4*M
%         k=(n-1)*4*M+m;
%         H1(k,(n-1)*8*M+m)=1;
%         H1(k,(n+4*M-1)*8*M+m+4*M)=-1;
%         H2(k,(n+4*M-1)*8*M+m)=1;
%         H2(k,(n-1)*8*M+m+4*M)=-1;
%     end
% end
% H=[Hm;Hn;Hc;Hs;Hmut;H1;H2];
% d=[dm;dn;dc;ds;dmut;d1;d2];

H = [Hm; Hn; Hc; Hs; Hmut];
d = [dm; dn; dc; ds; dmut];
% R=rref([H eye(size(H,1))]);
% r=size(H,1)+1-find(R(end:-1:1,size(H,2)),1);
% Hf=R(1:r,1:(8*M)^2);
% df=R(1:r,size(H,2)+1:end)*d;
% pd=pt-Hf'*((Hf*Hf')\(Hf*pt-df));
pd = lsqlin(eye(size(pt, 1)), pt, [], [], H, d, zeros(size(pt)), []);
Pmatrix = zeros(4 * M);

for n = 1:4 * M
    
    for m = 1:4 * M
        Pmatrix(n, m) = (pd((n - 1) * 8 * M + m) + pd((n + 4 * M - 1) * 8 * M + m)) / 2;
    end
    
end

% P=zeros(8*M);
% for n=1:8*M
%     for m=1:8*M
%         P(n,m)=pd((n-1)*8*M+m);
%     end
% end
end
