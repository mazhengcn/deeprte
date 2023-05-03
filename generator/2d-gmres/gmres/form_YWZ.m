function [Y,W,Z] = form_YWZ(J,A,M)
[mJ,nJ]=size(J); [mA,nA]=size(A); [mM,nM]=size(M);
if nJ~=nA || nA~=nM
    error('column numbers of input matrix are not same');
end
m=mJ+mA+mM; n=nJ;
G=[J;A;M];
[Q,~]=qr(G); Q=Q';
Y=full(Q(n+1:m,1:mJ)); W=full(Q(n+1:m,mJ+1:mJ+mA)); Z=full(Q(n+1:m,mJ+mA+1:m));
end
