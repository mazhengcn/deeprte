function [X,Y,W,Z] = form_XYWZ(B,D,A,M)
[mB,nB]=size(B); [mD,nD]=size(D); [mA,nA]=size(A); [mM,nM]=size(M);
if nD~=nB || nA~=nB || nM~=nB
    error('column numbers of input matrix are not same');
end
m=mB+mD+mA+mM; n=nB;
G=[B;D;A;M];
[Q,~]=qr(G); Q=Q';
X=Q(n+1:m,1:mB); Y=Q(n+1:m,mB+1:mB+mD); W=Q(n+1:m,mB+mD+1:mB+mD+mA); Z=Q(n+1:m,mB+mD+mA+1:end);
end
