function [p,mij,sij]=ip(z,L,l)
% check whether index z is in layer set L(p=0/1),
% and if so, which layer (sij), and where in the layer (mij)
p=0; mij=0; sij=0;
[~,~,smax]=size(L);
for s=1:smax
    for m=1:l(s)
        if L(m,1,s)==z(1) && L(m,2,s)==z(2)
            p=1;
            sij=s;
            mij=m;
            break
        end
    end
    if p==1
        break
    end
end
end
