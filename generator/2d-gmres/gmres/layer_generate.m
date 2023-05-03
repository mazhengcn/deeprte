function [L,S,l]=layer_generate(LC,I,J)
L=zeros(1,10,1);%index set in the layer (max size of layer,1(i)/2(j)/3(i-1)/4(i+1)/5(j-1)/6(j+1),index of layer)
% for second dimension in L, the first two value is i,j, the 3~6th value is position of i-1,i+1,j-1,j+1
% (0 for boundary or the sth layer, -1 for (s-1)th layer, 1 for (s+1)th layer)
% 7~10th value is their index
% 一个改进的方向是把L写成cell
l=zeros(1,1);%size set of layer (size of layer,index of layer)
[l(1,1),~]=size(LC);
L(1:l(1,1),1:2,1)=LC;
% information of LC
s=1;
if sum(l)==I*J
    S=s;
else
    S=Inf;
end
while sum(l)<=I*J
    s=s+1; % present layer
    if s<S
        L(1,1,s)=0; l(s)=0; % extend one layer
    end
    for m=1:l(s-1)
        i=L(m,1,s-1); j=L(m,2,s-1); % take one index (i,j) in last layer
        if i>1
            [p,mij,sij]=ip([i-1,j],L(:,:,max(1,s-2):min(s,S)),l(max(1,s-2):min(s,S)));
            if p==0
                l(s)=l(s)+1;
                L(l(s),1,s)=i-1; L(l(s),2,s)=j;
                L(m,3,s-1)=1;
                L(m,7,s-1)=l(s);
            else
                if s>2
                    if sij==3
                        L(m,3,s-1)=1;
                    elseif sij==1
                        L(m,3,s-1)=-1;
                    end
                else
                    if sij==2
                        L(m,3,s-1)=1;
                    end
                end
                L(m,7,s-1)=mij;
            end
            
        end
        % check if (i-1,j) is already counted
        if i<I
            [p,mij,sij]=ip([i+1,j],L(:,:,max(1,s-2):min(s,S)),l(max(1,s-2):min(s,S)));
            if p==0
                l(s)=l(s)+1;
                L(l(s),1,s)=i+1; L(l(s),2,s)=j;
                L(m,4,s-1)=1;
                L(m,8,s-1)=l(s);
            else
                if s>2
                    if sij==3
                        L(m,4,s-1)=1;
                    elseif sij==1
                        L(m,4,s-1)=-1;
                    end
                else
                    if sij==2
                        L(m,4,s-1)=1;
                    end
                end
                L(m,8,s-1)=mij;
            end
        end
        % check if (i+1,j) is already counted
        if j>1
            [p,mij,sij]=ip([i,j-1],L(:,:,max(1,s-2):min(s,S)),l(max(1,s-2):min(s,S)));
            if p==0
                l(s)=l(s)+1;
                L(l(s),1,s)=i; L(l(s),2,s)=j-1;
                L(m,5,s-1)=1;
                L(m,9,s-1)=l(s);
            else
                if s>2
                    if sij==3
                        L(m,5,s-1)=1;
                    elseif sij==1
                        L(m,5,s-1)=-1;
                    end
                else
                    if sij==2
                        L(m,5,s-1)=1;
                    end
                end
                L(m,9,s-1)=mij;
            end
        end
        % check if (i,j-1) is already counted
        if j<J
            [p,mij,sij]=ip([i,j+1],L(:,:,max(1,s-2):min(s,S)),l(max(1,s-2):min(s,S)));
            if p==0
                l(s)=l(s)+1;
                L(l(s),1,s)=i; L(l(s),2,s)=j+1;
                L(m,6,s-1)=1;
                L(m,10,s-1)=l(s);
            else
                if s>2
                    if sij==3
                        L(m,6,s-1)=1;
                    elseif sij==1
                        L(m,6,s-1)=-1;
                    end
                else
                    if sij==2
                        L(m,6,s-1)=1;
                    end
                end
                L(m,10,s-1)=mij;
            end
        end
        % check if (i,j+1) is already counted
    end
    if S==s-1
        break
    end
    if sum(l)==I*J
        S=s;
    end
end
end
