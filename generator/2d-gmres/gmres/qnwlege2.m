function [omega,ct,st,M,theta]=qnwlege2(N)
M=N*(N+1)/2; 
[zeta0,omega0]=qnwlege1(2*N,-1,1);
zeta1=zeta0(2*N:-1:N+1);
omega1=omega0(2*N:-1:N+1)/4;
zeta=zeros(4*M,1);
omega=zeta;
theta=zeta;
for n=1:N
    for i=1:n
        theta(n*(n-1)/2+i)=(2*i-1)/(4*n)*pi;
        omega(n*(n-1)/2+i)=omega1(n)/n;
        zeta(n*(n-1)/2+i)=zeta1(n);
    end
end
for i=1:3
theta(1+i*M:(i+1)*M)=theta(1:M)+i*pi/2;
omega(1+i*M:(i+1)*M)=omega(1:M);
zeta(1+i*M:(i+1)*M)=zeta(1:M);
end
ct=((1-zeta.^2).^(0.5)).*cos(theta);
st=((1-zeta.^2).^(0.5)).*sin(theta);
end