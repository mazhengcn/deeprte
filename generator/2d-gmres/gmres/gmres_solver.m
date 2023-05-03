function [Tgmres,alpha_gmres]=gmres_solver(tol,I,J,M,psiL,psiR,psiB,psiT,fsml,fsmr,fsmb,fsmt,Psi0l,Psi0r,Psi0b,Psi0t)
%% nested function
    function [LIST]=sparse_trans(direct,io,i,j,A)
        % ��ϡ�������Ŀ����Mת��ΪLIST��ʽ,kΪ��ʼ���-1
        % directλ�� 1 �� 2 �� 3 �� 4 ��
        % ioλ�� 1��ʾ���� 0��ʾ����
        [m,n]=size(A);
        LIST=zeros(m*n,3);
        k=(i-1)*J+j;
        jlist=(k-1)*8*M+1:k*8*M;
        if io==1
            if direct==1
                if j>1
                    k=(i-1)*J+j-1;
                    direct=3;
                else
                    error('io/direct is wrong');
                end
            elseif direct==2
                if i<I
                    k=i*J+j;
                    direct=4;
                else
                    error('io/direct is wrong');
                end
            elseif direct==3
                if j<J
                    k=(i-1)*J+j+1;
                    direct=1;
                else
                    error('io/direct is wrong');
                end
            elseif direct==4
                if i>1
                    k=(i-2)*J+j;
                    direct=2;
                else
                    error('io/direct is wrong');
                end
            end
        end
        ilist=(k-1)*8*M+1+(direct-1)*2*M:(k-1)*8*M+direct*2*M;
        for ki=1:2*M
            for kj=1:8*M
                LIST((ki-1)*8*M+kj,:)=[ilist(ki),jlist(kj),A(ki,kj)];
            end
        end
    end

tic
LIST=zeros(0,3);
b=zeros(8*M*I*J,1);
ind=0;
for i=1:I
    for j=1:J
        % ע�⣡LIST������A,b������ǲ�ͬ��
        k=(i-1)*J+j;
        P=[fsmb(1:2*M,:,i,j);fsmr(M+1:3*M,:,i,j);fsmt(2*M+1:4*M,:,i,j);fsml(3*M+1:4*M,:,i,j);fsml(1:M,:,i,j)];
        %��ǰcell�·�����
        LIST(ind+1:ind+16*M^2,:)=sparse_trans(1,0,i,j,fsmb(1:2*M,:,i,j)/P);
        ind=ind+16*M^2;
        if j==1
            b((k-1)*8*M+1:(k-1)*8*M+2*M)=psiB(:,i)-Psi0b(1:2*M,i,j);
        else
            b((k-1)*8*M+1:(k-1)*8*M+2*M)=Psi0t(1:2*M,i,j-1)-Psi0b(1:2*M,i,j);
            Pb=[fsmb(1:2*M,:,i,j-1);fsmr(M+1:3*M,:,i,j-1);fsmt(2*M+1:4*M,:,i,j-1);fsml(3*M+1:4*M,:,i,j-1);fsml(1:M,:,i,j-1)];
            LIST(ind+1:ind+16*M^2,:)=sparse_trans(3,1,i,j-1,-fsmt(1:2*M,:,i,j-1)/Pb);
            ind=ind+16*M^2;
        end
        %��ǰcell�ҷ�����
        LIST(ind+1:ind+16*M^2,:)=sparse_trans(2,0,i,j,fsmr(M+1:3*M,:,i,j)/P);
        ind=ind+16*M^2;
        if i==I
            b((k-1)*8*M+2*M+1:(k-1)*8*M+4*M)=psiR(:,j)-Psi0r(M+1:3*M,i,j);
        else
            b((k-1)*8*M+2*M+1:(k-1)*8*M+4*M)=Psi0l(M+1:3*M,i+1,j)-Psi0r(M+1:3*M,i,j);
            Pr=[fsmb(1:2*M,:,i+1,j);fsmr(M+1:3*M,:,i+1,j);fsmt(2*M+1:4*M,:,i+1,j);fsml(3*M+1:4*M,:,i+1,j);fsml(1:M,:,i+1,j)];
            LIST(ind+1:ind+16*M^2,:)=sparse_trans(4,1,i+1,j,-fsml(M+1:3*M,:,i+1,j)/Pr);
            ind=ind+16*M^2;
        end
        %��ǰcell�Ϸ�����
        LIST(ind+1:ind+16*M^2,:)=sparse_trans(3,0,i,j,fsmt(2*M+1:4*M,:,i,j)/P);
        ind=ind+16*M^2;
        if j==J
            b((k-1)*8*M+4*M+1:(k-1)*8*M+6*M)=psiT(:,i)-Psi0t(2*M+1:4*M,i,j);
        else
            b((k-1)*8*M+4*M+1:(k-1)*8*M+6*M)=Psi0b(2*M+1:4*M,i,j+1)-Psi0t(2*M+1:4*M,i,j);
            Pt=[fsmb(1:2*M,:,i,j+1);fsmr(M+1:3*M,:,i,j+1);fsmt(2*M+1:4*M,:,i,j+1);fsml(3*M+1:4*M,:,i,j+1);fsml(1:M,:,i,j+1)];
            LIST(ind+1:ind+16*M^2,:)=sparse_trans(1,1,i,j+1,-fsmb(2*M+1:4*M,:,i,j+1)/Pt);
            ind=ind+16*M^2;
        end
        %��ǰcell������
        LIST(ind+1:ind+16*M^2,:)=sparse_trans(4,0,i,j,[fsml(3*M+1:4*M,:,i,j);fsml(1:M,:,i,j)]/P);
        ind=ind+16*M^2;
        if i==1
            b((k-1)*8*M+6*M+1:k*8*M)=psiL(:,j)-[Psi0l(3*M+1:4*M,i,j);Psi0l(1:M,i,j)];
        else
            b((k-1)*8*M+6*M+1:k*8*M)=[Psi0r(3*M+1:4*M,i-1,j);Psi0r(1:M,i-1,j)]-[Psi0l(3*M+1:4*M,i,j);Psi0l(1:M,i,j)];
            Pl=[fsmb(1:2*M,:,i-1,j);fsmr(M+1:3*M,:,i-1,j);fsmt(2*M+1:4*M,:,i-1,j);fsml(3*M+1:4*M,:,i-1,j);fsml(1:M,:,i-1,j)];
            LIST(ind+1:ind+16*M^2,:)=sparse_trans(2,1,i-1,j,-[fsmr(3*M+1:4*M,:,i-1,j);fsmr(1:M,:,i-1,j)]/Pl);
            ind=ind+16*M^2;
        end
    end
end
BM=sparse(LIST(:,1),LIST(:,2),LIST(:,3),8*M*I*J,8*M*I*J);
maxit = 20;
[L1,U1] = ilu(BM,struct('type','ilutp','droptol',1e-6));
[alpha_gmres_pre,flag,~,~,resvec]=gmres(BM,b,5,tol,maxit,L1,U1);
% flag
% log10(resvec)
Tgmres=toc;
% err_gmres=max(abs(BM*alpha_gmres_pre-b))
alpha_gmres=zeros(8*M,I,J);
for i=1:I
    for j=1:J
        k=(i-1)*J+j;
        P=[fsmb(1:2*M,:,i,j);fsmr(M+1:3*M,:,i,j);fsmt(2*M+1:4*M,:,i,j);fsml(3*M+1:4*M,:,i,j);fsml(1:M,:,i,j)];
        alpha_gmres(:,i,j)=P\alpha_gmres_pre((k-1)*8*M+1:k*8*M);
    end
end
%% simple_check
% alpha=alpha_gmres;
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
end
