function [LIST] = sparse_trans(direct, io, i, j, A)
% ��ϡ�������Ŀ����Mת��ΪLIST��ʽ,kΪ��ʼ���-1
% directλ�� 1 �� 2 �� 3 �� 4 ��
% ioλ�� 1��ʾ���� 0��ʾ����
[m, n] = size(A);
LIST = zeros(m * n, 3);
k = (i - 1) * J + j;
jlist = (k - 1) * 8 * M + 1:k * 8 * M;

if io == 1
    
    if direct == 1
        
        if j > 1
            k = (i - 1) * J + j - 1;
            direct = 3;
        else
            error('io/direct is wrong');
        end
        
    elseif direct == 2
        
        if i < I
            k = i * J + j;
            direct = 4;
        else
            error('io/direct is wrong');
        end
        
    elseif direct == 3
        
        if j < J
            k = (i - 1) * J + j + 1;
            direct = 1;
        else
            error('io/direct is wrong');
        end
        
    elseif direct == 4
        
        if i > 1
            k = (i - 2) * J + j;
            direct = 2;
        else
            error('io/direct is wrong');
        end
        
    end
    
end

ilist = (k - 1) * 8 * M + 1 + (direct - 1) * 2 * M:(k - 1) * 8 * M + direct * 2 * M;

for ki = 1:2 * M
    
    for kj = 1:8 * M
        LIST((ki - 1) * 8 * M + kj, :) = [ilist(ki), jlist(kj), A(ki, kj)];
    end
    
end

end
