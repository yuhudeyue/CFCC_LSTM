function [train_data,test_data]=data_process_day1(day,testdata)
%% 数据加载并完成初始归�?��
jun=testdata(3,3,day);
mi=int16(jun)-250;
mi=double(mi);
ma=int16(jun)+250;
M=5;
N=5;
    %原始数据切割�?*1的小矩阵, 原始数据分为150�?50列，�?天为�?��，共依次�?7组，作为训练数据
    train_data = zeros(75,M,N,day);
    for i = 1:day
        for m = 3:M-2
            for n = 3:N-2
                ij=1;
                for it=-2:2
                    for jt=-2:2
                        t=testdata(m+it,n+jt,i);
                        train_data(ij,m,n,i)=t;
                        ij=ij+1;
                    end
                end
                ij=1;
                for it=-2:2
                    for jt=-2:2
                        t=testdata(m+it,n+jt,i+1);
                        train_data(ij+25,m,n,i)=t;
                        ij=ij+1;
                    end
                end
                ij=1;
                for it=-2:2
                    for jt=-2:2
                        t=testdata(m+it,n+jt,i+2);
                        train_data(ij+50,m,n,i)=t;
                        ij=ij+1;
                    end
                end
                train_data(1:75,m,n,i) = (train_data(1:75,m,n,i)-mi)/(500);
            end
        end
    end
    test_data = zeros(25,M,N,day);
    for i = 1:day
        for m = 3:M-2
            for n = 3:N-2
                ij=1;
                for it=-2:2
                    for jt=-2:2
                        t=testdata(m+it,n+jt,i+3);
                        test_data(ij,m,n,i)=t;
                        ij=ij+1;
                    end
                end
%                  guiyi(m,n,i)=sqrt(nansum(test_data(:,m,n,i).^2));
                 test_data(1:25,m,n,i) = (test_data(1:25,m,n,i)-mi)/(500);
            end
        end
    end
    %测试数据选取4*1小矩阵，分别为第4~100�?    

%     save(['data',num2str(day)],'train_data','test_data');
    



