function Training_1_day1(date_1,date_2,date_3,date_4,testdata,day)

testdata=double(int16(testdata));

M=5;
N=5;
[train_data,test_data]=data_process_day1(day,testdata);
data_length=size(train_data,1);
data_num=size(train_data,4);
group_x_num = M;   %150
group_y_num = N;   %450
%% 网络参数初始�?% 结点数设�?
input_num=75;
cell_num=14;
output_num=25;
jun=testdata(3,3,day);
mi=int16(jun)-250;
mi=double(mi);
ma=int16(jun)+250;
d1=1;
d2=5;
% 网络状�?初始�?
cost_gate=1e-6;
% % 加载网络参数
HANG=600;
LIE=450;
liflag=0;

if (exist(['weight_' num2str(d1) '_' num2str(d2) '.mat'],'file') ~= 0)
        [bias_input_gate,...
            bias_forget_gate,...
            bias_output_gate,...
            weight_input_x,...
            weight_input_h,...
            weight_inputgate_x,...
            weight_inputgate_c,...
            weight_forgetgate_x,...
            weight_forgetgate_c,...
            weight_outputgate_x,...
            weight_outputgate_c,...
            weight_preh_h,...
            h_state] = load_weight(['weight_' num2str(d1) '_' num2str(d2) '.mat']);
else
    [bias_input_gate,...
        bias_forget_gate,...
        bias_output_gate,...
        weight_input_x,...
        weight_input_h,...
        weight_inputgate_x,...
        weight_inputgate_c,...
        weight_forgetgate_x,...
        weight_forgetgate_c,...
        weight_outputgate_x,...
        weight_outputgate_c,...
        weight_preh_h,...
        h_state] = generate_weight(cell_num,input_num,output_num,data_num);
end
[weight(1:M,1:N).bias_input_gate] = deal(bias_input_gate);
[weight(1:M,1:N).bias_forget_gate] = deal(bias_forget_gate);
[weight(1:M,1:N).bias_output_gate] = deal(bias_output_gate);
[weight(1:M,1:N).weight_input_x] = deal(weight_input_x);
[weight(1:M,1:N).weight_input_h] = deal(weight_input_h);
[weight(1:M,1:N).weight_inputgate_x] = deal(weight_inputgate_x);
[weight(1:M,1:N).weight_inputgate_c] = deal(weight_inputgate_c);
[weight(1:M,1:N).weight_forgetgate_x] = deal(weight_forgetgate_x);
[weight(1:M,1:N).weight_forgetgate_c] = deal(weight_forgetgate_c);
[weight(1:M,1:N).weight_outputgate_x] = deal(weight_outputgate_x);
[weight(1:M,1:N).weight_outputgate_c] = deal(weight_outputgate_c);
[weight(1:M,1:N).weight_preh_h] = deal(weight_preh_h);
[weight(1:M,1:N).h_state] = deal(h_state);
% [weight(1:M,1:N).cell_state] = deal(cell_state);


ii = 1;
Error_Cost = zeros(1,1000);
yita=0.2;
    
    for group_x = 3:group_x_num-2
        for group_y = 3:group_y_num-2
            for iter=1:30
                yita=0.2;           
                for m=1:data_num
                   
                    if(m==1)
                        
                        gate=tanh(train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_input_x);
                        input_gate_input=train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_inputgate_x+weight(group_x,group_y).bias_input_gate;
                        output_gate_input=train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_outputgate_x+weight(group_x,group_y).bias_output_gate;
                        input_gate = zeros(1,cell_num);
                        output_gate = zeros(1,cell_num);
                        for n=1:cell_num
                            input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                            output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
                        end
                        forget_gate=zeros(1,cell_num);
                        forget_gate_input=zeros(1,cell_num);
                        weight(group_x,group_y).cell_state(:,m)=(input_gate.*gate)';
                    else
                        gate=tanh(train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_input_x+weight(group_x,group_y).h_state(:,m-1)'*weight(group_x,group_y).weight_input_h);
                        input_gate_input=train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_inputgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_inputgate_c+weight(group_x,group_y).bias_input_gate;
                        forget_gate_input=train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_forgetgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_forgetgate_c+weight(group_x,group_y).bias_forget_gate;
                        output_gate_input=train_data(:,group_x,group_y,m)'*weight(group_x,group_y).weight_outputgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_outputgate_c+weight(group_x,group_y).bias_output_gate;
                        for n=1:cell_num
                            input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                            forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                            output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
                        end
                        weight(group_x,group_y).cell_state(:,m)=(input_gate.*gate+weight(group_x,group_y).cell_state(:,m-1)'.*forget_gate)';
                    end
                    pre_h_state=tanh(weight(group_x,group_y).cell_state(:,m)').*output_gate;
                    weight(group_x,group_y).h_state(:,m)=(pre_h_state*weight(group_x,group_y).weight_preh_h)';
                    
                  
                    Error=weight(group_x,group_y).h_state(:,m)-test_data(:,group_x,group_y,m);
                    Error_Cost(1,iter)=sum(abs(Error));
                    
                    if(Error_Cost(1,iter)<cost_gate)
                        flag=1;
                        break;
                    else
                        [   weight(group_x,group_y).weight_input_x,...
                            weight(group_x,group_y).weight_input_h,...
                            weight(group_x,group_y).weight_inputgate_x,...
                            weight(group_x,group_y).weight_inputgate_c,...
                            weight(group_x,group_y).weight_forgetgate_x,...
                            weight(group_x,group_y).weight_forgetgate_c,...
                            weight(group_x,group_y).weight_outputgate_x,...
                            weight(group_x,group_y).weight_outputgate_c,...
                            weight(group_x,group_y).weight_preh_h ]=updata_weight(m,yita,Error,...
                            weight(group_x,group_y).weight_input_x,...
                            weight(group_x,group_y).weight_input_h,...
                            weight(group_x,group_y).weight_inputgate_x,...
                            weight(group_x,group_y).weight_inputgate_c,...
                            weight(group_x,group_y).weight_forgetgate_x,...
                            weight(group_x,group_y).weight_forgetgate_c,...
                            weight(group_x,group_y).weight_outputgate_x,...
                            weight(group_x,group_y).weight_outputgate_c,...
                            weight(group_x,group_y).weight_preh_h,...
                            weight(group_x,group_y).cell_state,...
                            weight(group_x,group_y).h_state,...
                            input_gate,forget_gate,...
                            output_gate,gate,...
                            train_data,pre_h_state,...
                            input_gate_input,...
                            output_gate_input,...
                            forget_gate_input,group_x,group_y);
                        
                    end
                end
                
            end
        end
    end


test_final = train_data(:,:,:,day);
test_output = test_data(:,:,:,day);
out=zeros(25,group_x_num,group_y_num,1);
out_final=zeros(25,group_x_num,group_y_num,1);
output_final=zeros(25,group_x_num,group_y_num,1);

for group_x = 3:group_x_num-2
    for group_y = 3:group_y_num-2
        %          m=m+1;
        gate=tanh(test_final(:,group_x,group_y,1)'*weight(group_x,group_y).weight_input_x+weight(group_x,group_y).h_state(:,m-1)'*weight(group_x,group_y).weight_input_h);
        input_gate_input=test_final(:,group_x,group_y,1)'*weight(group_x,group_y).weight_inputgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_inputgate_c+weight(group_x,group_y).bias_input_gate;
        forget_gate_input=test_final(:,group_x,group_y,1)'*weight(group_x,group_y).weight_forgetgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_forgetgate_c+weight(group_x,group_y).bias_forget_gate;
        output_gate_input=test_final(:,group_x,group_y,1)'*weight(group_x,group_y).weight_outputgate_x+weight(group_x,group_y).cell_state(:,m-1)'*weight(group_x,group_y).weight_outputgate_c+weight(group_x,group_y).bias_output_gate;
        for n=1:cell_num
            input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
            forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
            output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
        end
        cell_state_test=(input_gate.*gate+weight(group_x,group_y).cell_state(:,m-1,:,:)'.*forget_gate)';
        pre_h_state=tanh(cell_state_test').*output_gate;
        h_state_test=(pre_h_state*weight(group_x,group_y).weight_preh_h)';
        out(:,group_x,group_y)=h_state_test;
        test_output(:,group_x,group_y);
        output_final(:,group_x,group_y)=test_output(:,group_x,group_y);
    end
end
out=out.*500+mi;
output_final=output_final.*500+mi;
out(13,3,3)
output_final(13,3,3)

for group_x = 3:group_x_num-2
    for group_y = 3:group_y_num-2
        
        final_bias_input_gate = weight(group_x,group_y).bias_input_gate;
        final_bias_forget_gate = weight(group_x,group_y).bias_forget_gate;
        final_bias_output_gate = weight(group_x,group_y).bias_output_gate;
        
        final_weight_input_x = weight(group_x,group_y).weight_input_x;
        final_weight_input_h = weight(group_x,group_y).weight_input_h;
        final_weight_inputgate_x = weight(group_x,group_y).weight_inputgate_x;
        final_weight_inputgate_c = weight(group_x,group_y).weight_inputgate_c;
        final_weight_forgetgate_x = weight(group_x,group_y).weight_forgetgate_x;
        final_weight_forgetgate_c = weight(group_x,group_y).weight_forgetgate_c;
        final_weight_outputgate_x = weight(group_x,group_y).weight_outputgate_x;
        final_weight_outputgate_c = weight(group_x,group_y).weight_outputgate_c;
        
        final_weight_preh_h = weight(group_x,group_y).weight_preh_h;
        final_h_state = weight(group_x,group_y).h_state;
        final_cell_state = weight(group_x,group_y).cell_state;
        
        save_weight2(final_bias_input_gate,...
            final_bias_output_gate,...
            final_bias_forget_gate,...
            final_weight_input_x,...
            final_weight_input_h,...
            final_weight_inputgate_x,...
            final_weight_inputgate_c,...
            final_weight_forgetgate_x,...
            final_weight_forgetgate_c,...
            final_weight_outputgate_x,...
            final_weight_outputgate_c,...
            final_weight_preh_h,...
            final_h_state,final_cell_state,date_1,date_2);
    end
end

end