function out = test_lstm_day1(testdata,date_1,date_2,day,m)
testdata=double(int16(testdata));
M=5;
N=5;
[train_data,test_data]=data_process_day1(day,testdata);
group_x_num = 5;   %150
group_y_num = 5;   %450
input_num=75;
cell_num=14;
output_num=25;
jun=testdata(3,3,4);
mi=int16(jun)-250;
mi=double(mi);
ma=int16(jun)+250;
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
    h_state,cell_state] = load_weight2(['weight_' num2str(date_1) '_' num2str(date_2) '.mat']);
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
[weight(1:M,1:N).cell_state] = deal(cell_state);
test_final = train_data(:,:,:,day);
test_output = test_data(:,:,:,day);
out=zeros(25,group_x_num,group_y_num,1);
out_final=zeros(25,group_x_num,group_y_num,1);
output_final=zeros(25,group_x_num,group_y_num,1);
%前馈
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
        %         out(:,group_x,group_y)=out(:,group_x,group_y).*3248.1;
        %         output_final(:,group_x,group_y)=test_output(:,group_x,group_y).*3248.1;
        output_final(:,group_x,group_y)=test_output(:,group_x,group_y);
    end
end
out=out.*500+mi;
output_final=output_final.*500+mi;
out(13,3,3)
output_final(13,3,3)
end