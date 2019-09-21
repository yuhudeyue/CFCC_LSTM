function [   weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h ]=updata_weight(n,yita,Error,...
                                                   weight_input_x, weight_input_h, weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,...
                                                   cell_state,h_state,input_gate,forget_gate,output_gate,gate,train_data,pre_h_state,input_gate_input, output_gate_input,forget_gate_input,groupx,groupy)
%%% 权重更新函数
output_num=25;
weight_preh_h_temp=weight_preh_h;

%% 更新weight_preh_h权重x
delta_weight_preh_h_temp=(2*Error(:,1)*pre_h_state)';

weight_preh_h_temp=weight_preh_h_temp-yita*delta_weight_preh_h_temp;

%% 更新weight_outputgate_x
delta_weight_outputgate_x = zeros(size(train_data,1),size(weight_preh_h,1));
for num=1:output_num
    delta_weight_outputgate_x(:,:)=train_data(:,groupx,groupy,n)*((2*weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2));
    weight_outputgate_x=weight_outputgate_x-yita*delta_weight_outputgate_x;
end
%% 更新weight_inputgate_x
delta_weight_inputgate_x = zeros(size(train_data,1),size(weight_preh_h,1));
for num=1:output_num
delta_weight_inputgate_x(:,:)=train_data(:,groupx,groupy,n)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2));
weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end


if(n~=1)
    %% 更新weight_input_x
    delta_weight_input_x = zeros(size(train_data,1),size(weight_preh_h,1));
    temp=train_data(:,groupx,groupy,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    delta_weight_input_x(:,:)=train_data(:,groupx,groupy,n)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2)));

    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end
    %% 更新weight_forgetgate_x
    delta_weight_forgetgate_x = zeros(size(train_data,1),size(weight_preh_h,1));
    for num=1:output_num
    delta_weight_forgetgate_x(:,:)=train_data(:,groupx,groupy,n)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2));

    weight_forgetgate_x=weight_forgetgate_x-yita*delta_weight_forgetgate_x;
    end
    %% 更新weight_inputgate_c
    delta_weight_inputgate_c = zeros(size(cell_state,1),size(weight_preh_h,1));
    for num=1:output_num
    delta_weight_inputgate_c(:,:)=cell_state(:,n-1)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2));

    weight_inputgate_c=weight_inputgate_c-yita*delta_weight_inputgate_c;
    end
    %% 更新weight_forgetgate_c  123
    delta_weight_forgetgate_c = zeros(size(cell_state,1),size(weight_preh_h,1));
    for num=1:output_num
    
        delta_weight_forgetgate_c(:,:)=cell_state(:,n-1)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2));
    
    weight_forgetgate_c=weight_forgetgate_c-yita*delta_weight_forgetgate_c;
    end
    %% 更新weight_outputgate_c
    delta_weight_outputgate_c = zeros(size(cell_state,1),size(weight_preh_h,1));
    for num=1:output_num

    delta_weight_outputgate_c(:,:)=cell_state(:,n-1)*(2*(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2));

    weight_outputgate_c=weight_outputgate_c-yita*delta_weight_outputgate_c;
    end
    %% 更新weight_input_h
    
    temp=train_data(:,groupx,groupy,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    delta_weight_input_h = zeros(size(Error,1),size(weight_preh_h,1));
    for num=1:output_num
        delta_weight_input_h(:,:)=h_state(:,n-1)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2)));
    
    weight_input_h=weight_input_h-yita*delta_weight_input_h;
    end
else
    %% 更新weight_input_x
    temp=train_data(:,groupx,groupy,n)'*weight_input_x;
    delta_weight_input_x = zeros(size(train_data,1),size(weight_preh_h,1));
    for num=1:output_num
        delta_weight_input_x(:,:)=train_data(:,groupx,groupy,n)*(2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2)));

    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end
end
weight_preh_h=weight_preh_h_temp;
            weight_input_x(isnan(weight_input_x))=0.0001;
            weight_input_h(isnan(weight_input_h))=0.0001;
            weight_inputgate_x(isnan(weight_inputgate_x))=0.0001;
            weight_inputgate_c(isnan(weight_inputgate_c))=0.0001;
            weight_forgetgate_x(isnan(weight_forgetgate_x))=0.0001;
            weight_forgetgate_c(isnan(weight_forgetgate_c))=0.0001;
            weight_outputgate_x(isnan(weight_outputgate_x))=0.0001;
            weight_outputgate_c(isnan(weight_outputgate_c))=0.0001;
            weight_preh_h(isnan(weight_preh_h))=0.0001;
end