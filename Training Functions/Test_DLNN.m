%This script is to run the test for the trained DLNN file

clc;
clear;
close all;

load(['DLNN_SGD_tim.mat']); % Load the trained data
load('PD_data_4_4.mat'); % Load the PID input and output data

%input = [1 0 0 0;];
N= 2759;
f_val = zeros(2759,3); % Initialize matrix for storing the final predicted outputs
for j = 1:N
varIN = out.Input(j,:);
%input = reshape(input,inputs,1);
varIN= reshape(varIN(1,:),4,1);
input_of_hidden_layer1 = w1*varIN;
output_of_hidden_layer1 = jer_Swish(input_of_hidden_layer1);

input_of_hidden_layer2 = w2*output_of_hidden_layer1;
output_of_hidden_layer2 = jer_Swish(input_of_hidden_layer2);

%input_of_hidden_layer3 = w3*output_of_hidden_layer2;
%output_of_hidden_layer3 = jer_Swish(input_of_hidden_layer3);

input_of_output_node = w3*output_of_hidden_layer2;

%Final output should be close to output values that were specified
final_output = jer_LeakyReLU(input_of_output_node)';
f_val(j:j,1:3) = final_output;  % Store predicted output data
%plot(out.Output,final_output);
end

%Set parameters for graph
out_1 = out.Output(1:2759, 1:1); % <--- Identify columns and rows of PID output data to plot
in_1 = out.Input(1:2759, 1:1); % <--- Identify columns and rows of PID input data to plot
f_val_1 = f_val(1:2759, 1:1); % <--- Identify columns and rows of predicted output data to plot

out_2 = out.Output(1:2759, 2:2); % <--- Identify columns and rows of PID output data to plot
in_2 = out.Input(1:2759, 2:2); % <--- Identify columns and rows of PID input data to plot
f_val_2 = f_val(1:2759, 2:2); % <--- Identify columns and rows of predicted output data to plot

out_3 = out.Output(1:2759, 3:3); % <--- Identify columns and rows of PID output data to plot
in_3 = out.Input(1:2759, 3:3); % <--- Identify columns and rows of PID input data to plot
f_val_3 = f_val(1:2759, 3:3); % <--- Identify columns and rows of predicted output data to plot

no_of_samples=[1:2759 ]';

subplot(3,1,1);
figure(1)
plot(no_of_samples,out_1,'LineWidth',1.5);
hold on
plot(no_of_samples,f_val_1,'LineWidth',1.5);
hold off
legend('PD','Neural Network')
title('X Axis')
xlabel('Number of Samples') 
ylabel('Response') 

subplot(3,1,2);
%figure(2)
plot(no_of_samples,out_2,'LineWidth',1.5);
hold on
plot(no_of_samples,f_val_2,'LineWidth',1.5);
hold off
legend('PD','Neural Network')
title('Y Axis')
xlabel('Number of Samples') 
ylabel('Response') 

subplot(3,1,3);
%figure(3)
plot(no_of_samples,out_3,'LineWidth',1.5);
hold on
plot(no_of_samples,f_val_3,'LineWidth',1.5);
hold off
legend('PD','Neural Network')
title('Z Axis')
xlabel('Number of Samples') 
ylabel('Response') 
sgtitle('Adam Trained')

MSE = jer_MSE(out_1,f_val_1);
MSE2 = jer_MSE(out_2,f_val_2);
MSE3 = jer_MSE(out_3,f_val_3);

RMSE = sqrt(MSE);
RMSE2 = sqrt(MSE2);
RMSE3 = sqrt(MSE3);

MAE = jer_MAE(out_1,f_val_1);
MAE2 = jer_MAE(out_2,f_val_2);
MAE3 = jer_MAE(out_3,f_val_3);


avg_MSE = (MSE+MSE3+MSE2)/3;
fprintf('The Mean Square Error is: %.10f\n',avg_MSE)

avg_RMSE = (RMSE+RMSE3+RMSE2)/3;
fprintf('The Root Mean Square Error is: %.10f\n',avg_RMSE)

avg_MAE = (MAE+MAE3+MAE2)/3;
fprintf('The Mean Absolute Error is: %.10f\n',avg_MAE)


